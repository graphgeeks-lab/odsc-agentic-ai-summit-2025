"""
This script builds the graph database using Kuzu using the FHIR JSON data extracted by BAML.
"""

import shutil

import kuzu
import polars as pl


def setup_db() -> kuzu.Connection:
    shutil.rmtree(DB_NAME, ignore_errors=True)
    db = kuzu.Database(DB_NAME)
    conn = kuzu.Connection(db)
    # -- Nodes --
    conn.execute(
        "CREATE NODE TABLE Allergy(id STRING PRIMARY KEY, category STRING, manifestation STRING)"
    )
    conn.execute("CREATE NODE TABLE Substance(name STRING PRIMARY KEY)")
    conn.execute(
        """
        CREATE NODE TABLE Patient(
            patient_id INT64 PRIMARY KEY,
            prefix STRING,
            gender_inferred STRING,
            surname STRING,
            givenName STRING,
            birthDate DATE,
            phone STRING,
            email STRING,
            maritalStatus STRING,
            primaryLanguage STRING
        )
        """
    )
    conn.execute(
        """
        CREATE NODE TABLE Practitioner(
            id STRING PRIMARY KEY,
            surname STRING,
            givenName STRING,
            address STRING,
            phone STRING,
            email STRING
        )
        """
    )
    conn.execute(
        """
        CREATE NODE TABLE Immunization(
            id STRING PRIMARY KEY,
            status STRING,
            occurrenceDateTime TIMESTAMP,
            traits STRING
        )
        """
    )
    conn.execute(
        """
        CREATE NODE TABLE Address(
            id STRING PRIMARY KEY,
            street STRING,
            city STRING,
            state STRING,
            postalCode STRING,
            country STRING
        )
        """
    )
    # Relationships
    conn.execute("CREATE REL TABLE LIVES_IN(FROM Patient TO Address)")
    conn.execute("CREATE REL TABLE EXPERIENCES(FROM Patient TO Allergy)")
    conn.execute("CREATE REL TABLE HAS_IMMUNIZATION(FROM Patient TO Immunization)")
    conn.execute("CREATE REL TABLE TREATS(FROM Practitioner TO Patient)")
    conn.execute("CREATE REL TABLE CAUSES(FROM Substance TO Allergy)")
    return conn


def prep_address_df(df: pl.DataFrame) -> pl.DataFrame:
    df_address = df.select("record_id", "address").unnest("address")
    df_address = df_address.with_columns(
        pl.concat_str([pl.col("line"), pl.col("postalCode")], separator="_")
        .str.to_lowercase()
        .str.replace(r"\\.", "")
        .alias("id"),
        pl.col("line").alias("street"),
        pl.col("city"),
        pl.col("state"),
        pl.col("postalCode"),
        pl.col("country"),
    ).drop("line")
    return df_address


def prep_patient_df(df: pl.DataFrame) -> pl.DataFrame:
    df_patient = df.with_columns(
        pl.col("record_id").alias("patient_id"),
        pl.col("name").struct.field("prefix"),
        pl.col("name").struct.field("family").alias("surname"),
        pl.col("name").struct.field("given").list.join(separator=" ").alias("givenName"),
        pl.col("birthDate").alias("birthDate"),
        pl.col("phone"),
        pl.col("email"),
        pl.col("maritalStatus"),
        pl.col("primaryLanguage"),
    ).drop("address", "practitioner", "immunization", "allergy", "name", "record_id")
    return df_patient


def prep_practitioner_df(df: pl.DataFrame) -> pl.DataFrame:
    df_practitioner = df.select("record_id", "practitioner").unnest("practitioner")
    df_practitioner = df_practitioner.with_columns(
        pl.concat_str(
            [
                pl.col("name").struct.field("prefix").str.to_lowercase(),
                pl.col("name").struct.field("given").list.join(separator="_").str.to_lowercase(),
                pl.col("name").struct.field("family").str.to_lowercase(),
            ],
            separator="_",
        )
        .str.replace(r"\\.", "")
        .alias("id"),
        pl.col("name").struct.field("given").list.join(separator="").alias("givenName"),
    )
    return df_practitioner


def prep_substance_df(df: pl.DataFrame) -> pl.DataFrame:
    df_substance = (
        df.select("record_id", "allergy")
        .unnest("allergy")
        .filter(pl.col("substance").is_not_null())
    )
    df_substance = (
        df_substance.explode("substance")
        .with_columns(
            pl.col("substance")
            .struct.field("manifestation")
            .list.join(separator=", ")
            .alias("manifestation"),
            pl.concat_str(
                [
                    pl.col("record_id"),
                    pl.coalesce(
                        pl.col("substance").struct.field("category"), pl.lit("unknown")
                    ).str.to_lowercase(),
                    pl.coalesce(
                        pl.col("substance").struct.field("name"), pl.lit("unknown")
                    ).str.to_lowercase(),
                ],
                separator="_",
            )
            .str.replace(r"\\.", "")
            .alias("id"),
        )
        .select(
            pl.col("record_id"),
            pl.col("id"),
            pl.col("substance").struct.field("name").str.to_lowercase().alias("name"),
            pl.col("substance").struct.field("category").str.to_lowercase().alias("category"),
            pl.col("manifestation").str.to_lowercase().alias("manifestation"),
        )
    )
    return df_substance


def prep_immunization_df(df: pl.DataFrame) -> pl.DataFrame:
    df_immunization = (
        df.select("record_id", "immunization")
        .explode("immunization")
        .unnest("immunization")
        .with_columns(
            pl.col("traits").list.join(separator=", ").alias("traits"),
            pl.col("occurrenceDateTime"),
            pl.concat_str(
                [
                    pl.col("record_id"),
                    pl.coalesce(pl.col("status"), pl.lit("unknown")).str.to_lowercase(),
                ],
                separator="_",
            ).alias("id"),
        )
        # Only filter out rows where ALL values related to immunization are null
        .filter(~pl.all_horizontal(pl.col(["status", "occurrenceDateTime", "traits"]).is_null()))
        .select(
            pl.col("record_id"),
            pl.col("id"),
            pl.col("status").str.to_lowercase(),
            pl.col("occurrenceDateTime"),
            pl.col("traits").str.to_lowercase(),
        )
    )
    return df_immunization


def ingest_address_nodes(conn: kuzu.Connection, df_address: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_address
        WHERE id IS NOT NULL
        WITH DISTINCT id, street, city, state, postalCode, country
        MERGE (a:Address {id: id})
        SET a.street = street,
            a.city = city,
            a.state = state,
            a.postalCode = postalCode,
            a.country = country
        RETURN COUNT(*) AS num_addresses
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_patient_nodes(conn: kuzu.Connection, df_patient: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_patient
        WITH *,
        // Infer gender from gender field or prefix
        CASE
            WHEN gender = "male" OR gender = "Male" THEN "M"
            WHEN gender = "female" OR gender = "Female" THEN "F"
            WHEN prefix = "Mr." THEN "M"
            WHEN prefix = "Mrs." OR prefix = "Ms." THEN "F"
            ELSE NULL
        END AS gender_inferred
        // Merge patient node
        MERGE (p:Patient {patient_id: patient_id})
        SET p.prefix = prefix,
            p.gender_inferred = gender_inferred,
            p.surname = surname,
            p.givenName = givenName,
            p.birthDate = CAST(birthDate AS DATE),
            p.phone = phone,
            p.email = email,
            p.maritalStatus = maritalStatus,
            p.primaryLanguage = primaryLanguage
        RETURN COUNT(*) AS num_patients
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_lives_in(conn: kuzu.Connection, df_address: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_address
        WHERE id IS NOT NULL AND record_id IS NOT NULL
        WITH DISTINCT record_id AS patient_id, id
        MATCH (p:Patient {patient_id: patient_id}), (a:Address {id: id})
        MERGE (p)-[:LIVES_IN]->(a)
        RETURN COUNT(*) AS num_lives_in
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_practitioner_nodes(conn: kuzu.Connection, df_practitioner: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_practitioner
        WHERE id IS NOT NULL
        WITH DISTINCT id, name.family AS surname, givenName, address, phone, email
        MERGE (p:Practitioner {id: id})
        SET p.surname = surname,
          p.givenName = givenName,
          p.address = address,
          p.phone = phone,
          p.email = email
        RETURN COUNT(*) AS num_practitioners
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_treats(conn: kuzu.Connection, df_practitioner: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_practitioner
        WITH DISTINCT record_id AS patient_id, id
        MATCH (p1:Patient {patient_id: patient_id}), (p2:Practitioner {id: id})
        MERGE (p2)-[:TREATS]->(p1)
        RETURN COUNT(*) AS num_treatments
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_substance_nodes(conn: kuzu.Connection, df_substance: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_substance
        WHERE name IS NOT NULL
        WITH DISTINCT name
        MERGE (s:Substance {name: name})
        RETURN COUNT(*) AS num_substances
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_allergy_nodes(conn: kuzu.Connection, df_substance: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_substance
        WHERE id IS NOT NULL
        MERGE (a:Allergy {id: id})
        SET a.category = category,
            a.manifestation = manifestation
        RETURN COUNT(*) AS num_allergies
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_immunization_nodes(conn: kuzu.Connection, df_immunization: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_immunization
        MERGE (i:Immunization {id: id})
        SET i.status = status,
            i.occurrenceDateTime = CAST(occurrenceDateTime AS TIMESTAMP),
            i.traits = traits
        RETURN COUNT(*) AS num_immunizations
        """
    )
    print(res.get_as_pl())  # type: ignore  # type: ignore


def ingest_experiences_allergy(conn: kuzu.Connection, df_substance: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_substance
        WITH DISTINCT record_id AS patient_id, id
        MATCH (p:Patient {patient_id: patient_id}), (a:Allergy {id: id})
        MERGE (p)-[:EXPERIENCES]->(a)
        RETURN COUNT(*) AS num_allergies
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_causes_allergy(conn: kuzu.Connection, df_substance: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_substance
        WHERE name IS NOT NULL
        MATCH (s:Substance {name: name}), (a:Allergy {id: id})
        MERGE (s)-[:CAUSES]->(a)
        RETURN COUNT(*) AS num_causes_allergy
        """
    )
    print(res.get_as_pl())  # type: ignore


def ingest_has_immunization(conn: kuzu.Connection, df_immunization: pl.DataFrame) -> None:
    res = conn.execute(
        """
        LOAD FROM df_immunization
        MATCH (p:Patient {patient_id: record_id}), (i:Immunization {id: id})
        MERGE (p)-[:HAS_IMMUNIZATION]->(i)
        RETURN COUNT(*) AS num_has_immunizations
        """
    )
    print(res.get_as_pl())  # type: ignore


def main(data_path: str) -> None:
    conn = setup_db()
    df = pl.read_json(DATA_PATH)
    # Prepare DataFrames
    df_address = prep_address_df(df)
    df_patient = prep_patient_df(df)
    df_practitioner = prep_practitioner_df(df)
    df_substance = prep_substance_df(df)
    df_immunization = prep_immunization_df(df)
    # Ingest nodes
    ingest_address_nodes(conn, df_address)
    ingest_patient_nodes(conn, df_patient)
    ingest_practitioner_nodes(conn, df_practitioner)
    ingest_substance_nodes(conn, df_substance)
    ingest_allergy_nodes(conn, df_substance)
    ingest_immunization_nodes(conn, df_immunization)
    # Ingest relationships
    ingest_lives_in(conn, df_address)
    ingest_treats(conn, df_practitioner)
    ingest_experiences_allergy(conn, df_substance)
    ingest_causes_allergy(conn, df_substance)
    ingest_has_immunization(conn, df_immunization)


if __name__ == "__main__":
    DB_NAME = "fhir_kuzu_db"
    DATA_PATH = "../data/results/extracted_fhir_1_200.json"
    main(DATA_PATH)
