"""
Use ollama to generate embeddings for the notes in the FHIR graph database.

The embeddings are persisted to LanceDB, an embedded vector database.
"""

import lancedb
import polars as pl
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# Create embedding registry
registry = get_registry()
model = registry.get("ollama").create(name="nomic-embed-text")


class Note(LanceModel):
    record_id: int
    prefix: str | None = None
    surname: str | None = None
    given_name: str | None = None
    note: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()  # type: ignore


def main(db_path: str, limit: int = 10000) -> None:
    NOTES_DATA_PATH = "../data/note.json"
    RECORDS_DATA_PATH = "../data/extracted_fhir.json"
    TABLE_NAME = "notes"

    db = lancedb.connect(db_path)
    table = db.create_table(TABLE_NAME, schema=Note, mode="overwrite")
    df_patients = pl.read_json(NOTES_DATA_PATH).head(limit)
    df_records = pl.read_json(RECORDS_DATA_PATH).head(limit)

    # Join on the record_id column to get metadata for the note
    df = df_patients.join(df_records, on="record_id").select(
        "record_id",
        pl.col("name").struct.field("prefix").alias("prefix"),
        pl.col("name").struct.field("family").alias("surname"),
        pl.col("name").struct.field("given").list.join(" ").alias("given_name"),
        "note",
    )
    # Add DataFrame, compute embeddings and persist them all to LanceDB
    table.add(df)
    print(f"Created table with {len(table)} rows")

    # Generate FTS index in LanceDB
    table.create_fts_index("note", replace=True)
    print(f"Finished creating FTS and vector indices for '{TABLE_NAME}' table")


if __name__ == "__main__":
    DB_PATH = "./fhir_lance_db"
    LIMIT = 10000
    main(DB_PATH, LIMIT)

    query = "Did Ms. Sonia María Bañuelos receive the influenza vaccine?"
    db = lancedb.connect(DB_PATH)
    table = db.open_table("notes")
    result = table.search(query, query_type="vector").limit(5).to_pydantic(Note)[0]
    print(result)
