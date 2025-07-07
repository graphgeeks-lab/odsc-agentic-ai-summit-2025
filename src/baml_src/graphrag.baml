// --- Data models ---
class Query {
  cypher string @description("Valid Cypher query with no newlines")
}

class Node {
  label string
  properties Property[]
}

class Edge {
  label string @description("Relationship label")
  src string @alias("from") @description("Source node label")
  dst string @alias("to") @description("Destination node label")
  properties Property[]
}

class Property {
  name string
  type string @description("Data type of the property")
}

class GraphSchema {
    nodes Node[]
    edges Edge[]
}

// --- Functions ---

function CompressSchema(schema: string, question: string) -> GraphSchema {
  client OpenRouterGoogleGemini2Flash
  prompt #"
    Understand the given labelled property graph schema and the given user question. Your task
    is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
    relevant to the question.

    <INSTRUCTIONS>
    - The schema is a list of nodes and edges in a property graph.
    - The nodes are the entities in the graph.
    - The edges are the relationships between the nodes.
    - Properties of nodes and edges are their attributes, which helps answer the question.
    </INSTRUCTIONS>


    {{ _.role("user") }}
    <GRAPH_SCHEMA>
    {{ schema }}
    </GRAPH_SCHEMA>

    <QUESTION>
    {{ question }}
    </QUESTION>

    {{ ctx.output_format }}
  "#
}

function RAGText2Cypher(schema: string, question: string, additional_context: string | null) -> Query {
  client OpenRouterGoogleGemini2Flash
  prompt #"
    Translate the given question into a valid Cypher query that respects the given graph schema.

    <INSTRUCTIONS>
    - ALWAYS respect the relationship directions (from --> to) as provided in the <structure> section.
    - Unless otherwise specified, always limit the results to 10 rows.
    - Use only the provided nodes, relationships and properties in your Cypher statement.
    - Properties can be on nodes or relationships - check the schema carefully to figure out where they are.
    - When returning results, return property values rather than the entire node or relationship.
    - When comparing string properties, ALWAYS do the following:
      - Lowercase the property values before comparison
      - Use the WHERE clause
      - Use the CONTAINS operator to check for presence of one substring in the other
    - When non-empty, pay attention to the ADDITIONAL_CONTEXT to figure out what to add in the
    WHERE clause.
    - DO NOT use APOC as the database does not support it.
    - For datetime queries, the database supports the TIMESTAMP type, which combines the date and time.
    </INSTRUCTIONS>

    {{ _.role("user") }}
    <QUESTION>
    {{ question }}
    </QUESTION>

    <ADDITIONAL_CONTEXT>
    {{ additional_context }}
    <ADDITIONAL_CONTEXT>

    <SCHEMA>
    {{ schema }}
    </SCHEMA>

    <OUTPUT_FORMAT>
    {{ ctx.output_format }}
    </OUTPUT_FORMAT>
  "#
}

function RAGAnswerQuestion(question: string, cypher: string, context: string) -> string {
  client OpenRouterGoogleGemini2Flash
  prompt #"
    You are an AI assistant for Retrieval-Augmented Generation (RAG).

    <INSTRUCTIONS>
    - Use the provided question, the generated Cypher query and the CONTEXT to answer the question.
    - If the CONTEXT is empty, state that you don't have enough information to answer the question.
    </INSTRUCTIONS>

    {{ _.role("user") }}
    <QUESTION>
    {{ question }}
    </QUESTION>
    
    <CYPHER>
    {{ cypher }}
    </CYPHER>

    <CONTEXT>
    {{ context }}
    </CONTEXT>

    {{ ctx.output_format }}

    RESPONSE:
  "#
}


// --- Test cases ---

test CompressSchema1 {
  functions [CompressSchema]
  args {
    schema #"
    <structure>
      <rel label="CAUSES" from="Substance" to="Allergy" />
      <rel label="LIVES_IN" from="Patient" to="Address" />
      <rel label="TREATS" from="Practitioner" to="Patient" />
      <rel label="HAS_IMMUNIZATION" from="Patient" to="Immunization" />
      <rel label="EXPERIENCES" from="Patient" to="Allergy" />
    </structure>
    <nodes>
    <node label="Address">
        <property name="id" type="STRING" />
        <property name="street" type="STRING" />
        <property name="city" type="STRING" />
        <property name="state" type="STRING" />
        <property name="postalCode" type="STRING" />
        <property name="country" type="STRING" />
    </node>
    <node label="Substance">
        <property name="name" type="STRING" />
    </node>
    <node label="Immunization">
        <property name="id" type="STRING" />
        <property name="status" type="STRING" />
        <property name="occurrenceDateTime" type="STRING" />
        <property name="traits" type="STRING" />
    </node>
    <node label="Practitioner">
        <property name="id" type="STRING" />
        <property name="surname" type="STRING" />
        <property name="givenName" type="STRING" />
        <property name="address" type="STRING" />
        <property name="phone" type="STRING" />
        <property name="email" type="STRING" />
    </node>
    <node label="Allergy">
        <property name="id" type="STRING" />
        <property name="category" type="STRING" />
        <property name="manifestation" type="STRING" />
    </node>
    <node label="Patient">
        <property name="patient_id" type="INT64" />
        <property name="prefix" type="STRING" />
        <property name="gender_inferred" type="STRING" />
        <property name="surname" type="STRING" />
        <property name="givenName" type="STRING" />
        <property name="birthDate" type="STRING" />
        <property name="phone" type="STRING" />
        <property name="email" type="STRING" />
        <property name="maritalStatus" type="STRING" />
        <property name="primaryLanguage" type="STRING" />
      </node>
    </nodes>
    <relationships>
      <rel label="CAUSES" />
      <rel label="LIVES_IN" />
      <rel label="TREATS" />
      <rel label="HAS_IMMUNIZATION" />
      <rel label="EXPERIENCES" />
    </relationships>
    "#
    question "How many patients received more than one immunization?"
  }
}

test RAGText2Cypher1 {
  functions [RAGText2Cypher]
  args {
    schema #"
    <structure>
        <rel label="HAS_IMMUNIZATION" from="Patient" to="Immunization" />
    </structure>
    <nodes>
    <node label="Patient">
        <property name="patient_id" type="INT64" />
    </node>
    <node label="Immunization">
        <property name="id" type="STRING" />
    </node>
    </nodes>
    <relationships>
        <rel label="HAS_IMMUNIZATION" />
    </relationships>
    "#
    question "How many patients received more than one immunization?"
    additional_context ""
  }
}

test RAGAnswerQuestion1 {
  functions [RAGAnswerQuestion]
  args {
    question "How many patients received more than one immunization?"
    cypher "MATCH (p:Patient)-[r:HAS_IMMUNIZATION]->(i:Immunization) WITH p, count(i) AS immunizationCount WHERE immunizationCount > 1 RETURN count(DISTINCT p)"
    context ""
  }
}