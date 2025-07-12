// --- Data models for Graph RAG---
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

// Data models for vector and FTS RAG

class EntityKeyword {
  key string  @description("Type of the entity from the graph schema")
  value string @description("Value of the entity from the graph schema")
}

// --- Functions ---

function PruneSchema(schema: string, question: string) -> GraphSchema {
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

function Text2Cypher(question: string, schema: string, important_terms: string | null) -> Query {
  client OpenRouterGoogleGemini2Flash
  prompt #"
    Translate the given question into a valid Cypher query that respects the given graph schema.

    <INSTRUCTIONS>
    - Pay special attention to the <IMPORTANT_TERMS> and attempt to include them in the Cypher query.
    - ALWAYS respect the relationship directions (from --> to) as provided in the <structure>.
    - Unless otherwise specified, ALWAYS limit the results to 10 rows.
    - Use only the provided nodes, relationships and properties in your Cypher statement.
    - When returning results, return property values rather than the entire node or relationship.
    - When comparing string properties, ALWAYS do the following:
      - Lowercase the property values before comparison
      - Use the WHERE clause
      - Use the CONTAINS operator to check for presence of one substring in the other
    - DO NOT use APOC as the database does not support it.
    - For datetime queries, the database supports the TIMESTAMP type, which combines the date and time.
    </INSTRUCTIONS>

    {{ _.role("user") }}

    <QUESTION>
    {{ question }}
    </QUESTION>

    <IMPORTANT_TERMS>
    {{ important_terms }}
    </IMPORTANT_TERMS>

    <SCHEMA>
    {{ schema }}
    </SCHEMA>

    <OUTPUT_FORMAT>
    {{ ctx.output_format }}
    </OUTPUT_FORMAT>
  "#
}

function AnswerQuestion(question: string, context: string) -> string {
  client OpenRouterGoogleGemini2FlashGenerate
  prompt #"
    You are an AI assistant for Retrieval-Augmented Generation (RAG).

    <INSTRUCTIONS>
    - Use the provided question, the generated Cypher query and the CONTEXT to answer the question.
    - Only use the information provided in the CONTEXT to answer the question. Do not make up any information.
    - If the CONTEXT is empty, state that you don't have enough information to answer the question.
    </INSTRUCTIONS>

    {{ _.role("user") }}
    <QUESTION>
    {{ question }}
    </QUESTION>

    <CONTEXT>
    {{ context }}
    </CONTEXT>

    {{ ctx.output_format }}

    RESPONSE:
  "#
}

function ExtractEntityKeywords(question: string, schema: string) -> EntityKeyword[] {
  client OpenRouterGoogleGemini2Flash
  prompt #"
    For the user's question, extract important keywords that can help search for relevant information.
    - Focus on keywords that have high signal, for e.g., person full names, cities, conditions, etc.
    - Use the graph schema to map the entities to the nodes or relationships in the graph.
    - Ensure that the keys that are mapped are actually present in the graph schema.

    {{ _.role("user")}}

    {{question}}

    {{ ctx.output_format }}
  "#
}

function SynthesizeAnswers(question: string, vector_answer: string, graph_answer: string) -> string {
  client OpenRouterGoogleGemini2FlashGenerate
  prompt #"
    Synthesize the provided answers from Graph RAG and Vector RAG into one single, coherent answer.

    - If there is conflicting information between the two answers, provide both answers and state
    the source.
    - One of the answers may not have sufficient information to answer the question. In such cases,
    use the answer that has the information to carry out your task.
    - For questions involving counting or numerical values, beware that vector RAG is limited to only the
    top 2 results. If both answers have numbers, only return the number from Graph RAG.
    - If neither answer has sufficient information to answer the question,state that you don't have
    enough information to answer the question.

    <QUESTION>
    {{ question }}
    </QUESTION>

    {{ _.role("user") }}

    <VECTOR_ANSWER>
    {{ vector_answer }}
    </VECTOR_ANSWER>

    <GRAPH_ANSWER>
    {{ graph_answer }}
    </GRAPH_ANSWER>

    FINAL ANSWER:
  "#
}

// --- Test cases ---

test PruneSchema1 {
  functions [PruneSchema]
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
    question "How many patients are allergic to the substance 'seafood'?"
  }
}

test Text2Cypher1 {
  functions [Text2Cypher]
  args {
    schema #"
    <structure>
        <rel label="TREATS" from="Practitioner" to="Patient" />
    </structure>
    <nodes>
    <node label="Patient">
        <property name="patient_id" type="INT64" />
    </node>
    <node label="Practitioner">
        <property name="id" type="STRING" />
        <property name="surname" type="STRING" />
        <property name="givenName" type="STRING" />
    </node>
    </nodes>
    <relationships>
        <rel label="TREATS" />
    </relationships>
    "#
    question "Did the practitioner 'Arla Fritsch' treat multiple patients?"
  }
}


test AnswerQuestion1 {
  functions [AnswerQuestion]
  args {
    question "How many patients are allergic to the substance 'seafood'?"
    context #"
    <CYPHER_QUERY>
    MATCH (s:Substance)-[:CAUSES]->(a:Allergy)<-[:EXPERIENCES]-(p:Patient) WHERE toLower(s.name) CONTAINS toLower('seafood') RETURN count(DISTINCT p)
    </CYPHER_QUERY>

    <RESULT>
    [{'COUNT(DISTINCT p._ID)': 19}]
    </RESULT>
    "#
  }
}

test ExtractEntityKeywords1 {
  functions [ExtractEntityKeywords]
  args {
    question "How many patients are allergic to the substance 'seafood'?"
    schema #"
    <structure>
        <rel label="EXPERIENCES" from="Patient" to="Allergy" />
        <rel label="CAUSES" from="Substance" to="Allergy" />
    </structure>
    <nodes>
    <node label="Patient">
        <property name="patient_id" type="INT64" />
    </node>
    <node label="Allergy">
        <property name="id" type="STRING" />
    </node>
    <node label="Substance">
        <property name="name" type="STRING" />
    </node>
    </nodes>
    <relationships>
        <rel label="EXPERIENCES" />
        <rel label="CAUSES" />
    </relationships>
    "#
    }
}

test ExtractEntityKeywords2 {
  functions [ExtractEntityKeywords]
  args {
    question "Did the practitioner 'Arla Fritsch' treat multiple patients?"
    schema #"
    <structure>
        <rel label="TREATS" from="Practitioner" to="Patient" />
    </structure>
    <nodes>
    <node label="Patient">
        <property name="patient_id" type="INT64" />
    </node>
    <node label="Practitioner">
        <property name="id" type="STRING" />
        <property name="surname" type="STRING" />
        <property name="givenName" type="STRING" />
    </node>
    </nodes>
    <relationships>
        <rel label="TREATS" />
    </relationships>
    "#
    }
}

test SynthesizeAnswers1 {
  functions [SynthesizeAnswers]
  args {
    question "To what substances is Ms. Lachelle Adriane Huels allergic to?"
    vector_answer "Ms. Lachelle Adriane Huels is allergic to House dust mite (organism). The severity of the allergy is low, and it was recorded on June 13, 2003."
    graph_answer "Based on the provided information, it's not clear whether Ms. Lachelle Adriane Huels has any allergies."
  }
}

test SynthesizeAnswers2 {
  functions [SynthesizeAnswers]
  args {
    question "To what substances is Ms. Lily Abbie Brekke allergic to?"
    vector_answer "Based on the provided information, it's not clear whether Ms. Lily Abbie Brekke has any allergies."
    graph_answer ""
  }
}

test SynthesizeAnswers3 {
  functions [SynthesizeAnswers]
  args {
    question "Did the practitioner 'Arla Fritsch' treat more than one patient?"
    vector_answer "The practitioner 'Arla Fritsch' treated 4 patients."
    graph_answer "The practitioner 'Arla Fritsch' treated 3 patients."
  }
}