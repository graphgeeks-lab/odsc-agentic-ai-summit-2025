"""
Tools for running RAG workflows on the FHIR graph database
- Graph-based RAG (Kuzu)
- Vector and FTS-based RAG (LanceDB)
"""

import asyncio
import os
from textwrap import dedent

import lancedb
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.rerankers import RRFReranker

import utils
from baml_client.async_client import b

load_dotenv()
os.environ["BAML_LOG"] = "WARN"
# Set embedding registry in LanceDB to use ollama
embedding_model = get_registry().get("ollama").create(name="nomic-embed-text")
kuzu_db_manager = utils.KuzuDatabaseManager("fhir_db.kuzu")

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")


async def prune_schema(question: str) -> str:

    schema = kuzu_db_manager.get_schema_dict
    schema_xml = kuzu_db_manager.get_schema_xml(schema)

    pruned_schema = await b.PruneSchema(schema_xml, question)

    pruned_schema_xml = kuzu_db_manager.get_schema_xml(pruned_schema.model_dump())

    print("Generated pruned schema XML")
    return pruned_schema_xml


async def answer_question(question: str, context: str) -> str:
    answer = await b.AnswerQuestion(question, context)
    
    return answer


async def execute_graph_rag(question: str, schema_xml: str, important_entities: str) -> str:
    response_cypher = await b.Text2Cypher(question, schema_xml, important_entities)
    
    if response_cypher.cypher:
        # Run the Cypher query on the graph database
        conn = kuzu_db_manager.get_connection()
        query = response_cypher.cypher
        response = conn.execute(query)
        result = response.get_as_pl().to_dicts()  # type: ignore
        print("Ran Cypher query")
    else:
        print("No Cypher query was generated from the given question and schema")
        result = ""
        query = ""
    
    context = dedent(
        f"""
        <CYPHER>
        {query}
        </CYPHER>

        <RESULT>
        {result}
        </RESULT>
        """
    )
    
    answer = await answer_question(question, context)
    return answer


async def execute_vector_and_fts_rag(
    question: str, schema_xml: str, important_entities: str, top_k: int = 2
) -> str:
    lancedb_table_name = "notes"
    lancedb_db_manager = await lancedb.connect_async("./fhir_lance_db")
    async_tbl = await lancedb_db_manager.open_table(lancedb_table_name)
    reranker = RRFReranker()
    
    if important_entities:
        response = await async_tbl.search(important_entities, query_type="hybrid")
        response_polars = (
            await response.rerank(reranker=reranker)
            .limit(top_k)
            .select(["record_id", "note"])
            .to_polars()
        )
        response_dicts = response_polars.to_dicts()
        context = " ".join([f"{row['note']}\n" for row in response_dicts])
        print("Generated vector context")
        
    else:
        print("[INFO]: No important entities found, skipping querying vector database...")
        context = ""
    
    return context


async def get_vector_context(question, pruned_schema_xml, important_entities, top_k=2):
    return await execute_vector_and_fts_rag(question, pruned_schema_xml, important_entities, top_k)


async def get_graph_answer(question, pruned_schema_xml, important_entities):
    return await execute_graph_rag(question, pruned_schema_xml, important_entities)


async def extract_entity_keywords(question: str, pruned_schema_xml: str):
    entities = await b.ExtractEntityKeywords(question, pruned_schema_xml)
    
    return entities


async def run_hybrid_rag(question: str) -> tuple[str, str]:
    print(f"---\nQ: {question}")
    
    pruned_schema_xml = await prune_schema(question)
    entities = await extract_entity_keywords(question, pruned_schema_xml)
    important_entities = " ".join(
        [f"{entity.key} {entity.value}".replace("_", " ") for entity in entities]
    )
    
    # Start both RAG tasks concurrently
    vector_context_task = asyncio.create_task(
        get_vector_context(question, pruned_schema_xml, important_entities)
    )
    graph_answer_task = asyncio.create_task(
        get_graph_answer(question, pruned_schema_xml, important_entities)
    )
    
    # As soon as vector context is ready, start answer generation
    vector_context = await vector_context_task
    vector_answer_task = asyncio.create_task(answer_question(question, vector_context))

    # Await both vector answer generation and graph answer generation before returning
    vector_answer, graph_answer = await asyncio.gather(vector_answer_task, graph_answer_task)
    
    return vector_answer, graph_answer


async def synthesize_answers(question: str, vector_answer: str, graph_answer: str) -> str:
    synthesized_answer = await b.SynthesizeAnswers(question, vector_answer, graph_answer)
        
    return synthesized_answer


async def main(question: str) -> None:
    vector_answer, graph_answer = await run_hybrid_rag(question)
    print(f"A1: {vector_answer}A2: {graph_answer}")
    synthesized_answer = await synthesize_answers(question, vector_answer, graph_answer)
    print(f"Final answer: {synthesized_answer}")
    


if __name__ == "__main__":
    questions = [
        "How many patients with the last name 'Rosenbaum' received multiple immunizations?",
        "What are the full names of the patients treated by the practitioner named Josef Klein?",
        "Did the practitioner 'Arla Fritsch' treat more than one patient?",
        "What are the unique categories of substances patients are allergic to?",
        "How many patients were born in between the years 1990 and 2000?",
        "How many patients have been immunized after January 1, 2022?",
        "Which practitioner treated the most patients? Return their full name and how many patients they treated.",
        "Is the patient ID 45 allergic to the substance 'shellfish'? If so, what city and state do they live in, and what is the full name of the practitioner who treated them?",
        "How many patients are immunized for influenza?",
        "How many substances cause allergies in the category 'food'?",
    ]
    for question in questions:
        asyncio.run(main(question))
