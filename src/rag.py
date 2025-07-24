"""
Tools for running RAG workflows on the FHIR graph database
- Graph-based RAG (Kuzu)
- Vector and FTS-based RAG (LanceDB)
"""

import asyncio
import os
from textwrap import dedent

import lancedb
import opik
from opik import opik_context
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.rerankers import RRFReranker

import utils
from baml_client.async_client import b
from baml_py import Collector

load_dotenv()
os.environ["BAML_LOG"] = "WARN"
# Set embedding registry in LanceDB to use ollama
embedding_model = get_registry().get("ollama").create(name="nomic-embed-text")
kuzu_db_manager = utils.KuzuDatabaseManager("fhir_db.kuzu")

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")


@opik.track(flush=True)
async def prune_schema(question: str) -> str:
    # Initialize BAML Collector
    collector = Collector(name="prune_schema_collector")
    function_logs = collector.logs
    assert len(function_logs) == 0

    schema = kuzu_db_manager.get_schema_dict
    schema_xml = kuzu_db_manager.get_schema_xml(schema)

    pruned_schema = await b.PruneSchema(schema_xml, question, baml_options={"collector": [collector]})

    # Now that we have the pruned schema, verify that the collector has a log
    function_logs = collector.logs
    assert len(function_logs) == 1

    # Verify that the last log is the one we expect
    log = collector.last
    assert log is not None
    assert log.log_type == "call"

    # Verify calls
    calls = log.calls
    assert len(calls) == 1

    # Get last call
    call = calls[0]

    # Verify timing fields
    call_timing = call.timing
    assert call_timing.start_time_utc_ms > 0
    assert call_timing.duration_ms is not None and call_timing.duration_ms > 0

    # Verify call usage
    call_usage = call.usage
    assert call_usage.input_tokens is not None and call_usage.input_tokens > 0
    assert call_usage.output_tokens is not None and call_usage.output_tokens > 0
    # AND it matches the log usage
    assert call_usage.input_tokens == log.usage.input_tokens
    assert call_usage.output_tokens == log.usage.output_tokens

    pruned_schema_xml = kuzu_db_manager.get_schema_xml(pruned_schema.model_dump())

    # Update opik context
    opik_context.update_current_span(
        name="pruned_schema",
        metadata={
            "function_name": log.function_name,
            "duration_ms": call_timing.duration_ms,
        },
        usage={
            "prompt_tokens": call_usage.input_tokens,
            "completion_tokens": call_usage.output_tokens,
            "total_tokens": call_usage.input_tokens + call_usage.output_tokens,
        },
        provider=call.provider,
        model=call.client_name,
    )

    print("Generated pruned schema XML")
    return pruned_schema_xml


@opik.track(flush=True)
async def answer_question(question: str, context: str) -> str:
    # Initialize BAML Collector
    collector = Collector(name="answer_question_collector")
    
    answer = await b.AnswerQuestion(question, context, baml_options={"collector": [collector]})
    
    # Update opik context with BAML data
    if collector.last is not None:
        log = collector.last
        call = log.calls[0] if log.calls else None
        
        if call and call.usage:
            opik_context.update_current_span(
                name="answer_question",
                metadata={
                    "function_name": log.function_name,
                    "duration_ms": call.timing.duration_ms if call.timing else None,
                },
                usage={
                    "prompt_tokens": call.usage.input_tokens,
                    "completion_tokens": call.usage.output_tokens,
                    "total_tokens": (call.usage.input_tokens or 0) + (call.usage.output_tokens or 0),
                },
                provider=call.provider,
                model=call.client_name,
            )
    
    return answer


@opik.track(flush=True)
async def execute_graph_rag(question: str, schema_xml: str, important_entities: str) -> str:
    # Initialize BAML Collector
    collector = Collector(name="execute_graph_rag_collector")
    
    response_cypher = await b.Text2Cypher(question, schema_xml, important_entities, baml_options={"collector": [collector]})
    
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
    
    # Update opik context with BAML data
    if collector.last is not None:
        log = collector.last
        call = log.calls[0] if log.calls else None
        
        if call and call.usage:
            opik_context.update_current_span(
                name="execute_graph_rag",
                metadata={
                    "function_name": log.function_name,
                    "duration_ms": call.timing.duration_ms if call.timing else None,
                    "cypher_generated": bool(response_cypher.cypher),
                    "result_count": len(result) if result else 0,
                },
                usage={
                    "prompt_tokens": call.usage.input_tokens,
                    "completion_tokens": call.usage.output_tokens,
                    "total_tokens": (call.usage.input_tokens or 0) + (call.usage.output_tokens or 0),
                },
                provider=call.provider,
                model=call.client_name,
            )
    
    answer = await answer_question(question, context)
    return answer


@opik.track(flush=True)
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
        
        # Update opik context with vector search data
        opik_context.update_current_span(
            name="execute_vector_and_fts_rag",
            metadata={
                "top_k": top_k,
                "entities_found": bool(important_entities),
                "results_count": len(response_dicts),
                "search_type": "hybrid",
            },
        )
    else:
        print("[INFO]: No important entities found, skipping querying vector database...")
        context = ""
        
        # Update opik context for skipped search
        opik_context.update_current_span(
            name="execute_vector_and_fts_rag",
            metadata={
                "top_k": top_k,
                "entities_found": False,
                "results_count": 0,
                "search_type": "skipped",
            },
        )
    
    return context


@opik.track(flush=True)
async def get_vector_context(question, pruned_schema_xml, important_entities, top_k=2):
    return await execute_vector_and_fts_rag(question, pruned_schema_xml, important_entities, top_k)


@opik.track(flush=True)
async def get_graph_answer(question, pruned_schema_xml, important_entities):
    return await execute_graph_rag(question, pruned_schema_xml, important_entities)


@opik.track(flush=True)
async def extract_entity_keywords(question: str, pruned_schema_xml: str):
    # Initialize BAML Collector
    collector = Collector(name="extract_entity_keywords_collector")
    
    entities = await b.ExtractEntityKeywords(question, pruned_schema_xml, baml_options={"collector": [collector]})
    
    # Update opik context with BAML data
    if collector.last is not None:
        log = collector.last
        call = log.calls[0] if log.calls else None
        
        if call and call.usage:
            opik_context.update_current_span(
                name="extract_entity_keywords",
                metadata={
                    "function_name": log.function_name,
                    "duration_ms": call.timing.duration_ms if call.timing else None,
                    "entities_extracted": len(entities),
                },
                usage={
                    "prompt_tokens": call.usage.input_tokens,
                    "completion_tokens": call.usage.output_tokens,
                    "total_tokens": (call.usage.input_tokens or 0) + (call.usage.output_tokens or 0),
                },
                provider=call.provider,
                model=call.client_name,
            )
    
    return entities


@opik.track(flush=True)
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
    
    # Update opik context with workflow summary
    opik_context.update_current_span(
        name="run_hybrid_rag",
        metadata={
            "question": question,
            "entities_extracted": len(entities),
            "vector_context_generated": bool(vector_context),
            "graph_answer_generated": bool(graph_answer),
        },
    )
    
    return vector_answer, graph_answer


@opik.track(flush=True)
async def synthesize_answers(question: str, vector_answer: str, graph_answer: str) -> str:
    # Initialize BAML Collector
    collector = Collector(name="synthesize_answers_collector")
    
    synthesized_answer = await b.SynthesizeAnswers(question, vector_answer, graph_answer, baml_options={"collector": [collector]})
    
    # Update opik context with BAML data
    if collector.last is not None:
        log = collector.last
        call = log.calls[0] if log.calls else None
        
        if call and call.usage:
            opik_context.update_current_span(
                name="synthesize_answers",
                metadata={
                    "function_name": log.function_name,
                    "duration_ms": call.timing.duration_ms if call.timing else None,
                },
                usage={
                    "prompt_tokens": call.usage.input_tokens,
                    "completion_tokens": call.usage.output_tokens,
                    "total_tokens": (call.usage.input_tokens or 0) + (call.usage.output_tokens or 0),
                },
                provider=call.provider,
                model=call.client_name,
            )
    
    return synthesized_answer


@opik.track(flush=True)
async def main(question: str) -> None:
    vector_answer, graph_answer = await run_hybrid_rag(question)
    print(f"A1: {vector_answer}A2: {graph_answer}")
    synthesized_answer = await synthesize_answers(question, vector_answer, graph_answer)
    print(f"Final answer: {synthesized_answer}")
    
    # Update opik context with main workflow summary
    opik_context.update_current_span(
        name="main",
        metadata={
            "question": question,
            "vector_answer_length": len(vector_answer),
            "graph_answer_length": len(graph_answer),
            "synthesized_answer_length": len(synthesized_answer),
        },
    )


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
