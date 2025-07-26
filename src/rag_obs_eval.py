"""
Observability & evaluation tools for RAG workflows on the FHIR graph database
- Graph-based RAG (Kuzu)
- Vector and FTS-based RAG (LanceDB)
- Evaluation and observability using Opik

The observability tool used is Opik, by Comet:
https://www.comet.com/site/products/opik/
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
from openai import OpenAI
from opik.integrations.openai import track_openai

import utils
from baml_client.async_client import b
from baml_instrumentation import BAMLInstrumentation, track_baml_call, run_post_call_metrics

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPIK_API_KEY = os.environ.get("OPIK_API_KEY")
OPIK_WORKSPACE = os.environ.get("OPIK_WORKSPACE")
OPIK_PROJECT_NAME = "ODSC-RAG"

# Set a reasonable sample rate for metrics to avoid overwhelming the system
os.environ["METRICS_SAMPLE_RATE"] = "0.05"  # 5% of calls will run metrics

# Configure BAML logging
os.environ["BAML_LOG"] = "WARN"

# Set embedding registry in LanceDB to use ollama
embedding_model = get_registry().get("ollama").create(name="nomic-embed-text")
kuzu_db_manager = utils.KuzuDatabaseManager("fhir_db.kuzu")

# Initialize the OpenAI client with OpenRouter base URL
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
client = track_openai(client)

# Optional headers for OpenRouter leaderboard
headers = {
    "HTTP-Referer": "graphgeeks.org",  # Optional. Site URL for rankings
    "X-Title": "GraphGeeks",  # Optional. Site title for rankings
}

# Configure Opik
if OPIK_API_KEY and OPIK_WORKSPACE:
    os.environ["OPIK_API_KEY"] = OPIK_API_KEY
    os.environ["OPIK_WORKSPACE"] = OPIK_WORKSPACE
    os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT_NAME
    
    # Set OpenRouter API key for Opik metrics (LLM as a Judge)
    if OPENROUTER_API_KEY:
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    else:
        print("Warning: OPENROUTER_API_KEY not set. Opik metrics may fail.")
    
    # Configure Opik for cloud usage
    opik.configure(use_local=False)
    print("Opik configured for cloud tracking")
else:
    print(
        "Please set the OPIK_API_KEY and OPIK_WORKSPACE environment variables to enable opik tracking"
    )
    # Disable Opik tracking if credentials are not provided
    opik.configure(use_local=True)
    print("Opik configured for local tracking (no cloud credentials)")


# Core RAG Functions
@opik.track(flush=True)
async def prune_schema(question: str) -> str:
    schema = kuzu_db_manager.get_schema_dict
    schema_xml = kuzu_db_manager.get_schema_xml(schema)

    pruned_schema = await track_baml_call(
        b.PruneSchema,
        "prune_schema_collector",
        "pruned_schema",
        schema_xml,
        question
    )

    pruned_schema_xml = kuzu_db_manager.get_schema_xml(pruned_schema.model_dump())
    print("Generated pruned schema XML")
    return pruned_schema_xml


@opik.track(flush=True)
async def answer_question(question: str, context: str) -> str:
    answer = await track_baml_call(
        b.AnswerQuestion,
        "answer_question_collector",
        "answer_question",
        question,
        context
    )
    return answer


@opik.track(flush=True)
async def execute_graph_rag(question: str, schema_xml: str, important_entities: str) -> str:
    response_cypher = await track_baml_call(
        b.Text2Cypher,
        "execute_graph_rag_collector",
        "execute_graph_rag",
        question,
        schema_xml,
        important_entities
    )
    
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
    
    # Update opik context with additional metadata
    opik_context.update_current_span(
        name="execute_graph_rag",
        metadata={
            "cypher_generated": bool(response_cypher.cypher),
            "cypher": response_cypher.cypher,
            "result_count": len(result) if result else 0,
        }
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
    entities = await track_baml_call(
        b.ExtractEntityKeywords,
        "extract_entity_keywords_collector",
        "extract_entity_keywords",
        question,
        pruned_schema_xml,
        additional_metadata={"entities_extracted": lambda: len(entities)}
    )

    # Convert entities to a string representation for metrics
    entities_str = "\n".join([f"- key: {entity.key}\n  value: {entity.value}" for entity in entities])
    
    # Use Opik Contains metric to check if the question contains the extracted entities
    await run_post_call_metrics(
        "extract_entity_keywords_collector",
        "extract_entity_keywords",
        input=question,
        output=question,  # The question is what we're checking
        context=[pruned_schema_xml],
        metrics=[
            {"type": "Contains", "params": {"output": question, "reference": entities_str}}
        ]
    )

    return entities


@opik.track(flush=True)
async def run_hybrid_rag(question: str, question_number: int = None) -> tuple[str, str]:
    print(f"---\nQuestion {question_number}: {question}")
    
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
    # Simple manual comparison of vector and graph answers
    if vector_answer and graph_answer:
        # Basic consistency check
        vector_words = set(vector_answer.lower().split())
        graph_words = set(graph_answer.lower().split())
        common_words = vector_words.intersection(graph_words)
        similarity = len(common_words) / max(len(vector_words), len(graph_words)) if max(len(vector_words), len(graph_words)) > 0 else 0
        
        print(f"[INFO] Simple similarity score: {similarity:.3f}")
        
        # Update Opik context with simple comparison
        opik_context.update_current_span(
            name="simple_answer_comparison",
            metadata={
                "vector_answer_length": len(vector_answer),
                "graph_answer_length": len(graph_answer),
                "simple_similarity_score": similarity,
                "common_words_count": len(common_words),
            }
        )
    
    synthesized_answer = await track_baml_call(
        b.SynthesizeAnswers,
        "synthesize_answers_collector",
        "synthesize_answers",
        question,
        vector_answer,
        graph_answer,
    )
    
    # Run metrics after the BAML call completes
    await run_post_call_metrics(
        "synthesize_answers_collector",
        "synthesize_answers",
        input=question,
        output=synthesized_answer,
        context=[graph_answer + vector_answer],
        metrics=[
            {"type": "Hallucination", "params": {"model": "openrouter/openai/gpt-4o"}},
            {"type": "AnswerRelevance", "params": {"model": "openrouter/openai/gpt-4o"}},
            {"type": "Moderation", "params": {"model": "openrouter/openai/gpt-4o"}},
            {"type": "Usefulness", "params": {"model": "openrouter/openai/gpt-4o"}},
        ]
    )
    
    return synthesized_answer


# Evaluation Functions
@opik.track(flush=True)
async def generate_response(question: str, question_number: int = None) -> str | None:
    graph_answer, vector_answer = await run_hybrid_rag(question, question_number)
    synthesized_answer = await synthesize_answers(question, vector_answer, graph_answer)
    

    
    # Update the current span with question-specific information
    span_name = f"Question {question_number}" if question_number else "Question"
    opik_context.update_current_span(
        name=span_name,
        metadata={
            "workflow_type": "rag_evaluation",
            "question_number": question_number,
            "question": question,
            "vector_answer_length": len(vector_answer),
            "graph_answer_length": len(graph_answer),
            "synthesized_answer_length": len(synthesized_answer),
            "has_vector_answer": bool(vector_answer),
            "has_graph_answer": bool(graph_answer),
        },
    )
    
    return synthesized_answer


@opik.track(flush=True)
async def run_evaluation() -> None:
    """Run the evaluation suite with predefined questions."""
    questions = [
        "How many patients with the last name 'Rosenbaum' received multiple immunizations?",
        "What are the full names of the patients treated by the practitioner named Josef Klein?",
        "Did the practitioner 'Arla Fritsch' treat more than one patient?",
        "What are the unique categories of substances patients are allergic to?",
        "How many patients were born in between the years 1990 and 2000?",
        "How many patients were immunized after January 1, 2022?",
        "Which practitioner treated the most patients? Return their full name and how many patients they treated.",
        "Is the patient ID 45 allergic to the substance 'shellfish'? If so, what city and state do they live in, and what is the full name of the practitioner who treated them?",
        "How many patients are immunized for influenza?",
        "How many substances cause allergies in the category 'food'?",
    ]
    
    # Create the top-level trace for the entire evaluation suite
    opik_context.update_current_trace(
        name="RAG Evaluation Suite",
        input={"total_questions": len(questions)},
        metadata={
            "evaluation_type": "full_suite",
            "total_questions": len(questions),
        },
        tags=["rag", "evaluation", "fhir", "healthcare", "suite"]
    )
    
    # Get number of questions to evaluate, default to all questions
    num_questions = int(os.environ.get("NUM_EVAL_QUESTIONS", len(questions)))
    
    # Ensure num_questions doesn't exceed available questions
    num_questions = min(num_questions, len(questions))
    
    for i, question in enumerate(questions[:num_questions], 1):
        result = await generate_response(question, question_number=i)
        print(f"Answer {i}: {result}")
        print("-" * 80)


if __name__ == "__main__":
    # Run evaluation by default
    asyncio.run(run_evaluation()) 