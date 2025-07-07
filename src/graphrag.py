"""
Graph RAG workflow
"""

import asyncio
from typing import Any

from dotenv import load_dotenv

import utils
from baml_client import b
from tests import test_data

load_dotenv()

DB_PATH = "./fhir_kuzu_db"
db_manager = utils.DatabaseManager(DB_PATH)
conn = db_manager.get_connection()
schema_xml = db_manager.get_schema_xml(db_manager.get_schema_dict)


async def run_graphrag(test_case: dict[str, Any]) -> dict[str, str | list[dict[str, str]] | None]:
    question = test_case["question"]
    compressed_schema = await asyncio.to_thread(b.CompressSchema, schema_xml, question)
    compressed_schema = compressed_schema.model_dump()
    compressed_schema_xml = await asyncio.to_thread(db_manager.get_schema_xml, compressed_schema)
    response = await asyncio.to_thread(b.RAGText2Cypher, compressed_schema_xml, question, "")
    try:
        res = await asyncio.to_thread(conn.execute, response.cypher)
        context = res.get_as_pl().to_dicts()  # type: ignore
    except Exception as e:
        print(f"Error executing query: {e}")
        print(f"Query:{question}\n{response.cypher}")
        context = None
    if context:
        answer = await asyncio.to_thread(
            b.RAGAnswerQuestion, question, response.cypher, str(context)
        )
    else:
        answer = None
    return {
        "question": question,
        "cypher": response.cypher,
        "context": context,
        "answer": answer,
    }


if __name__ == "__main__":
    import os

    os.environ["BAML_LOG"] = "WARN"

    for i, test_case in enumerate(test_data.test_cases):
        result = asyncio.run(run_graphrag(test_case))
        print(f"Question {i + 1}: {result['question']}")
        print(f"Cypher: {result['cypher']}")
        print(f"Context: {result['context']}")
        print(f"Answer: {result['answer']}")
        print("-" * 40)
