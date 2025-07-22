"""
Observability & evals for the RAG system.

The observability tool used is Opik, by Comet:
https://www.comet.com/site/products/opik/
"""

import asyncio
import os

import opik
from dotenv import load_dotenv
from openai import OpenAI
from opik.integrations.openai import track_openai

from baml_client.async_client import b
from rag import run_hybrid_rag

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPIK_API_KEY = os.environ.get("OPIK_API_KEY")
OPIK_WORKSPACE = os.environ.get("OPIK_WORKSPACE")
OPIK_PROJECT_NAME = "ODSC-RAG"

# Initialize the OpenAI client with OpenRouter base URL
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
client = track_openai(client)

# Optional headers for OpenRouter leaderboard
headers = {
    "HTTP-Referer": "graphgeeks.org",  # Optional. Site URL for rankings
    "X-Title": "GraphGeeks",  # Optional. Site title for rankings
}

if OPIK_API_KEY and OPIK_WORKSPACE:
    os.environ["OPIK_API_KEY"] = OPIK_API_KEY
    os.environ["OPIK_WORKSPACE"] = OPIK_WORKSPACE
    os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT_NAME
else:
    print(
        "Please set the OPIK_API_KEY and OPIK_WORKSPACE environment variables to enable opik tracking"
    )


@opik.track
async def generate_response(question: str) -> str | None:
    graph_answer, vector_answer = await run_hybrid_rag(question)
    synthesized_answer = await b.SynthesizeAnswers(question, vector_answer, graph_answer)
    return synthesized_answer


async def main() -> None:
    questions = [
        # "How many patients with the last name 'Rosenbaum' received multiple immunizations?",
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
        result = await generate_response(question)  # type: ignore
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
