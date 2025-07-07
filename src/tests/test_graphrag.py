import pytest

from graphrag import run_graphrag
from tests import test_data

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",  # ...add more as needed
}


def number_variants(s: str) -> list[str]:
    """
    Convert a number to a list of possible variants, including word variants.
    This is useful for testing when the Graph RAG answer may be presented in a string format, but
    the expected answer is in number format.
    """
    s = s.lower()
    if s.isdigit():
        for word, num in NUMBER_WORDS.items():
            if num == s:
                return [s, word]
        return [s]
    elif s in NUMBER_WORDS:
        return [s, NUMBER_WORDS[s]]
    return [s]


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", test_data.test_cases)
async def test_graphrag_eval_expected_answer(test_case):
    result = await run_graphrag(test_case)
    for key in ["question", "cypher", "context", "answer"]:
        assert key in result

    assert result["answer"] is not None
    answer_str = str(result["answer"]).lower()
    failed = []
    for expected in test_case["expected_values"]:
        variants = number_variants(expected)
        if not any(variant in answer_str for variant in variants):
            failed.append((expected.lower(), variants))
    assert not failed, f"Failed to find expected values: {failed} in answer: {answer_str}"
