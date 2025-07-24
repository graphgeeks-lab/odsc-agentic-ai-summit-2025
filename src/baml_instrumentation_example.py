"""
Example usage of BAML instrumentation utilities.

This file demonstrates different ways to use the BAML instrumentation
utilities for Opik tracking.
"""

import asyncio
from baml_client.async_client import b
from baml_instrumentation import BAMLInstrumentation, track_baml_call


# Example 1: Using the class-based approach
async def example_class_based():
    """Example using the BAMLInstrumentation class."""
    print("=== Class-based approach ===")
    
    # Create an instrumentation instance
    instrumentation = BAMLInstrumentation("example_collector")
    
    # Track a BAML call
    result = await instrumentation.track_call(
        b.AnswerQuestion,
        "example_span",
        "What is 2+2?",
        "The answer is 4.",
        additional_metadata={"example_type": "class_based"}
    )
    
    print(f"Result: {result}")
    return result


# Example 2: Using the utility function
async def example_utility_function():
    """Example using the track_baml_call utility function."""
    print("=== Utility function approach ===")
    
    result = await track_baml_call(
        b.AnswerQuestion,
        "example_collector_2",
        "example_span_2",
        "What is 3+3?",
        "The answer is 6.",
        additional_metadata={"example_type": "utility_function"}
    )
    
    print(f"Result: {result}")
    return result


# Example 3: Reusing the same instrumentation instance for multiple calls
async def example_reuse_instrumentation():
    """Example reusing the same instrumentation instance."""
    print("=== Reuse instrumentation instance ===")
    
    instrumentation = BAMLInstrumentation("reuse_collector")
    
    # First call
    result1 = await instrumentation.track_call(
        b.AnswerQuestion,
        "first_call",
        "What is 1+1?",
        "The answer is 2.",
        additional_metadata={"call_number": 1}
    )
    
    # Second call with the same instance
    result2 = await instrumentation.track_call(
        b.AnswerQuestion,
        "second_call",
        "What is 2+2?",
        "The answer is 4.",
        additional_metadata={"call_number": 2}
    )
    
    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}")
    return result1, result2


async def main():
    """Run all examples."""
    print("BAML Instrumentation Examples\n")
    
    # Run all examples
    await example_class_based()
    print()
    
    await example_utility_function()
    print()
    
    await example_reuse_instrumentation()
    print()
    
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main()) 