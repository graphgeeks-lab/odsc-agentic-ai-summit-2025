"""
Example script demonstrating email guardrails in a RAG application.

This script shows how to integrate the custom guardrails with a RAG system
to protect against email address leakage in both inputs and outputs.
"""

import asyncio
import os
from dotenv import load_dotenv

from guardrails import (
    EmailGuardrail,
    GuardrailAction,
    GuardrailSeverity,
    validate_input_with_guardrails,
    validate_output_with_guardrails,
    GuardrailValidationFailed
)

# Load environment variables
load_dotenv()


async def simulate_rag_query(user_question: str) -> str:
    """
    Simulate a RAG query response.
    
    In a real application, this would be your actual RAG pipeline.
    """
    # Simulate some processing time
    await asyncio.sleep(0.1)
    
    # Simulate a response that might contain email addresses
    if "contact" in user_question.lower():
        return "You can contact our support team at support@company.com or reach out to sales@business.org for more information."
    elif "email" in user_question.lower():
        return "Please send your inquiry to info@example.com and we'll get back to you within 24 hours."
    else:
        return "I don't have specific contact information for that query. Please check our website for general information."


async def process_user_query_with_guardrails(user_question: str) -> str:
    """
    Process a user query with email guardrails applied.
    
    This demonstrates how to integrate guardrails into a RAG workflow.
    """
    print(f"\n=== Processing Query: {user_question} ===")
    
    # Step 1: Validate input with guardrails
    print("1. Validating input...")
    try:
        input_guardrails = [
            EmailGuardrail(
                action=GuardrailAction.WARN,
                severity=GuardrailSeverity.MEDIUM,
                mask_emails=True,
                block_common_domains=False
            )
        ]
        
        processed_question = validate_input_with_guardrails(
            user_question,
            input_guardrails,
            "user_input_validation"
        )
        
        if processed_question != user_question:
            print(f"   Input processed: {processed_question}")
            user_question = processed_question
        else:
            print("   Input validation passed")
            
    except GuardrailValidationFailed as e:
        print(f"   Input blocked by guardrail: {e}")
        return "I'm sorry, but I cannot process this request due to security concerns."
    except Exception as e:
        print(f"   Input validation error: {e}")
    
    # Step 2: Generate RAG response
    print("2. Generating response...")
    try:
        response = await simulate_rag_query(user_question)
        print(f"   Raw response: {response}")
    except Exception as e:
        print(f"   Response generation error: {e}")
        return "I encountered an error while processing your request."
    
    # Step 3: Validate output with guardrails
    print("3. Validating output...")
    try:
        output_guardrails = [
            EmailGuardrail(
                action=GuardrailAction.WARN,
                severity=GuardrailSeverity.MEDIUM,
                mask_emails=True,
                block_common_domains=False
            )
        ]
        
        processed_response = validate_output_with_guardrails(
            response,
            output_guardrails,
            "rag_output_validation"
        )
        
        if processed_response != response:
            print(f"   Output processed: {processed_response}")
            response = processed_response
        else:
            print("   Output validation passed")
            
    except GuardrailValidationFailed as e:
        print(f"   Output blocked by guardrail: {e}")
        return "I'm sorry, but I cannot provide this response due to security concerns."
    except Exception as e:
        print(f"   Output validation error: {e}")
    
    return response


async def demonstrate_different_guardrail_configurations():
    """Demonstrate different guardrail configurations."""
    print("\n" + "="*60)
    print("DEMONSTRATING DIFFERENT GUARDRAIL CONFIGURATIONS")
    print("="*60)
    
    # Test cases with different scenarios
    test_cases = [
        {
            "name": "Basic email detection",
            "question": "What's the contact email for support?",
            "description": "Standard email detection and masking"
        },
        {
            "name": "Question with email",
            "question": "Can you help me with john.doe@gmail.com?",
            "description": "Input contains email address"
        },
        {
            "name": "Multiple emails in question",
            "question": "Contact info: admin@company.com and user@test.org",
            "description": "Multiple emails in input"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Description: {test_case['description']}")
        
        response = await process_user_query_with_guardrails(test_case['question'])
        print(f"Final response: {response}")


async def demonstrate_blocking_guardrails():
    """Demonstrate blocking guardrails for sensitive domains."""
    print("\n" + "="*60)
    print("DEMONSTRATING BLOCKING GUARDRAILS")
    print("="*60)
    
    # Create a more restrictive guardrail
    blocking_guardrails = [
        EmailGuardrail(
            action=GuardrailAction.BLOCK,
            severity=GuardrailSeverity.HIGH,
            block_common_domains=True,
            allowed_domains=["company.com"],
            blocked_domains=["competitor.com"]
        )
    ]
    
    test_questions = [
        "Contact me at user@gmail.com",
        "Send to admin@company.com",
        "Email at info@competitor.com",
        "Contact support@business.org"
    ]
    
    for question in test_questions:
        print(f"\nTesting: {question}")
        try:
            processed = validate_input_with_guardrails(
                question,
                blocking_guardrails,
                "blocking_guardrail_test"
            )
            print(f"Result: PASSED - {processed}")
        except GuardrailValidationFailed as e:
            print(f"Result: BLOCKED - {e}")
        except Exception as e:
            print(f"Result: ERROR - {e}")


async def main():
    """Main function to run the guardrail examples."""
    print("Email Guardrail Examples for RAG Applications")
    print("=" * 60)
    
    # Basic examples
    await demonstrate_different_guardrail_configurations()
    
    # Blocking examples
    await demonstrate_blocking_guardrails()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nKey features demonstrated:")
    print("- Email detection and masking")
    print("- Input and output validation")
    print("- Different action types (WARN, BLOCK)")
    print("- Domain filtering (whitelist/blacklist)")
    print("- Integration with Opik tracking")
    print("- Error handling and graceful degradation")


if __name__ == "__main__":
    asyncio.run(main()) 