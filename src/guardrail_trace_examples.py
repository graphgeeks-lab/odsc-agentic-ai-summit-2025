"""
Comprehensive Examples of Guardrail Integration with Opik Traces and Spans

This script demonstrates various ways to integrate guardrail results with Opik
traces and spans for comprehensive observability and debugging.
"""

import asyncio
import os
from typing import Any, Dict, List
from datetime import datetime

from dotenv import load_dotenv
from opik import opik_context

from guardrails import (
    EmailGuardrail,
    GuardrailAction,
    GuardrailSeverity,
    GuardrailValidationFailed
)
from enhanced_guardrail_integration import (
    EnhancedGuardrailManager,
    GuardrailTraceDecorator
)

# Load environment variables
load_dotenv()


async def example_1_basic_span_integration():
    """
    Example 1: Basic span integration with guardrail results.
    
    This shows how guardrail results are automatically logged to Opik spans.
    """
    print("\n=== Example 1: Basic Span Integration ===")
    
    # Create a simple guardrail
    guardrail = EmailGuardrail(
        action=GuardrailAction.WARN,
        severity=GuardrailSeverity.MEDIUM,
        mask_emails=True
    )
    
    # Test text with emails
    text = "Contact support@company.com or admin@gmail.com for help"
    
    # This will automatically create a span with guardrail results
    result = guardrail.validate(text)
    
    # Manually update the current span with guardrail information
    opik_context.update_current_span(
        name="basic_guardrail_validation",
        metadata={
            "guardrail_type": "email",
            "triggered": result.triggered,
            "action": result.action.value,
            "severity": result.severity.value,
            "entities_found": result.entities_found,
            "message": result.message,
            "details": result.details
        },
        tags=["guardrail", "email", "basic"]
    )
    
    print(f"Guardrail triggered: {result.triggered}")
    print(f"Entities found: {result.entities_found}")
    print("Span created with guardrail metadata")


async def example_2_enhanced_tracing():
    """
    Example 2: Enhanced tracing with detailed metrics and tags.
    
    This shows how to use the EnhancedGuardrailManager for comprehensive tracing.
    """
    print("\n=== Example 2: Enhanced Tracing ===")
    
    # Create enhanced manager with multiple guardrails
    manager = EnhancedGuardrailManager([
        EmailGuardrail(
            action=GuardrailAction.WARN,
            severity=GuardrailSeverity.MEDIUM,
            mask_emails=True
        ),
        EmailGuardrail(
            action=GuardrailAction.BLOCK,
            severity=GuardrailSeverity.HIGH,
            block_common_domains=True
        )
    ])
    
    # Test with different texts
    test_cases = [
        "Contact support@company.com",
        "Email admin@gmail.com for assistance",
        "No emails in this text"
    ]
    
    for i, text in enumerate(test_cases):
        try:
            processed_text = await manager.validate_with_detailed_tracing(
                text,
                span_name=f"enhanced_validation_{i}",
                trace_tags=["enhanced", "multi_guardrail", f"test_case_{i}"],
                custom_metadata={
                    "test_case": i,
                    "original_text": text,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            print(f"Case {i}: {text} -> {processed_text}")
            
        except GuardrailValidationFailed as e:
            print(f"Case {i}: BLOCKED - {e}")
    
    # Get aggregated metrics
    metrics = manager.get_guardrail_metrics()
    print(f"Aggregated metrics: {metrics}")


async def example_3_trace_level_integration():
    """
    Example 3: Trace-level integration with guardrail summaries.
    
    This shows how to integrate guardrail results at the trace level.
    """
    print("\n=== Example 3: Trace-Level Integration ===")
    
    # Start a new trace for this example
    opik_context.update_current_trace(
        name="Guardrail Trace Example",
        input={"example_type": "trace_level_integration"},
        metadata={
            "guardrail_summary": {
                "total_validations": 0,
                "total_entities_found": 0,
                "total_blocks": 0,
                "total_warnings": 0,
                "critical_violations": 0,
                "high_severity_violations": 0
            }
        },
        tags=["guardrail", "trace_level", "example"]
    )
    
    # Create guardrails
    guardrails = [
        EmailGuardrail(action=GuardrailAction.WARN, mask_emails=True),
        EmailGuardrail(action=GuardrailAction.BLOCK, severity=GuardrailSeverity.HIGH, block_common_domains=True)
    ]
    
    # Process multiple inputs
    inputs = [
        "Contact me at user@example.com",
        "Send to admin@gmail.com",
        "No sensitive data here"
    ]
    
    for i, input_text in enumerate(inputs):
        # Create a span for each validation
        span_name = f"trace_validation_{i}"
        
        # Run guardrails
        results = []
        for guardrail in guardrails:
            result = guardrail.validate(input_text)
            results.append(result)
        
        # Update span with results
        triggered_results = [r for r in results if r.triggered]
        all_entities = []
        for result in results:
            if result.entities_found:
                all_entities.extend(result.entities_found)
        
        opik_context.update_current_span(
            name=span_name,
            metadata={
                "input_text": input_text,
                "guardrails_run": len(results),
                "guardrails_triggered": len(triggered_results),
                "entities_found": list(set(all_entities)),
                "results": [
                    {
                        "action": result.action.value,
                        "severity": result.severity.value,
                        "triggered": result.triggered,
                        "message": result.message
                    }
                    for result in results
                ]
            },
            tags=["trace_validation", f"input_{i}"]
        )
        
        # Update trace summary
        try:
            current_trace_data = opik_context.get_current_trace_data()
            current_metadata = current_trace_data.metadata if current_trace_data else {}
            guardrail_summary = current_metadata.get("guardrail_summary", {})
            
            guardrail_summary["total_validations"] += 1
            guardrail_summary["total_entities_found"] += len(set(all_entities))
            
            for result in triggered_results:
                if result.action == GuardrailAction.BLOCK:
                    guardrail_summary["total_blocks"] += 1
                elif result.action == GuardrailAction.WARN:
                    guardrail_summary["total_warnings"] += 1
                
                if result.severity == GuardrailSeverity.CRITICAL:
                    guardrail_summary["critical_violations"] += 1
                elif result.severity == GuardrailSeverity.HIGH:
                    guardrail_summary["high_severity_violations"] += 1
            
            current_metadata["guardrail_summary"] = guardrail_summary
            opik_context.update_current_trace(metadata=current_metadata)
        except Exception as e:
            # Silently handle errors when updating trace metadata
            pass
        
        print(f"Input {i}: {len(triggered_results)} guardrails triggered")


async def example_4_decorator_integration():
    """
    Example 4: Using decorators for automatic guardrail tracing.
    
    This shows how to use the GuardrailTraceDecorator for automatic integration.
    """
    print("\n=== Example 4: Decorator Integration ===")
    
    # Create a function with guardrail decorator
    @GuardrailTraceDecorator([
        EmailGuardrail(action=GuardrailAction.WARN, mask_emails=True)
    ], "decorated_function")
    async def process_user_message(message: str) -> str:
        """Process a user message with automatic guardrail validation."""
        # Simulate processing
        await asyncio.sleep(0.1)
        return f"Processed: {message}"
    
    # Test the decorated function
    test_messages = [
        "Hello, how are you?",
        "Contact support@company.com for help",
        "My email is user@gmail.com"
    ]
    
    for message in test_messages:
        try:
            result = await process_user_message(message)
            print(f"Input: {message}")
            print(f"Output: {result}")
        except GuardrailValidationFailed as e:
            print(f"Input: {message}")
            print(f"Blocked: {e}")


async def example_5_custom_span_creation():
    """
    Example 5: Custom span creation with guardrail results.
    
    This shows how to create custom spans with detailed guardrail information.
    """
    print("\n=== Example 5: Custom Span Creation ===")
    
    # Create guardrails
    email_guardrail = EmailGuardrail(
        action=GuardrailAction.WARN,
        severity=GuardrailSeverity.MEDIUM,
        mask_emails=True
    )
    
    # Test text
    text = "Contact support@company.com and admin@gmail.com for assistance"
    
    # Run validation
    result = email_guardrail.validate(text)
    
    # Create custom span with detailed information
    span_metadata = {
        "validation_type": "email_detection",
        "input_length": len(text),
        "triggered": result.triggered,
        "action_taken": result.action.value,
        "severity_level": result.severity.value,
        "entities_detected": result.entities_found,
        "processing_details": {
            "total_emails": result.details.get("total_emails", 0),
            "blocked_emails": result.details.get("blocked_emails", []),
            "allowed_emails": result.details.get("allowed_emails", []),
            "email_details": result.details.get("email_details", [])
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Create span with custom tags
    tags = ["custom_span", "email_validation"]
    if result.triggered:
        tags.extend(["triggered", result.action.value, result.severity.value])
    if result.entities_found:
        tags.append("entities_found")
    
    opik_context.update_current_span(
        name="custom_guardrail_span",
        metadata=span_metadata,
        tags=tags
    )
    
    print(f"Custom span created with {len(span_metadata)} metadata fields")
    print(f"Tags: {tags}")


async def example_6_rag_workflow_integration():
    """
    Example 6: Integration in a RAG workflow.
    
    This shows how guardrails integrate with a complete RAG workflow.
    """
    print("\n=== Example 6: RAG Workflow Integration ===")
    
    # Start a RAG workflow trace
    opik_context.update_current_trace(
        name="RAG Workflow with Guardrails",
        input={"workflow_type": "rag_with_guardrails"},
        metadata={
            "workflow_steps": ["input_validation", "rag_processing", "output_validation"],
            "guardrail_config": {
                "input_guardrails": ["email_detection"],
                "output_guardrails": ["email_detection", "content_moderation"]
            }
        },
        tags=["rag", "guardrails", "workflow"]
    )
    
    # Simulate RAG workflow steps
    workflow_steps = [
        ("input_validation", "Contact support@company.com for help"),
        ("rag_processing", "Processing user query..."),
        ("output_validation", "Based on our database, you can contact support@company.com or admin@gmail.com for assistance")
    ]
    
    # Create guardrails
    guardrails = [
        EmailGuardrail(action=GuardrailAction.WARN, mask_emails=True)
    ]
    
    for step_name, step_data in workflow_steps:
        # Create span for each workflow step
        opik_context.update_current_span(
            name=f"rag_workflow_{step_name}",
            metadata={
                "step": step_name,
                "data": step_data,
                "step_number": workflow_steps.index((step_name, step_data)) + 1
            },
            tags=["rag_workflow", step_name]
        )
        
        # Apply guardrails if it's input or output validation
        if "validation" in step_name:
            for guardrail in guardrails:
                result = guardrail.validate(step_data)
                
                # Update span with guardrail results
                opik_context.update_current_span(
                    name=f"rag_workflow_{step_name}",
                    metadata={
                        "guardrail_results": {
                            "triggered": result.triggered,
                            "action": result.action.value,
                            "entities_found": result.entities_found
                        }
                    }
                )
                
                if result.triggered:
                    print(f"{step_name}: Guardrail triggered - {result.message}")
    
    print("RAG workflow completed with guardrail integration")


async def main():
    """Run all guardrail trace integration examples."""
    print("Guardrail Trace Integration Examples")
    print("=" * 50)
    
    # Run all examples
    await example_1_basic_span_integration()
    await example_2_enhanced_tracing()
    await example_3_trace_level_integration()
    await example_4_decorator_integration()
    await example_5_custom_span_creation()
    await example_6_rag_workflow_integration()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nKey integration points demonstrated:")
    print("- Basic span creation with guardrail metadata")
    print("- Enhanced tracing with detailed metrics")
    print("- Trace-level integration with summaries")
    print("- Decorator-based automatic integration")
    print("- Custom span creation with detailed information")
    print("- RAG workflow integration")


if __name__ == "__main__":
    asyncio.run(main()) 