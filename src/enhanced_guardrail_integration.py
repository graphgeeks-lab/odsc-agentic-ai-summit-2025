"""
Enhanced Guardrail Integration with Opik Traces and Spans

This module demonstrates advanced ways to integrate guardrail results
with Opik traces and spans for better observability and debugging.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from opik import opik_context
from guardrails import (
    EmailGuardrail,
    GuardrailAction,
    GuardrailSeverity,
    GuardrailResult,
    GuardrailValidationFailed
)


@dataclass
class GuardrailSpanInfo:
    """Information for creating guardrail-specific spans."""
    span_name: str
    guardrail_type: str
    input_text: str
    output_text: str
    results: List[GuardrailResult]
    processing_time_ms: float
    timestamp: datetime


class EnhancedGuardrailManager:
    """
    Enhanced guardrail manager with advanced Opik integration.
    
    This class provides more sophisticated integration with Opik traces
    and spans, including custom metrics, tags, and detailed logging.
    """
    
    def __init__(self, guardrails: List[Any] = None):
        """Initialize the enhanced guardrail manager."""
        self.guardrails = guardrails or []
        self.span_history: List[GuardrailSpanInfo] = []
    
    def add_guardrail(self, guardrail: Any) -> None:
        """Add a guardrail to the manager."""
        self.guardrails.append(guardrail)
    
    async def validate_with_detailed_tracing(
        self,
        text: str,
        span_name: str = "guardrail_validation",
        trace_tags: List[str] = None,
        custom_metadata: Dict[str, Any] = None,
        validation_type: str = "general"  # "input" or "output"
    ) -> str:
        """
        Validate text with detailed Opik tracing and metrics.
        
        Args:
            text: Text to validate
            span_name: Name for the Opik span
            trace_tags: Additional tags for the trace
            custom_metadata: Additional metadata to include
            validation_type: Type of validation ("input" or "output")
            
        Returns:
            Processed text
            
        Raises:
            GuardrailValidationFailed: If any guardrail blocks the text
        """
        start_time = datetime.now()
        
        # Run all guardrails
        results = []
        for guardrail in self.guardrails:
            result = guardrail.validate(text)
            results.append(result)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Create detailed span information
        span_info = GuardrailSpanInfo(
            span_name=span_name,
            guardrail_type="email",  # Could be extended for other types
            input_text=text,
            output_text=text,  # Will be updated if masking occurs
            results=results,
            processing_time_ms=processing_time,
            timestamp=start_time
        )
        
        # Process results and handle actions
        processed_text = text
        triggered_results = [r for r in results if r.triggered]
        
        # Track masking actions
        masking_applied = False
        masking_details = []
        
        # Apply masking if needed
        for guardrail in self.guardrails:
            if isinstance(guardrail, EmailGuardrail) and guardrail.mask_emails:
                original_text = processed_text
                processed_text = guardrail.mask_text(processed_text)
                if processed_text != original_text:
                    masking_applied = True
                    masking_details.append({
                        "guardrail_type": "email",
                        "masking_method": "character_replacement",
                        "mask_char": guardrail.mask_char,
                        "original_length": len(original_text),
                        "masked_length": len(processed_text)
                    })
        
        span_info.output_text = processed_text
        
        # Add validation type and masking info to custom metadata
        enhanced_metadata = custom_metadata or {}
        enhanced_metadata.update({
            "validation_type": validation_type,
            "masking_applied": masking_applied,
            "masking_details": masking_details,
            "text_modified_by_masking": masking_applied
        })
        
        # Create comprehensive Opik span
        await self._create_detailed_span(span_info, trace_tags, enhanced_metadata)
        
        # Handle blocking actions
        for result in triggered_results:
            if result.action == GuardrailAction.BLOCK:
                raise GuardrailValidationFailed(
                    f"Guardrail validation failed: {result.message}",
                    result
                )
        
        # Store span info for later analysis
        self.span_history.append(span_info)
        
        return processed_text
    
    async def _create_detailed_span(
        self,
        span_info: GuardrailSpanInfo,
        trace_tags: List[str] = None,
        custom_metadata: Dict[str, Any] = None
    ) -> None:
        """Create a detailed Opik span with comprehensive guardrail information."""
        
        # Prepare metadata
        metadata = {
            "guardrail_type": span_info.guardrail_type,
            "processing_time_ms": span_info.processing_time_ms,
            "input_length": len(span_info.input_text),
            "output_length": len(span_info.output_text),
            "guardrails_run": len(span_info.results),
            "guardrails_triggered": len([r for r in span_info.results if r.triggered]),
            "text_modified": span_info.input_text != span_info.output_text,
            "timestamp": span_info.timestamp.isoformat(),
            "validation_phase": custom_metadata.get("validation_type", "general") if custom_metadata else "general",
        }
        
        # Add custom metadata
        if custom_metadata:
            metadata.update(custom_metadata)
        
        # Prepare detailed results
        detailed_results = []
        all_entities = []
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        action_counts = {"block": 0, "warn": 0, "log": 0}
        
        for result in span_info.results:
            # Count severities and actions
            severity_counts[result.severity.value] += 1
            if result.triggered:
                action_counts[result.action.value] += 1
            
            # Collect entities
            if result.entities_found:
                all_entities.extend(result.entities_found)
            
            # Create detailed result entry
            detailed_result = {
                "action": result.action.value,
                "severity": result.severity.value,
                "triggered": result.triggered,
                "message": result.message,
                "entities_found": result.entities_found,
                "details": result.details
            }
            detailed_results.append(detailed_result)
        
        # Add aggregated information
        metadata.update({
            "entities_found": list(set(all_entities)),
            "severity_distribution": severity_counts,
            "action_distribution": action_counts,
            "results": detailed_results
        })
        
        # Prepare tags
        tags = ["guardrail", "validation", span_info.guardrail_type]
        if trace_tags:
            tags.extend(trace_tags)
        
        # Add severity-based tags
        if any(r.severity == GuardrailSeverity.CRITICAL for r in span_info.results):
            tags.append("critical")
        if any(r.severity == GuardrailSeverity.HIGH for r in span_info.results):
            tags.append("high_severity")
        
        # Add action-based tags
        if any(r.action == GuardrailAction.BLOCK for r in span_info.results):
            tags.append("blocked")
        if any(r.action == GuardrailAction.WARN for r in span_info.results):
            tags.append("warned")
        
        # Create the span
        try:
            opik_context.update_current_span(
                name=span_info.span_name,
                metadata=metadata,
                tags=tags
            )
        except Exception as e:
            # Silently handle errors when creating spans
            # This prevents guardrail errors from breaking the main application
            pass
        
        # Also update the parent trace with guardrail summary
        self._update_trace_with_guardrail_summary(span_info)
    
    def _update_trace_with_guardrail_summary(self, span_info: GuardrailSpanInfo) -> None:
        """Update the parent trace with guardrail summary information."""
        
        try:
            # Get current trace data
            current_trace_data = opik_context.get_current_trace_data()
            current_metadata = current_trace_data.metadata if current_trace_data else {}
            
            # Add guardrail summary
            guardrail_summary = current_metadata.get("guardrail_summary", {
                "total_validations": 0,
                "total_entities_found": 0,
                "total_blocks": 0,
                "total_warnings": 0,
                "critical_violations": 0,
                "high_severity_violations": 0
            })
            
            # Update summary
            guardrail_summary["total_validations"] += 1
            
            for result in span_info.results:
                if result.entities_found:
                    guardrail_summary["total_entities_found"] += len(result.entities_found)
                
                if result.triggered:
                    if result.action == GuardrailAction.BLOCK:
                        guardrail_summary["total_blocks"] += 1
                    elif result.action == GuardrailAction.WARN:
                        guardrail_summary["total_warnings"] += 1
                    
                    if result.severity == GuardrailSeverity.CRITICAL:
                        guardrail_summary["critical_violations"] += 1
                    elif result.severity == GuardrailSeverity.HIGH:
                        guardrail_summary["high_severity_violations"] += 1
            
            # Update trace metadata
            current_metadata["guardrail_summary"] = guardrail_summary
            opik_context.update_current_trace(metadata=current_metadata)
        except Exception as e:
            # Silently handle errors when updating trace metadata
            # This prevents guardrail errors from breaking the main application
            pass
    
    def get_guardrail_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from guardrail history."""
        if not self.span_history:
            return {}
        
        total_validations = len(self.span_history)
        total_entities = sum(len(info.results) for info in self.span_history)
        total_triggered = sum(
            sum(1 for r in info.results if r.triggered) 
            for info in self.span_history
        )
        
        # Calculate average processing time
        avg_processing_time = sum(
            info.processing_time_ms for info in self.span_history
        ) / total_validations
        
        return {
            "total_validations": total_validations,
            "total_entities_checked": total_entities,
            "total_triggered": total_triggered,
            "trigger_rate": total_triggered / total_entities if total_entities > 0 else 0,
            "avg_processing_time_ms": avg_processing_time,
            "span_history_count": len(self.span_history)
        }


class GuardrailTraceDecorator:
    """
    Decorator for adding guardrail tracing to functions.
    
    This decorator automatically wraps function calls with guardrail
    validation and detailed tracing.
    """
    
    def __init__(self, guardrails: List[Any], span_name_prefix: str = "guardrail"):
        """
        Initialize the decorator.
        
        Args:
            guardrails: List of guardrails to apply
            span_name_prefix: Prefix for span names
        """
        self.guardrails = guardrails
        self.span_name_prefix = span_name_prefix
        self.manager = EnhancedGuardrailManager(guardrails)
    
    def __call__(self, func):
        """Decorator implementation."""
        async def wrapper(*args, **kwargs):
            # Extract text from function arguments (assumes first arg is text)
            if args and isinstance(args[0], str):
                text = args[0]
                span_name = f"{self.span_name_prefix}_{func.__name__}"
                
                # Validate with guardrails
                processed_text = await self.manager.validate_with_detailed_tracing(
                    text,
                    span_name=span_name,
                    trace_tags=[func.__name__, "decorated"],
                    custom_metadata={
                        "function_name": func.__name__,
                        "decorator_type": "guardrail_trace"
                    }
                )
                
                # Call function with processed text
                if args:
                    args = (processed_text,) + args[1:]
                
                return await func(*args, **kwargs)
            else:
                # No text argument found, call function normally
                return await func(*args, **kwargs)
        
        return wrapper


# Example usage functions
async def example_function_with_guardrails():
    """Example function demonstrating guardrail integration."""
    
    # Create guardrails
    guardrails = [
        EmailGuardrail(
            action=GuardrailAction.WARN,
            severity=GuardrailSeverity.MEDIUM,
            mask_emails=True
        )
    ]
    
    # Create enhanced manager
    manager = EnhancedGuardrailManager(guardrails)
    
    # Test text with emails
    test_text = "Contact support@company.com or admin@gmail.com for assistance"
    
    try:
        processed_text = await manager.validate_with_detailed_tracing(
            test_text,
            span_name="example_validation",
            trace_tags=["example", "demo"],
            custom_metadata={
                "test_type": "email_detection",
                "environment": "development"
            }
        )
        
        print(f"Original: {test_text}")
        print(f"Processed: {processed_text}")
        
        # Get metrics
        metrics = manager.get_guardrail_metrics()
        print(f"Metrics: {metrics}")
        
    except GuardrailValidationFailed as e:
        print(f"Guardrail blocked: {e}")


# Decorator example
@GuardrailTraceDecorator([
    EmailGuardrail(action=GuardrailAction.WARN, mask_emails=True)
], "decorated_function")
async def process_user_input(text: str) -> str:
    """Example function decorated with guardrail tracing."""
    # Simulate processing
    await asyncio.sleep(0.1)
    return f"Processed: {text}"


async def main():
    """Main function to demonstrate enhanced integration."""
    print("Enhanced Guardrail Integration Examples")
    print("=" * 50)
    
    # Example 1: Direct usage
    print("\n1. Direct Enhanced Manager Usage:")
    await example_function_with_guardrails()
    
    # Example 2: Decorator usage
    print("\n2. Decorator Usage:")
    result = await process_user_input("Contact me at user@example.com")
    print(f"Result: {result}")
    
    print("\nEnhanced integration completed!")


if __name__ == "__main__":
    asyncio.run(main()) 