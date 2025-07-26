"""
BAML Instrumentation utilities for Opik tracking.

This module provides reusable classes and functions for instrumenting BAML calls
with Opik observability tracking.
"""

import asyncio
import os
from typing import Any, Callable, Dict, Optional, TypeVar

from opik import opik_context
from baml_py import Collector
from opik.evaluation.metrics import (
    Hallucination,
    Contains,
    AnswerRelevance,
    Moderation,
    Usefulness,
    ContextRecall,
    ContextPrecision,
    base_metric,
    score_result,
)



class BAMLInstrumentation:
    """
    A class for instrumenting BAML calls with Opik tracking.
    
    This class provides a reusable way to track BAML function calls with
    consistent metadata and usage tracking.
    """
    
    def __init__(self, collector_name: str, sample_rate: float = None):
        """
        Initialize the instrumentation with a collector name.
        
        Args:
            collector_name: Name for the BAML collector
            sample_rate: Fraction of calls to sample for metrics (defaults to METRICS_SAMPLE_RATE env var or 0.05)
        """
        self.collector_name = collector_name
        self.collector = Collector(name=collector_name)
        self.sample_rate = sample_rate or float(os.environ.get("METRICS_SAMPLE_RATE", 0.05))
    
    async def track_call(
        self,
        baml_function: Callable,
        span_name: str,
        *args,
        additional_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Track a BAML function call with Opik instrumentation.
        
        Args:
            baml_function: The BAML function to call
            span_name: Name for the Opik span
            *args: Arguments to pass to the BAML function
            additional_metadata: Additional metadata to include in the span
            **kwargs: Keyword arguments to pass to the BAML function
            
        Returns:
            The result of the BAML function call
        """
        # Add collector to BAML options
        baml_options = kwargs.get("baml_options", {})
        if "collector" not in baml_options:
            baml_options["collector"] = []
        baml_options["collector"].append(self.collector)
        kwargs["baml_options"] = baml_options
        
        # Call the BAML function
        result = await baml_function(*args, **kwargs)
        
        # Update Opik context with BAML data
        self._update_opik_context(span_name, additional_metadata or {})
        
        return result
    
    async def run_post_call_metrics(
        self,
        span_name: str,
        input: str,
        output: str,
        context: list = None,
        metrics: list = None,
        sample_rate: float = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run metrics after a BAML call has completed.
        
        This method is useful when you need to run metrics on the output of a BAML call
        that wasn't available during the initial call.
        
        Args:
            span_name: Name for the Opik span to update
            input: Input string to the LLM (for metrics)
            output: Output string from the LLM (for metrics)
            context: Optional context list (for metrics)
            metrics: List of metric configs to run (e.g., [{"type": "Hallucination", "params": {}}])
            sample_rate: Override the instance sample rate (optional)
            additional_metadata: Additional metadata to include in the span
        """
        # Use passed sample_rate, or read from env var, or use instance sample_rate
        if sample_rate is not None:
            effective_sample_rate = sample_rate
        else:
            # Read directly from env var, fall back to instance sample_rate
            import os as os_module
            env_sample_rate = os_module.environ.get("METRICS_SAMPLE_RATE")
            effective_sample_rate = float(env_sample_rate) if env_sample_rate else self.sample_rate
        
        # Check if we should run metrics based on sample rate
        should_run_metrics = (input is not None and output is not None and 
                            metrics is not None and effective_sample_rate > 0)
        
        if should_run_metrics:
            print(f"[DEBUG] Running metrics for span: {span_name} (sample_rate: {effective_sample_rate})")
            metric_results = []
            for metric_cfg in metrics:
                metric_type = metric_cfg["type"]
                params = metric_cfg.get("params", {})
                if metric_type == "Hallucination":
                    # Extract model parameter from params or use default
                    model = params.get("model", "gpt-4o")
                    metric = Hallucination(track = True, model=model, **{k: v for k, v in params.items() if k != "model"})
                    score_result = await metric.ascore(input=input, output=output, context=context)
                elif metric_type == "AnswerRelevance":
                    # Extract model parameter from params or use default
                    model = params.get("model", "gpt-4o")
                    metric = AnswerRelevance(track = True, model=model, **{k: v for k, v in params.items() if k != "model"})
                    score_result = await metric.ascore(input=input, output=output, context=context)
                elif metric_type == "Contains":
                    # Filter out 'output' and 'reference' from params as they're not constructor parameters
                    constructor_params = {k: v for k, v in params.items() if k not in ["output", "reference"]}
                    metric = Contains(track = True, **constructor_params)
                    reference = params.get("reference", "")
                    score_result = await metric.ascore(output=output, reference=reference)
                elif metric_type == "Moderation":
                    # Extract model parameter from params or use default
                    model = params.get("model", "openrouter/openai/gpt-4o")
                    metric = Moderation(track = True, model=model, **{k: v for k, v in params.items() if k != "model"})
                    score_result = await metric.ascore(output=output)
                elif metric_type == "Usefulness":
                    # Extract model parameter from params or use default
                    model = params.get("model", "openrouter/openai/gpt-4o")
                    metric = Usefulness(track = True, model=model, **{k: v for k, v in params.items() if k != "model"})
                    score_result = await metric.ascore(input=input, output=output)
                elif metric_type == "ContextRecall":
                    # Extract model parameter from params or use default
                    model = params.get("model", "openrouter/openai/gpt-4o")
                    metric = ContextRecall(track = True, model=model, **{k: v for k, v in params.items() if k != "model"})
                    score_result = await metric.ascore(output=output, context=context)
                elif metric_type == "ContextPrecision":
                    # Extract model parameter from params or use default    
                    model = params.get("model", "openrouter/openai/gpt-4o")
                    metric = ContextPrecision(track = True, model=model, **{k: v for k, v in params.items() if k != "model"})
                    score_result = await metric.ascore(output=output, context=context)

                else:
                    continue  # Unknown metric type
                # Handle both sync and async score results
                if hasattr(score_result, 'value'):
                    value = score_result.value
                    reason = getattr(score_result, "reason", None)
                elif isinstance(score_result, (int, float)):
                    # Direct numeric value
                    value = score_result
                    reason = None
                elif score_result is None:
                    value = None
                    reason = "No result returned"
                else:
                    # Try to convert to string and extract numeric value
                    try:
                        value = float(str(score_result))
                        reason = str(score_result)
                    except (ValueError, TypeError):
                        value = None
                        reason = f"Could not parse result: {score_result}"
                
                # Ensure value is a valid number, skip if None or invalid
                if value is None or not isinstance(value, (int, float)):
                    print(f"[WARNING] Skipping metric {metric_type} with invalid value: {value}")
                    continue
                
                # Ensure value is within valid range (0-1 for most metrics)
                if value < 0 or value > 1:
                    print(f"[WARNING] Metric {metric_type} value {value} out of range [0,1], clamping")
                    value = max(0.0, min(1.0, value))
                
                metric_results.append({
                    "name": metric_type,
                    "value": value,
                    "reason": reason,
                })
            
            print("[DEBUG] Metric results:")
            for result in metric_results:
                print(f"  - {result['name']}: {result['value']} ({result['reason']})\n")
            
            # Convert metric results to feedback scores format for Opik
            feedback_scores = []
            for result in metric_results:
                feedback_score = {
                    "name": result["name"],
                    "value": result["value"],
                    "reason": result["reason"] if result["reason"] else None
                }
                feedback_scores.append(feedback_score)
            
            # Update the existing span with metric results and feedback scores
            current_metadata = additional_metadata or {}
            current_metadata["metrics"] = metric_results
            opik_context.update_current_span(
                name=span_name,
                metadata=current_metadata,
                feedback_scores=feedback_scores
            )
            
            # Also update the parent trace with feedback scores
            opik_context.update_current_trace(
                feedback_scores=feedback_scores
            )
    
    def _update_opik_context(self, span_name: str, additional_metadata: Dict[str, Any]) -> None:
        """
        Update the Opik context with BAML collector data.
        
        Args:
            span_name: Name for the Opik span
            additional_metadata: Additional metadata to include
        """
        if self.collector.last is not None:
            log = self.collector.last
            call = log.calls[0] if log.calls else None
            
            if call and call.usage:
                metadata = {
                    "function_name": log.function_name,
                    "duration_ms": call.timing.duration_ms if call.timing else None,
                    **additional_metadata
                }
                
                usage = {
                    "prompt_tokens": call.usage.input_tokens,
                    "completion_tokens": call.usage.output_tokens,
                    "total_tokens": (call.usage.input_tokens or 0) + (call.usage.output_tokens or 0),
                }

                # Get pricing from environment variables with sensible defaults
                prompt_price_per_1k = float(os.environ.get("PROMPT_PRICE_PER_1K", "0.0005"))  # Default $0.0005 per 1k tokens
                completion_price_per_1k = float(os.environ.get("COMPLETION_PRICE_PER_1K", "0.000009"))  # Default $0.000009 per 1k tokens
                
                # Calculate cost
                cost = (usage["prompt_tokens"] / 1000) * prompt_price_per_1k + (usage["completion_tokens"] / 1000) * completion_price_per_1k
                
                opik_context.update_current_span(
                    name=span_name,
                    metadata=metadata,
                    usage=usage,
                    provider=call.provider,
                    model=call.client_name,
                    total_cost=cost,
                )


async def track_baml_call(
    baml_function: Callable,
    collector_name: str,
    span_name: str,
    *args,
    sample_rate: float = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Utility function to track a single BAML call with Opik instrumentation.
    
    Args:
        baml_function: The BAML function to call
        collector_name: Name for the BAML collector
        span_name: Name for the Opik span
        *args: Arguments to pass to the BAML function
        sample_rate: Fraction of calls to sample for metrics (if None, reads from METRICS_SAMPLE_RATE env var)
        additional_metadata: Additional metadata to include in the span
        **kwargs: Keyword arguments to pass to the BAML function
        
    Returns:
        The result of the BAML function call
    """
    instrumentation = BAMLInstrumentation(collector_name, sample_rate=sample_rate)
    return await instrumentation.track_call(
        baml_function,
        span_name,
        *args,
        additional_metadata=additional_metadata,
        **kwargs
    )


async def run_post_call_metrics(
    collector_name: str,
    span_name: str,
    input: str,
    output: str,
    context: list = None,
    metrics: list = None,
    sample_rate: float = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Utility function to run metrics after a BAML call has completed.
    
    Args:
        collector_name: Name for the BAML collector
        span_name: Name for the Opik span to update
        input: Input string to the LLM (for metrics)
        output: Output string from the LLM (for metrics)
        context: Optional context list (for metrics)
        metrics: List of metric configs to run (e.g., [{"type": "Hallucination", "params": {}}])
        sample_rate: Fraction of calls to sample for metrics (if None, reads from METRICS_SAMPLE_RATE env var)
        additional_metadata: Additional metadata to include in the span
    """
    instrumentation = BAMLInstrumentation(collector_name, sample_rate=sample_rate)
    await instrumentation.run_post_call_metrics(
        span_name,
        input,
        output,
        context=context,
        metrics=metrics,
        sample_rate=sample_rate,
        additional_metadata=additional_metadata,
    ) 