"""
BAML Instrumentation utilities for Opik tracking.

This module provides reusable classes and functions for instrumenting BAML calls
with Opik observability tracking.
"""

import asyncio
from typing import Any, Callable, Dict, Optional, TypeVar

import random
from opik import opik_context
from baml_py import Collector
from opik.evaluation.metrics import Hallucination, Contains

T = TypeVar('T')


class BAMLInstrumentation:
    """
    A class for instrumenting BAML calls with Opik tracking.
    
    This class provides a reusable way to track BAML function calls with
    consistent metadata and usage tracking.
    """
    
    def __init__(self, collector_name: str, sample_rate: float = 0.05):
        """
        Initialize the instrumentation with a collector name.
        
        Args:
            collector_name: Name for the BAML collector
            sample_rate: Fraction of calls to sample for metrics (default: 0.05)
        """
        self.collector_name = collector_name
        self.collector = Collector(name=collector_name)
        self.sample_rate = sample_rate
    
    async def track_call(
        self,
        baml_function: Callable,
        span_name: str,
        *args,
        input: str = None,
        output: str = None,
        context: list = None,
        metrics: list = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Track a BAML function call with Opik instrumentation and optionally run metrics.
        
        Args:
            baml_function: The BAML function to call
            span_name: Name for the Opik span
            *args: Arguments to pass to the BAML function
            input: Input string to the LLM (for metrics)
            output: Output string from the LLM (for metrics)
            context: Optional context list (for metrics)
            metrics: List of metric configs to run (e.g., [{"type": "Hallucination", "params": {}}])
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
        
        # Run metrics on a sample of calls if metrics are provided
        if (input is not None and output is not None and 
            metrics is not None and random.random() < self.sample_rate):
            metric_results = []
            for metric_cfg in metrics:
                metric_type = metric_cfg["type"]
                params = metric_cfg.get("params", {})
                if metric_type == "Hallucination":
                    metric = Hallucination(**params)
                    score_result = metric.score(input=input, output=output, context=context)
                elif metric_type == "Contains":
                    metric = Contains(**params)
                    reference = params.get("reference", "")
                    score_result = metric.score(output, reference)
                else:
                    continue  # Unknown metric type
                metric_results.append({
                    "metric": metric_type,
                    "value": getattr(score_result, "value", None),
                    "reason": getattr(score_result, "reason", None),
                })
            # Attach metric results to Opik span
            opik_context.update_current_span(
                name=span_name,
                metadata={"metrics": metric_results}
            )
        
        return result
    
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
                
                opik_context.update_current_span(
                    name=span_name,
                    metadata=metadata,
                    usage=usage,
                    provider=call.provider,
                    model=call.client_name,
                )


async def track_baml_call(
    baml_function: Callable,
    collector_name: str,
    span_name: str,
    *args,
    input: str = None,
    output: str = None,
    context: list = None,
    metrics: list = None,
    sample_rate: float = 0.05,
    additional_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Utility function to track a single BAML call with Opik instrumentation and metrics.
    
    Args:
        baml_function: The BAML function to call
        collector_name: Name for the BAML collector
        span_name: Name for the Opik span
        *args: Arguments to pass to the BAML function
        input: Input string to the LLM (for metrics)
        output: Output string from the LLM (for metrics)
        context: Optional context list (for metrics)
        metrics: List of metric configs to run (e.g., [{"type": "Hallucination", "params": {}}])
        sample_rate: Fraction of calls to sample for metrics (default: 0.05)
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
        input=input,
        output=output,
        context=context,
        metrics=metrics,
        additional_metadata=additional_metadata,
        **kwargs
    ) 