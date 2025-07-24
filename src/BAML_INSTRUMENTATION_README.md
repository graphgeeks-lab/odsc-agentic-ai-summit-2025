# BAML Instrumentation Utilities

This module provides reusable utilities for instrumenting BAML calls with Opik observability tracking. It eliminates the repetitive code pattern of manually creating collectors, calling BAML functions, and updating Opik contexts.

## Overview

The instrumentation utilities provide two main approaches:

1. **Class-based approach**: Create an instance of `BAMLInstrumentation` and reuse it for multiple calls
2. **Utility function approach**: Use `track_baml_call()` for one-off instrumentation

## Usage Examples

### 1. Class-based Approach

```python
from baml_instrumentation import BAMLInstrumentation

# Create an instrumentation instance
instrumentation = BAMLInstrumentation("my_collector")

# Track a BAML call
result = await instrumentation.track_call(
    b.AnswerQuestion,
    "my_span",
    "What is 2+2?",
    "The answer is 4.",
    additional_metadata={"example_type": "class_based"}
)
```

**Benefits:**
- Reuse the same instance for multiple calls
- Consistent collector naming
- Easy to manage lifecycle

### 2. Utility Function Approach

```python
from baml_instrumentation import track_baml_call

result = await track_baml_call(
    b.AnswerQuestion,
    "my_collector",
    "my_span",
    "What is 2+2?",
    "The answer is 4.",
    additional_metadata={"example_type": "utility_function"}
)
```

**Benefits:**
- Simple one-liner for single calls
- No need to manage instances
- Good for ad-hoc instrumentation

## Before vs After

### Before (Manual Approach)

```python
@opik.track(flush=True)
async def prune_schema(question: str) -> str:
    # Initialize BAML Collector
    collector = Collector(name="prune_schema_collector")
    
    pruned_schema = await b.PruneSchema(schema_xml, question, baml_options={"collector": [collector]})
    
    # Update opik context with BAML data
    if collector.last is not None:
        log = collector.last
        call = log.calls[0] if log.calls else None
        
        if call and call.usage:
            opik_context.update_current_span(
                name="pruned_schema",
                metadata={
                    "function_name": log.function_name,
                    "duration_ms": call.timing.duration_ms if call.timing else None,
                },
                usage={
                    "prompt_tokens": call.usage.input_tokens,
                    "completion_tokens": call.usage.output_tokens,
                    "total_tokens": (call.usage.input_tokens or 0) + (call.usage.output_tokens or 0),
                },
                provider=call.provider,
                model=call.client_name,
            )
    
    return pruned_schema_xml
```

### After (Using Instrumentation Utilities)

```python
@opik.track(flush=True)
async def prune_schema(question: str) -> str:
    pruned_schema = await track_baml_call(
        b.PruneSchema,
        "prune_schema_collector",
        "pruned_schema",
        schema_xml,
        question
    )
    
    return pruned_schema_xml
```

## API Reference

### BAMLInstrumentation Class

```python
class BAMLInstrumentation:
    def __init__(self, collector_name: str)
    async def track_call(
        self,
        baml_function: Callable,
        span_name: str,
        *args,
        additional_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any
```

### track_baml_call Function

```python
async def track_baml_call(
    baml_function: Callable,
    collector_name: str,
    span_name: str,
    *args,
    additional_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any
```

## Migration Guide

To migrate existing code:

1. **Import the utilities:**
   ```python
   from baml_instrumentation import track_baml_call
   ```

2. **Replace manual collector creation and tracking:**
   ```python
   # Old
   collector = Collector(name="my_collector")
   result = await b.MyFunction(args, baml_options={"collector": [collector]})
   # ... manual opik_context.update_current_span() ...
   
   # New
   result = await track_baml_call(
       b.MyFunction,
       "my_collector",
       "my_span",
       args
   )
   ```

3. **For functions with multiple BAML calls, use the class-based approach:**
   ```python
   instrumentation = BAMLInstrumentation("my_collector")
   
   result1 = await instrumentation.track_call(b.Function1, "span1", args1)
   result2 = await instrumentation.track_call(b.Function2, "span2", args2)
   ```

## Benefits

- **Reduced boilerplate**: Eliminates repetitive collector creation and tracking code
- **Consistency**: Ensures all BAML calls are tracked the same way
- **Maintainability**: Changes to tracking logic only need to be made in one place
- **Flexibility**: Multiple approaches for different use cases
- **Type safety**: Full type hints for better IDE support

## See Also

- `baml_instrumentation_example.py` - Complete examples of all approaches
- `rag_refactored.py` - Real-world example of refactored RAG code 