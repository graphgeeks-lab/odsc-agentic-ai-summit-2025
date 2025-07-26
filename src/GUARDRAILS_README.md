# Custom Guardrails for Opik Cloud Version

This module provides guardrail-like functionality for the Opik cloud version, since the official guardrails feature is only available in self-hosted installations. The guardrails check inputs and outputs for various risks and can be integrated with the existing Opik instrumentation system.

## Features

- **Email Detection**: Comprehensive email address detection using regex patterns
- **Email Masking**: Automatic masking of detected email addresses
- **Domain Filtering**: Whitelist/blacklist for specific email domains
- **Action Types**: Support for BLOCK, WARN, and LOG actions
- **Severity Levels**: Configurable severity levels (LOW, MEDIUM, HIGH, CRITICAL)
- **Opik Integration**: Full integration with Opik tracking and observability
- **Multiple Guardrails**: Support for running multiple guardrails simultaneously

## Quick Start

### Basic Email Detection

```python
from guardrails import EmailGuardrail, GuardrailAction, GuardrailSeverity

# Create a simple email guardrail
guardrail = EmailGuardrail(
    action=GuardrailAction.WARN,
    severity=GuardrailSeverity.MEDIUM
)

# Validate text
text = "Contact me at john.doe@example.com"
result = guardrail.validate(text)

print(f"Triggered: {result.triggered}")
print(f"Entities found: {result.entities_found}")
```

### Email Masking

```python
guardrail = EmailGuardrail(
    action=GuardrailAction.WARN,
    severity=GuardrailSeverity.MEDIUM,
    mask_emails=True,
    mask_char="*"
)

text = "Contact support@company.com or sales@business.org"
masked_text = guardrail.mask_text(text)
# Result: "Contact s****t@c****y.com or s***s@b****s.org"
```

### Domain Filtering

```python
guardrail = EmailGuardrail(
    action=GuardrailAction.BLOCK,
    severity=GuardrailSeverity.HIGH,
    block_common_domains=True,
    allowed_domains=["company.com"],
    blocked_domains=["competitor.com"]
)

# This will trigger the guardrail
text = "Contact user@gmail.com and admin@competitor.com"
result = guardrail.validate(text)
```

### Integration with Opik

```python
from guardrails import validate_input_with_guardrails, validate_output_with_guardrails

# Validate input
guardrails = [EmailGuardrail(action=GuardrailAction.WARN, mask_emails=True)]
processed_input = validate_input_with_guardrails(
    user_input,
    guardrails,
    "input_validation"
)

# Validate output
processed_output = validate_output_with_guardrails(
    llm_response,
    guardrails,
    "output_validation"
)
```

## Integration with RAG Pipeline

The guardrails have been integrated into the existing RAG evaluation script (`rag_obs_eval.py`). To enable guardrails:

1. Set the environment variable:
   ```bash
   export GUARDRAILS_ENABLED=true
   ```

2. The system will automatically:
   - Validate user inputs for email addresses
   - Mask detected emails in inputs
   - Validate LLM outputs for email addresses
   - Mask detected emails in outputs
   - Log all guardrail activities to Opik

## Configuration Options

### EmailGuardrail Parameters

- `action`: What to do when emails are detected
  - `GuardrailAction.BLOCK`: Raise exception and stop processing
  - `GuardrailAction.WARN`: Log warning and optionally mask emails
  - `GuardrailAction.LOG`: Just log the detection

- `severity`: Severity level for violations
  - `GuardrailSeverity.LOW`: Minor concern
  - `GuardrailSeverity.MEDIUM`: Moderate concern
  - `GuardrailSeverity.HIGH`: High concern
  - `GuardrailSeverity.CRITICAL`: Critical concern

- `block_common_domains`: Whether to block common email domains (gmail, yahoo, etc.)
- `allowed_domains`: Whitelist of allowed email domains
- `blocked_domains`: Blacklist of blocked email domains
- `mask_emails`: Whether to mask detected emails
- `mask_char`: Character to use for masking (default: "*")

## Examples

### Test the Guardrails

Run the test script to see all features in action:

```bash
cd src
python test_guardrails.py
```

### Run the Example Application

See how guardrails work in a practical RAG application:

```bash
cd src
python guardrail_example.py
```

### Run RAG Evaluation with Guardrails

```bash
cd src
export GUARDRAILS_ENABLED=true
python rag_obs_eval.py
```

## Opik Integration

All guardrail activities are automatically logged to Opik with comprehensive tracing and span integration:

### Basic Integration
- **Spans**: Each guardrail validation creates a span
- **Metadata**: Detailed information about what was detected
- **Entities**: List of found email addresses
- **Actions**: What action was taken (block, warn, log)
- **Severity**: Severity level of the violation

### Enhanced Integration
- **Detailed Metrics**: Processing time, entity counts, severity distributions
- **Custom Tags**: Automatic tagging based on guardrail results
- **Trace Summaries**: Aggregated guardrail statistics at the trace level
- **Span History**: Complete history of guardrail validations for analysis

### Integration Methods

#### 1. Basic Span Integration
```python
# Simple guardrail validation with automatic span creation
result = guardrail.validate(text)
# Span is automatically created with guardrail results
```

#### 2. Enhanced Tracing
```python
# Use EnhancedGuardrailManager for comprehensive tracing
manager = EnhancedGuardrailManager([guardrails])
processed_text = await manager.validate_with_detailed_tracing(
    text,
    span_name="custom_validation",
    trace_tags=["custom", "validation"],
    custom_metadata={"environment": "production"}
)
```

#### 3. Decorator Integration
```python
# Automatic guardrail tracing with decorators
@GuardrailTraceDecorator([guardrails], "decorated_function")
async def process_text(text: str) -> str:
    return f"Processed: {text}"
```

#### 4. Trace-Level Integration
```python
# Update trace with guardrail summaries
opik_context.update_current_trace(
    metadata={
        "guardrail_summary": {
            "total_validations": 10,
            "total_entities_found": 5,
            "total_blocks": 1,
            "total_warnings": 3
        }
    }
)
```

### Viewing in Opik Dashboard

You can view guardrail activities in your Opik dashboard by:

1. **Filtering by span names** containing "guardrail"
2. **Filtering by tags** like "guardrail", "email", "validation"
3. **Searching metadata** for guardrail-specific fields
4. **Viewing trace summaries** for aggregated guardrail statistics

### Enhanced Span Metadata Structure

Each guardrail span now includes enhanced metadata that clearly distinguishes between input and output validation:

```json
{
  "guardrail_type": "email",
  "processing_time_ms": 15.2,
  "input_length": 45,
  "output_length": 45,
  "guardrails_run": 1,
  "guardrails_triggered": 1,
  "text_modified": true,
  "validation_phase": "input",
  "masking_applied": true,
  "masking_details": [
    {
      "guardrail_type": "email",
      "masking_method": "character_replacement",
      "mask_char": "*",
      "original_length": 45,
      "masked_length": 45
    }
  ],
  "entities_found": ["user@example.com"],
  "severity_distribution": {"medium": 1},
  "action_distribution": {"warn": 1},
  "results": [
    {
      "action": "warn",
      "severity": "medium",
      "triggered": true,
      "message": "Email addresses detected: user@example.com",
      "entities_found": ["user@example.com"],
      "details": {
        "total_emails": 1,
        "blocked_emails": [],
        "allowed_emails": ["user@example.com"]
      }
    }
  ]
}
```

### Key Enhanced Features

#### 1. **Validation Phase Distinction**
- `validation_phase`: Clearly identifies whether this is "input" or "output" validation
- Different span names and tags for input vs output events

#### 2. **Masking Action Tracking**
- `masking_applied`: Boolean indicating if masking occurred
- `masking_details`: Detailed information about masking operations
- `text_modified`: Boolean indicating if text was modified by any guardrail action

#### 3. **Enhanced Results**
- `triggered`: Boolean indicating if each guardrail was triggered
- `details`: Comprehensive details about what was detected and why

#### 4. **Workflow Integration**
- `workflow_step`: Identifies the step in the RAG workflow
- `rag_type`: Type of RAG system being used
- `question_number`: For tracking specific questions in evaluation suites

## Error Handling

The guardrails include comprehensive error handling:

- `GuardrailValidationFailed`: Raised when a BLOCK action is triggered
- Graceful degradation: If guardrails fail, the main application continues
- Detailed logging: All activities are logged for debugging

## Extending the Guardrails

To add new guardrail types, follow the pattern of the `EmailGuardrail` class:

1. Create a new guardrail class
2. Implement the `validate()` method
3. Return a `GuardrailResult` object
4. Add it to the `GuardrailManager`

Example:

```python
class PhoneGuardrail:
    def __init__(self, action, severity):
        self.action = action
        self.severity = severity
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    
    def validate(self, text: str) -> GuardrailResult:
        # Implementation here
        pass
```

## Performance Considerations

- Email detection uses regex patterns for fast matching
- Guardrails are designed to be lightweight and non-blocking
- Opik integration is asynchronous and doesn't impact response times
- Masking operations are efficient string replacements

## Security Features

- **Input Sanitization**: Emails in user inputs are masked before processing
- **Output Protection**: Emails in LLM responses are masked before delivery
- **Domain Control**: Whitelist/blacklist for specific domains
- **Common Domain Blocking**: Option to block common personal email domains
- **Audit Trail**: All activities logged to Opik for compliance

## Troubleshooting

### Guardrails Not Working

1. Check that `GUARDRAILS_ENABLED=true` is set
2. Verify Opik configuration is correct
3. Check console output for guardrail messages

### Performance Issues

1. Reduce the number of guardrails if needed
2. Use LOG action instead of WARN for better performance
3. Consider disabling guardrails for high-throughput scenarios

### Opik Integration Issues

1. Verify Opik API key and workspace are set
2. Check network connectivity to Opik
3. Review Opik dashboard for error messages 