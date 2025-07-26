"""
Custom Guardrails for Opik Cloud Version

This module provides guardrail-like functionality for the Opik cloud version,
since the official guardrails feature is only available in self-hosted installations.

The guardrails check inputs and outputs for various risks and can be integrated
with the existing Opik instrumentation system.
"""

import re
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from opik import opik_context


class GuardrailAction(Enum):
    """Actions to take when a guardrail is triggered."""
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"


class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    triggered: bool
    action: GuardrailAction
    severity: GuardrailSeverity
    message: str
    details: Dict[str, Any]
    entities_found: List[str] = None


class GuardrailValidationFailed(Exception):
    """Exception raised when a guardrail validation fails."""
    
    def __init__(self, message: str, guardrail_result: GuardrailResult):
        super().__init__(message)
        self.guardrail_result = guardrail_result


class EmailGuardrail:
    """
    Guardrail to detect email addresses in text.
    
    This guardrail uses regex patterns to identify email addresses and can be
    configured to block, warn, or log when emails are detected.
    """
    
    def __init__(
        self,
        action: GuardrailAction = GuardrailAction.WARN,
        severity: GuardrailSeverity = GuardrailSeverity.MEDIUM,
        block_common_domains: bool = False,
        allowed_domains: List[str] = None,
        blocked_domains: List[str] = None,
        mask_emails: bool = False,
        mask_char: str = "*"
    ):
        """
        Initialize the email guardrail.
        
        Args:
            action: Action to take when emails are detected
            severity: Severity level for violations
            block_common_domains: Whether to block common email domains (gmail, yahoo, etc.)
            allowed_domains: List of allowed email domains (whitelist)
            blocked_domains: List of blocked email domains (blacklist)
            mask_emails: Whether to mask detected emails in the output
            mask_char: Character to use for masking
        """
        self.action = action
        self.severity = severity
        self.block_common_domains = block_common_domains
        self.allowed_domains = allowed_domains or []
        self.blocked_domains = blocked_domains or []
        self.mask_emails = mask_emails
        self.mask_char = mask_char
        
        # Email regex pattern (comprehensive)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Common email domains to potentially block
        self.common_domains = {
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'icloud.com', 'protonmail.com', 'mail.com'
        }
    
    def validate(self, text: str) -> GuardrailResult:
        """
        Validate text for email addresses.
        
        Args:
            text: Text to validate
            
        Returns:
            GuardrailResult with validation details
        """
        if not text:
            return GuardrailResult(
                triggered=False,
                action=self.action,
                severity=self.severity,
                message="No text provided for validation",
                details={},
                entities_found=[]
            )
        
        # Find all email addresses
        emails = self.email_pattern.findall(text)
        
        if not emails:
            return GuardrailResult(
                triggered=False,
                action=self.action,
                severity=self.severity,
                message="No email addresses detected",
                details={"total_emails": 0},
                entities_found=[]
            )
        
        # Analyze found emails
        email_details = []
        blocked_emails = []
        allowed_emails = []
        
        for email in emails:
            domain = email.split('@')[1].lower()
            is_common = domain in self.common_domains
            is_allowed = domain in self.allowed_domains if self.allowed_domains else True
            is_blocked = domain in self.blocked_domains
            
            email_info = {
                "email": email,
                "domain": domain,
                "is_common": is_common,
                "is_allowed": is_allowed,
                "is_blocked": is_blocked
            }
            email_details.append(email_info)
            
            if is_blocked:
                blocked_emails.append(email)
            elif is_allowed:
                allowed_emails.append(email)
        
        # Determine if guardrail should be triggered
        should_trigger = False
        trigger_reasons = []
        
        if self.blocked_domains and blocked_emails:
            should_trigger = True
            trigger_reasons.append(f"Blocked domains detected: {', '.join(blocked_emails)}")
        
        if self.block_common_domains:
            common_emails = [e for e in emails if e.split('@')[1].lower() in self.common_domains]
            if common_emails:
                should_trigger = True
                trigger_reasons.append(f"Common domains detected: {', '.join(common_emails)}")
        
        # If no specific blocking rules, trigger on any email if action is BLOCK
        if not should_trigger and self.action == GuardrailAction.BLOCK and emails:
            should_trigger = True
            trigger_reasons.append(f"Email addresses detected: {', '.join(emails)}")
        
        # Create result
        result = GuardrailResult(
            triggered=should_trigger,
            action=self.action,
            severity=self.severity,
            message="; ".join(trigger_reasons) if trigger_reasons else f"Email addresses detected: {', '.join(emails)}",
            details={
                "total_emails": len(emails),
                "blocked_emails": blocked_emails,
                "allowed_emails": allowed_emails,
                "email_details": email_details,
                "block_common_domains": self.block_common_domains,
                "has_blocked_domains": bool(self.blocked_domains),
                "has_allowed_domains": bool(self.allowed_domains)
            },
            entities_found=emails
        )
        
        return result
    
    def mask_text(self, text: str) -> str:
        """
        Mask email addresses in text.
        
        Args:
            text: Text containing email addresses
            
        Returns:
            Text with masked email addresses
        """
        if not self.mask_emails:
            return text
        
        def mask_email(match):
            email = match.group(0)
            parts = email.split('@')
            username = parts[0]
            domain = parts[1]
            
            # Mask username (keep first and last character)
            if len(username) <= 2:
                masked_username = self.mask_char * len(username)
            else:
                masked_username = username[0] + self.mask_char * (len(username) - 2) + username[-1]
            
            # Mask domain (keep first character of each part)
            domain_parts = domain.split('.')
            masked_domain_parts = []
            for part in domain_parts:
                if len(part) <= 1:
                    masked_domain_parts.append(part)
                else:
                    masked_domain_parts.append(part[0] + self.mask_char * (len(part) - 1))
            
            masked_domain = '.'.join(masked_domain_parts)
            return f"{masked_username}@{masked_domain}"
        
        return self.email_pattern.sub(mask_email, text)


class GuardrailManager:
    """
    Manager for multiple guardrails.
    
    This class allows you to run multiple guardrails on the same text
    and handle the results appropriately.
    """
    
    def __init__(self, guardrails: List[Any] = None):
        """
        Initialize the guardrail manager.
        
        Args:
            guardrails: List of guardrail instances
        """
        self.guardrails = guardrails or []
    
    def add_guardrail(self, guardrail: Any) -> None:
        """Add a guardrail to the manager."""
        self.guardrails.append(guardrail)
    
    def validate(self, text: str) -> List[GuardrailResult]:
        """
        Run all guardrails on the given text.
        
        Args:
            text: Text to validate
            
        Returns:
            List of guardrail results
        """
        results = []
        for guardrail in self.guardrails:
            result = guardrail.validate(text)
            results.append(result)
        return results
    
    def validate_and_handle(self, text: str, span_name: str = "guardrail_validation") -> str:
        """
        Validate text and handle results according to guardrail actions.
        
        Args:
            text: Text to validate
            span_name: Name for Opik span
            
        Returns:
            Processed text (potentially masked)
            
        Raises:
            GuardrailValidationFailed: If any guardrail with BLOCK action is triggered
        """
        results = self.validate(text)
        
        # Log results to Opik
        triggered_results = [r for r in results if r.triggered]
        all_entities = []
        for result in results:
            if result.entities_found:
                all_entities.extend(result.entities_found)
        
        # Update Opik context
        try:
            opik_context.update_current_span(
                name=span_name,
                metadata={
                    "guardrails_run": len(results),
                    "guardrails_triggered": len(triggered_results),
                    "entities_found": list(set(all_entities)),
                    "text_modified": processed_text != text,
                    "masking_applied": processed_text != text,
                    "validation_phase": "general",
                    "results": [
                        {
                            "action": result.action.value,
                            "severity": result.severity.value,
                            "triggered": result.triggered,
                            "message": result.message,
                            "entities_found": result.entities_found,
                            "details": result.details
                        }
                        for result in results
                    ]
                }
            )
        except Exception as e:
            # Silently handle errors when updating spans
            # This prevents guardrail errors from breaking the main application
            pass
        
        # Handle triggered guardrails
        processed_text = text
        for result in triggered_results:
            if result.action == GuardrailAction.BLOCK:
                raise GuardrailValidationFailed(
                    f"Guardrail validation failed: {result.message}",
                    result
                )
            elif result.action == GuardrailAction.WARN:
                print(f"[WARNING] Guardrail triggered: {result.message}")
                # If it's an email guardrail and masking is enabled, mask the text
                if hasattr(result, 'mask_emails') and result.mask_emails:
                    processed_text = self._mask_emails_in_text(processed_text)
            elif result.action == GuardrailAction.LOG:
                print(f"[INFO] Guardrail logged: {result.message}")
        
        return processed_text
    
    def _mask_emails_in_text(self, text: str) -> str:
        """Helper method to mask emails in text."""
        for guardrail in self.guardrails:
            if isinstance(guardrail, EmailGuardrail) and guardrail.mask_emails:
                text = guardrail.mask_text(text)
        return text


# Convenience functions for easy integration
def create_email_guardrail(
    action: str = "warn",
    severity: str = "medium",
    **kwargs
) -> EmailGuardrail:
    """
    Create an email guardrail with string-based configuration.
    
    Args:
        action: Action as string ("block", "warn", "log")
        severity: Severity as string ("low", "medium", "high", "critical")
        **kwargs: Additional arguments for EmailGuardrail
        
    Returns:
        Configured EmailGuardrail instance
    """
    action_enum = GuardrailAction(action.lower())
    severity_enum = GuardrailSeverity(severity.lower())
    
    return EmailGuardrail(
        action=action_enum,
        severity=severity_enum,
        **kwargs
    )


def validate_input_with_guardrails(
    text: str,
    guardrails: List[Any],
    span_name: str = "input_guardrail_validation"
) -> str:
    """
    Validate input text with guardrails and log to Opik.
    
    Args:
        text: Text to validate
        guardrails: List of guardrail instances
        span_name: Name for Opik span
        
    Returns:
        Processed text
        
    Raises:
        GuardrailValidationFailed: If any guardrail blocks the input
    """
    manager = GuardrailManager(guardrails)
    return manager.validate_and_handle(text, span_name)


def validate_output_with_guardrails(
    text: str,
    guardrails: List[Any],
    span_name: str = "output_guardrail_validation"
) -> str:
    """
    Validate output text with guardrails and log to Opik.
    
    Args:
        text: Text to validate
        guardrails: List of guardrail instances
        span_name: Name for Opik span
        
    Returns:
        Processed text
        
    Raises:
        GuardrailValidationFailed: If any guardrail blocks the output
    """
    manager = GuardrailManager(guardrails)
    return manager.validate_and_handle(text, span_name) 