from __future__ import annotations

from typing import List, Optional

from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedSignature,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator


class RemoveStopwordsExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Remove stopwords from text.

    This unified expression removes common stopwords from text columns while preserving
    semantic content. It supports multiple languages and custom stopword lists.
    Multiple consecutive removed stopwords are collapsed into a single space.

    Args:
        column: The input text column expression
        language: Language code string (e.g., "en", "es", "fr", "de", "it", "pt")
        custom_stopwords: Optional list of custom stopwords to use instead of language defaults

    Example:
        >>> # Using language-specific stopwords
        >>> RemoveStopwordsExpr(col("text"), lit("en"))

        >>> # Using custom stopwords
        >>> RemoveStopwordsExpr(col("text"), lit("en"), lit(["custom", "words"]))
    """

    function_name = "text.remove_stopwords"

    def __init__(
        self,
        column: LogicalExpr,
        language: LogicalExpr,
        custom_stopwords: Optional[LogicalExpr] = None,
    ):
        self.column = column
        self.language = language
        self.custom_stopwords = custom_stopwords
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        if self.custom_stopwords is not None:
            return [self.column, self.language, self.custom_stopwords]
        return [self.column, self.language]


class DetectLanguageExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Detect the language of text content."""

    function_name = "text.detect_language"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class DetectLanguageWithConfidenceExpr(
    ValidatedSignature, UnparameterizedExpr, LogicalExpr
):
    """Detect language with confidence scores."""

    function_name = "text.detect_language_with_confidence"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]
