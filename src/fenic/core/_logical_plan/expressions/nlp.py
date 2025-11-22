from __future__ import annotations

from typing import List

from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedSignature,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator


class RemoveStopwordsExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Remove stopwords from text based on language."""

    function_name = "text.remove_stopwords"

    def __init__(self, expr: LogicalExpr, language: LogicalExpr):
        self.expr = expr
        self.language = language
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.language]


class RemoveCustomStopwordsExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Remove custom stopwords from text."""

    function_name = "text.remove_custom_stopwords"

    def __init__(self, expr: LogicalExpr, stopwords: LogicalExpr):
        self.expr = expr
        self.stopwords = stopwords
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.stopwords]


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


class DetectLanguageWithConfidenceExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
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
