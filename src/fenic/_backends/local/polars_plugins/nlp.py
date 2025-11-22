from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


@pl.api.register_expr_namespace("nlp")
class NLP:
    """Namespace for NLP text preprocessing operations on Polars expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize an NLP Namespace with a Polars expression.

        Args:
            expr: A Polars expression containing the text data for NLP operations.
        """
        self.expr = expr

    def remove_stopwords(self, language: IntoExpr) -> pl.Expr:
        """Remove stopwords from text based on language.

        Args:
            language: ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'it', 'pt').

        Returns:
            Text with stopwords removed, or null if input is null.
        """
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="remove_stopwords",
            args=[self.expr, language],
            is_elementwise=True,
        )

    def remove_custom_stopwords(self, stopwords: IntoExpr) -> pl.Expr:
        """Remove custom stopwords from text.

        Args:
            stopwords: List of custom stopwords to remove (case-insensitive).

        Returns:
            Text with custom stopwords removed, or null if input is null.
        """
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="remove_custom_stopwords",
            args=[self.expr, stopwords],
            is_elementwise=True,
        )

    def detect_language(self) -> pl.Expr:
        """Detect the language of text content.

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr'), or null if detection fails.
        """
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="detect_language",
            args=[self.expr],
            is_elementwise=True,
        )

    def detect_language_with_confidence(self) -> pl.Expr:
        """Detect language with confidence scores.

        Returns:
            Struct with 'language' (str) and 'confidence' (float) fields, or null if detection fails.
        """
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="detect_language_with_confidence",
            args=[self.expr],
            is_elementwise=True,
        )
