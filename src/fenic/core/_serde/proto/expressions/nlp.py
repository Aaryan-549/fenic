"""NLP expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.nlp import (
    DetectLanguageExpr,
    DetectLanguageWithConfidenceExpr,
    RemoveStopwordsExpr,
)

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    DetectLanguageExprProto,
    DetectLanguageWithConfidenceExprProto,
    LogicalExprProto,
    RemoveStopwordsExprProto,
)

# =============================================================================
# RemoveStopwordsExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_remove_stopwords_expr(
    logical: RemoveStopwordsExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a remove stopwords expression with optional custom_stopwords."""
    proto = RemoveStopwordsExprProto(
        column=context.serialize_logical_expr("column", logical.column),
        language=context.serialize_logical_expr("language", logical.language),
    )

    if logical.custom_stopwords is not None:
        proto.custom_stopwords.CopyFrom(
            context.serialize_logical_expr("custom_stopwords", logical.custom_stopwords)
        )

    return LogicalExprProto(remove_stopwords=proto)


@_deserialize_logical_expr_helper.register
def _deserialize_remove_stopwords_expr(
    logical_proto: RemoveStopwordsExprProto, context: SerdeContext
) -> RemoveStopwordsExpr:
    """Deserialize a remove stopwords expression."""
    custom_stopwords = None
    if logical_proto.HasField("custom_stopwords"):
        custom_stopwords = context.deserialize_logical_expr(
            "custom_stopwords", logical_proto.custom_stopwords
        )

    return RemoveStopwordsExpr(
        column=context.deserialize_logical_expr("column", logical_proto.column),
        language=context.deserialize_logical_expr("language", logical_proto.language),
        custom_stopwords=custom_stopwords,
    )


# =============================================================================
# DetectLanguageExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_detect_language_expr(
    logical: DetectLanguageExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        detect_language=DetectLanguageExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_detect_language_expr(
    logical_proto: DetectLanguageExprProto, context: SerdeContext
) -> DetectLanguageExpr:
    return DetectLanguageExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
    )


# =============================================================================
# DetectLanguageWithConfidenceExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_detect_language_with_confidence_expr(
    logical: DetectLanguageWithConfidenceExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        detect_language_with_confidence=DetectLanguageWithConfidenceExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_detect_language_with_confidence_expr(
    logical_proto: DetectLanguageWithConfidenceExprProto, context: SerdeContext
) -> DetectLanguageWithConfidenceExpr:
    return DetectLanguageWithConfidenceExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
    )
