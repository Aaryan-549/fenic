"""NLP expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.nlp import (
    DetectLanguageExpr,
    DetectLanguageWithConfidenceExpr,
    RemoveCustomStopwordsExpr,
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
    RemoveCustomStopwordsExprProto,
    RemoveStopwordsExprProto,
)

# =============================================================================
# RemoveStopwordsExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_remove_stopwords_expr(
    logical: RemoveStopwordsExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        remove_stopwords=RemoveStopwordsExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            language=context.serialize_logical_expr(SerdeContext.EXPR, logical.language),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_remove_stopwords_expr(
    logical_proto: RemoveStopwordsExprProto, context: SerdeContext
) -> RemoveStopwordsExpr:
    return RemoveStopwordsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        language=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.language),
    )


# =============================================================================
# RemoveCustomStopwordsExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_remove_custom_stopwords_expr(
    logical: RemoveCustomStopwordsExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        remove_custom_stopwords=RemoveCustomStopwordsExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            stopwords=context.serialize_logical_expr(SerdeContext.EXPR, logical.stopwords),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_remove_custom_stopwords_expr(
    logical_proto: RemoveCustomStopwordsExprProto, context: SerdeContext
) -> RemoveCustomStopwordsExpr:
    return RemoveCustomStopwordsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        stopwords=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.stopwords),
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
