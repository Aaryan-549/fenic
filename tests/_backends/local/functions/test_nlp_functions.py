"""Tests for NLP text preprocessing functions."""

import pytest

from fenic import col, text


# Test data constants
ENGLISH_TEXT_1 = "The quick brown fox jumps over the lazy dog"
ENGLISH_TEXT_1_CLEANED = "quick brown fox jumps lazy dog"

ENGLISH_TEXT_2 = "Machine learning is a subset of artificial intelligence"
ENGLISH_TEXT_2_CLEANED = "Machine learning subset artificial intelligence"

ENGLISH_TEXT_3 = "This is a test sentence with many stopwords"


@pytest.fixture
def stopwords_test_df(local_session):
    """DataFrame with text for stopword removal testing."""
    data = {
        "text": [
            ENGLISH_TEXT_1,
            ENGLISH_TEXT_2,
            ENGLISH_TEXT_3,
            None,  # Test null handling
            "",  # Test empty string
        ]
    }
    return local_session.create_dataframe(data)


@pytest.fixture
def multilingual_test_df(local_session):
    """DataFrame with multilingual text for testing."""
    data = {
        "text": [
            "The quick brown fox jumps over the lazy dog",  # English
            "El rápido zorro marrón salta sobre el perro perezoso",  # Spanish
            "Le rapide renard brun saute par-dessus le chien paresseux",  # French
            "Der schnelle braune Fuchs springt über den faulen Hund",  # German
            "La volpe marrone veloce salta sopra il cane pigro",  # Italian
            "A rápida raposa marrom salta sobre o cão preguiçoso",  # Portuguese
        ]
    }
    return local_session.create_dataframe(data)


def test_remove_stopwords_english(stopwords_test_df):
    """Test English stopword removal."""
    result = stopwords_test_df.select(
        text.remove_stopwords(col("text"))
    ).to_polars()

    # Check cleaned text matches expected output
    assert result["text"][0] == ENGLISH_TEXT_1_CLEANED
    assert result["text"][1] == ENGLISH_TEXT_2_CLEANED

    # Test null handling
    assert result["text"][3] is None

    # Test empty string
    assert result["text"][4] == ""


def test_remove_stopwords_spanish(multilingual_test_df):
    """Test Spanish stopword removal."""
    # Get the Spanish text (index 1)
    df = multilingual_test_df.filter(col("text").contains("rápido"))

    result = df.select(
        text.remove_stopwords(col("text"), language="es")
    ).to_polars()

    text_result = result["text"][0]

    # "El", "sobre", "el" should be removed (Spanish stopwords)
    assert "El" not in text_result or "el" not in text_result.lower()
    assert "rápido" in text_result
    assert "zorro" in text_result


def test_remove_stopwords_french(multilingual_test_df):
    """Test French stopword removal."""
    # Get the French text (index 2)
    df = multilingual_test_df.filter(col("text").contains("rapide"))

    result = df.select(
        text.remove_stopwords(col("text"), language="fr")
    ).to_polars()

    text_result = result["text"][0]

    # "Le", "par-dessus", "le" should have stopwords removed
    assert "rapide" in text_result
    assert "renard" in text_result


def test_remove_custom_stopwords(stopwords_test_df):
    """Test custom stopword removal."""
    custom_stopwords = ["quick", "brown", "machine", "learning"]

    result = stopwords_test_df.select(
        text.remove_stopwords(col("text"), custom_stopwords=custom_stopwords)
    ).to_polars()

    # Custom stopwords should be removed
    assert "quick" not in result["text"][0].lower() if result["text"][0] else True
    assert "brown" not in result["text"][0].lower() if result["text"][0] else True
    assert "fox" in result["text"][0] if result["text"][0] else False

    # "machine" and "learning" should be removed from second row
    if result["text"][1]:
        assert "machine" not in result["text"][1].lower()
        assert "learning" not in result["text"][1].lower()


def test_remove_stopwords_invalid_language(stopwords_test_df):
    """Test that invalid language code raises an error."""
    from polars.exceptions import ComputeError

    with pytest.raises(ComputeError, match="Unsupported language code"):
        stopwords_test_df.select(
            text.remove_stopwords(col("text"), language="invalid")
        ).collect()


def test_detect_language_basic(multilingual_test_df):
    """Test basic language detection."""
    result = multilingual_test_df.select(
        col("text"),
        text.detect_language(col("text")).alias("language")
    ).to_polars()

    # Check detected languages (may not be perfect, but should be close)
    # English text
    assert result["language"][0] == "en"

    # Spanish text
    assert result["language"][1] == "es"

    # French text
    assert result["language"][2] == "fr"

    # German text
    assert result["language"][3] == "de"

    # Italian text
    assert result["language"][4] == "it"

    # Portuguese text
    assert result["language"][5] == "pt"


def test_detect_language_with_confidence(multilingual_test_df):
    """Test language detection with confidence scores."""
    result = multilingual_test_df.select(
        col("text"),
        text.detect_language(col("text"), return_confidence=True).alias("lang_info")
    ).to_polars()

    # Check that we get structs with language and confidence fields
    assert "lang_info" in result.columns

    # Extract the struct fields
    first_result = result["lang_info"][0]
    assert "language" in first_result
    assert "confidence" in first_result

    # Check that language matches
    assert first_result["language"] == "en"

    # Check that confidence is between 0 and 1
    assert 0.0 <= first_result["confidence"] <= 1.0


def test_detect_language_null_handling(local_session):
    """Test language detection with null and empty strings."""
    data = {
        "text": [
            "This is English text",
            None,
            "",
            "   ",  # Just whitespace
        ]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.detect_language(col("text")).alias("language")
    ).to_polars()

    # First should detect as English
    assert result["language"][0] == "en"

    # Null input should return null
    assert result["language"][1] is None

    # Empty string should return null
    assert result["language"][2] is None

    # Whitespace might return null or a language (implementation dependent)
    # Just verify it doesn't crash
    assert result["language"][3] is None or isinstance(result["language"][3], str)


def test_stopwords_with_preprocessing_chain(stopwords_test_df):
    """Test stopword removal in a preprocessing chain."""
    # Chain operations: lowercase -> remove stopwords
    result = stopwords_test_df.select(
        text.remove_stopwords(
            text.lower(col("text"))
        ).alias("cleaned")
    ).to_polars()

    # Check that stopwords are removed and text is lowercase
    if result["cleaned"][0]:
        assert result["cleaned"][0].islower()
        assert "the" not in result["cleaned"][0]


def test_language_detection_with_conditional_preprocessing(multilingual_test_df):
    """Test using language detection to apply conditional preprocessing."""
    from fenic import when

    # Detect language first
    df = multilingual_test_df.with_column(
        "detected_lang",
        text.detect_language(col("text"))
    )

    # Apply language-specific stopword removal
    df = df.with_column(
        "cleaned",
        when(col("detected_lang") == "en", text.remove_stopwords(col("text"), language="en"))
        .when(col("detected_lang") == "es", text.remove_stopwords(col("text"), language="es"))
        .when(col("detected_lang") == "fr", text.remove_stopwords(col("text"), language="fr"))
        .otherwise(col("text"))
    )

    result = df.select("text", "detected_lang", "cleaned").to_polars()

    # Verify that cleaning was applied based on detected language
    for i in range(3):  # Check first 3 rows (en, es, fr)
        assert result["cleaned"][i] is not None
        # The cleaned text should be different from original (stopwords removed)
        assert result["cleaned"][i] != result["text"][i]


def test_remove_stopwords_case_insensitive(local_session):
    """Test that stopword removal is case-insensitive."""
    data = {
        "text": [
            "THE QUICK BROWN FOX",
            "the quick brown fox",
            "The Quick Brown Fox",
        ]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.remove_stopwords(col("text"))
    ).to_polars()

    # All variations should remove "THE"/"the"/"The"
    for i in range(3):
        result_lower = result["text"][i].lower()
        assert "the" not in result_lower
        # "QUICK"/"quick"/"Quick" should remain
        assert "quick" in result_lower


def test_detect_language_short_text(local_session):
    """Test language detection with very short text."""
    data = {
        "text": [
            "Hello",  # Very short English
            "Hola",  # Very short Spanish
            "The",  # Single word
            "x",  # Single character
        ]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.detect_language(col("text")).alias("language")
    ).to_polars()

    # Detection may not be reliable for very short text
    # Just verify it doesn't crash and returns string or null
    for i in range(4):
        assert result["language"][i] is None or isinstance(result["language"][i], str)


def test_remove_stopwords_markdown_column(local_session):
    """Test stopword removal on markdown columns."""
    from fenic import markdown

    markdown_text = """# This is a heading

This is a paragraph with the many stopwords that should be removed.

## Another section

More text with stopwords."""

    data = {"markdown": [markdown_text]}
    df = local_session.create_dataframe(data)

    # Convert to string and remove stopwords
    result = df.select(
        text.remove_stopwords(col("markdown")).alias("cleaned")
    ).to_polars()

    cleaned = result["cleaned"][0]

    # Verify stopwords are removed
    assert "is" not in cleaned.lower() or cleaned.lower().count("is") < markdown_text.lower().count("is")
    assert "paragraph" in cleaned
    assert "stopwords" in cleaned
    assert "removed" in cleaned
