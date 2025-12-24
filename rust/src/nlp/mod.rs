use polars::datatypes::DataType;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::collections::HashSet;

mod stopwords;
use stopwords::STOPWORDS;

#[pyfunction]
pub fn py_validate_language_code(language: &str) -> PyResult<()> {
    if STOPWORDS.contains_key(language) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "Unsupported language code '{}'. Supported languages: {}",
            language,
            STOPWORDS.keys().cloned().collect::<Vec<_>>().join(", ")
        )))
    }
}

/// Remove stopwords from text column
///
/// This unified function removes common stopwords from text while preserving semantic content.
/// It supports both language-specific stopwords and custom stopword lists.
/// Multiple consecutive removed stopwords are collapsed into a single space.
#[polars_expr(output_type=String)]
fn remove_stopwords(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let language_series = inputs[1].str()?;
    let custom_stopwords = if inputs.len() > 2 {
        Some(&inputs[2])
    } else {
        None
    };

    let len = text_series.len();
    let language_is_literal = language_series.len() == 1;

    // Get stopwords - either custom or language-specific
    let custom_stopwords_owned: Option<Vec<String>> = if let Some(custom_series) = custom_stopwords
    {
        // Use custom stopwords - extract from list column
        let custom_list = custom_series.list()?;
        let stopwords_is_literal = custom_list.len() == 1;
        let idx = if stopwords_is_literal { 0 } else { 0 }; // Use first element for literal

        if let Some(first) = custom_list.get_as_series(idx) {
            let str_ca = first.str()?;
            Some(
                str_ca
                    .into_iter()
                    .filter_map(|opt_str| opt_str.map(|s| s.to_string()))
                    .collect(),
            )
        } else {
            None
        }
    } else {
        None
    };

    let mut result_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);
        let lang_idx = if language_is_literal { 0 } else { i };
        let lang_opt = language_series.get(lang_idx);

        let value = match text_opt {
            Some(text) => {
                // Build stopwords set for this row
                let stopwords_set: HashSet<String> =
                    if let Some(ref custom) = custom_stopwords_owned {
                        custom.iter().map(|s| s.to_lowercase()).collect()
                    } else if let Some(lang) = lang_opt {
                        if let Some(lang_stopwords) = STOPWORDS.get(lang) {
                            lang_stopwords.iter().map(|s| s.to_string()).collect()
                        } else {
                            return Err(PolarsError::ComputeError(
                                format!(
                                    "Unsupported language code '{}'. Supported languages: {}",
                                    lang,
                                    STOPWORDS.keys().cloned().collect::<Vec<_>>().join(", ")
                                )
                                .into(),
                            ));
                        }
                    } else {
                        HashSet::new()
                    };

                let words: Vec<&str> = text
                    .split_whitespace()
                    .filter(|word| {
                        let lowercase = word.to_lowercase();
                        !stopwords_set.contains(&lowercase)
                    })
                    .collect();

                // Join with single space (collapsing multiple removed stopwords into one space)
                Some(words.join(" "))
            }
            None => None,
        };

        result_vec.push(value);
    }

    Ok(StringChunked::from_iter_options(PlSmallStr::EMPTY, result_vec.into_iter()).into_series())
}

/// Detect language of text using whatlang
#[polars_expr(output_type=String)]
fn detect_language(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let len = text_series.len();
    let mut result_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);

        let value = match text_opt {
            Some(text) => {
                if text.is_empty() {
                    None // Empty string -> null
                } else {
                    // Detect language
                    match whatlang::detect(text) {
                        Some(info) => {
                            // Convert Lang to ISO 639-1 code
                            Some(info.lang().code().to_string())
                        }
                        None => None, // Could not detect
                    }
                }
            }
            None => None, // Null input -> null output
        };

        result_vec.push(value);
    }

    Ok(StringChunked::from_iter_options(PlSmallStr::EMPTY, result_vec.into_iter()).into_series())
}

/// Detect language with confidence scores
#[polars_expr(output_type_func=detect_language_with_confidence_output_type)]
fn detect_language_with_confidence(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let len = text_series.len();

    let mut language_vec = Vec::with_capacity(len);
    let mut confidence_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);

        match text_opt {
            Some(text) => {
                if text.is_empty() {
                    language_vec.push(None);
                    confidence_vec.push(None);
                } else {
                    match whatlang::detect(text) {
                        Some(info) => {
                            language_vec.push(Some(info.lang().code().to_string()));
                            confidence_vec.push(Some(info.confidence() as f64));
                        }
                        None => {
                            language_vec.push(None);
                            confidence_vec.push(None);
                        }
                    }
                }
            }
            None => {
                language_vec.push(None);
                confidence_vec.push(None);
            }
        }
    }

    // Create struct with language and confidence fields
    let language_series = StringChunked::from_iter_options(
        PlSmallStr::from_static("language"),
        language_vec.into_iter(),
    )
    .into_series();

    let confidence_series = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("confidence"),
        confidence_vec.into_iter(),
    )
    .into_series();

    let struct_chunked = StructChunked::from_series(
        PlSmallStr::EMPTY,
        len,
        [language_series, confidence_series].iter(),
    )?;

    Ok(struct_chunked.into_series())
}

fn detect_language_with_confidence_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(
        field.name().clone(),
        DataType::Struct(vec![
            Field::new(PlSmallStr::from_static("language"), DataType::String),
            Field::new(PlSmallStr::from_static("confidence"), DataType::Float64),
        ]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopwords_english_available() {
        assert!(STOPWORDS.contains_key("en"));
        let en_stopwords = STOPWORDS.get("en").unwrap();
        assert!(en_stopwords.contains("the"));
        assert!(en_stopwords.contains("is"));
        assert!(en_stopwords.contains("a"));
    }

    #[test]
    fn test_stopwords_multiple_languages() {
        // Test that we have stopwords for key languages
        assert!(STOPWORDS.contains_key("en")); // English
        assert!(STOPWORDS.contains_key("es")); // Spanish
        assert!(STOPWORDS.contains_key("fr")); // French
        assert!(STOPWORDS.contains_key("de")); // German
    }

    #[test]
    fn test_validate_language_code_valid() {
        let result = py_validate_language_code("en");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_language_code_invalid() {
        let result = py_validate_language_code("invalid");
        assert!(result.is_err());
    }

    // Note: Full integration tests for polars expressions are in Python tests
    // because they require the full Polars runtime
}
