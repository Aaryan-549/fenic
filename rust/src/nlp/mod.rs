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

/// Remove stopwords from text based on language
#[polars_expr(output_type=String)]
fn remove_stopwords(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let language_series = inputs[1].str()?;
    let len = text_series.len();

    // Check if language is a literal (broadcast)
    let language_is_literal = language_series.len() == 1;

    // Get the stopword set for the language
    let mut stopword_sets: Vec<Option<&HashSet<&str>>> = Vec::with_capacity(len);

    if language_is_literal {
        // Single language for all rows
        let lang_opt = language_series.get(0);
        let stopword_set = if let Some(lang) = lang_opt {
            STOPWORDS.get(lang)
        } else {
            None
        };

        // Validate language if provided
        if lang_opt.is_some() && stopword_set.is_none() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Unsupported language code '{}'. Supported languages: {}",
                    lang_opt.unwrap(),
                    STOPWORDS.keys().cloned().collect::<Vec<_>>().join(", ")
                ).into(),
            ));
        }

        stopword_sets.resize(len, stopword_set);
    } else {
        // Different language per row
        for i in 0..len {
            let lang_opt = language_series.get(i);
            let stopword_set = if let Some(lang) = lang_opt {
                let set = STOPWORDS.get(lang);
                if set.is_none() {
                    return Err(PolarsError::ComputeError(
                        format!(
                            "Unsupported language code '{}'. Supported languages: {}",
                            lang,
                            STOPWORDS.keys().cloned().collect::<Vec<_>>().join(", ")
                        ).into(),
                    ));
                }
                set
            } else {
                None
            };
            stopword_sets.push(stopword_set);
        }
    }

    let mut result_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);
        let stopword_set = stopword_sets[i];

        let value = match (text_opt, stopword_set) {
            (Some(text), Some(stopwords)) => {
                // Split text into words, filter out stopwords, and rejoin
                let filtered: Vec<&str> = text
                    .split_whitespace()
                    .filter(|word| {
                        // Convert to lowercase for case-insensitive matching
                        let word_lower = word.to_lowercase();
                        !stopwords.contains(word_lower.as_str())
                    })
                    .collect();

                Some(filtered.join(" "))
            }
            (Some(text), None) => Some(text.to_string()), // No stopwords, return original
            _ => None, // If text is null, return null
        };

        result_vec.push(value);
    }

    Ok(StringChunked::from_iter_options(PlSmallStr::EMPTY, result_vec.into_iter()).into_series())
}

/// Remove custom stopwords from text
#[polars_expr(output_type=String)]
fn remove_custom_stopwords(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let stopwords_series = inputs[1].list()?;

    let len = text_series.len();
    let stopwords_is_literal = stopwords_series.len() == 1;

    let mut result_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);
        let stopwords_idx = if stopwords_is_literal { 0 } else { i };
        let stopwords_list_opt = stopwords_series.get(stopwords_idx);

        let value = match (text_opt, stopwords_list_opt) {
            (Some(text), Some(stopwords_series_item)) => {
                // Convert Series to HashSet
                let stopwords_str = stopwords_series_item.str()?;
                let mut stopwords_set = HashSet::new();

                for j in 0..stopwords_str.len() {
                    if let Some(word) = stopwords_str.get(j) {
                        stopwords_set.insert(word.to_lowercase());
                    }
                }

                // Filter words
                let filtered: Vec<&str> = text
                    .split_whitespace()
                    .filter(|word| {
                        let word_lower = word.to_lowercase();
                        !stopwords_set.contains(&word_lower)
                    })
                    .collect();

                Some(filtered.join(" "))
            }
            (Some(text), None) => Some(text.to_string()), // No stopwords, return original
            _ => None, // If text is null, return null
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
        language_vec.into_iter()
    ).into_series();

    let confidence_series = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("confidence"),
        confidence_vec.into_iter()
    ).into_series();

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
