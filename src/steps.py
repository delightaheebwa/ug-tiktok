import re
from rapidfuzz.process import cdist
from rapidfuzz import fuzz
import numpy as np


def normalize_text_unit(text: str) -> str:
    """Normalize one input text unit while preserving meaning."""
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text




def load_words_to_set(file_path: str) -> set[str]:
    """Load the language words in their respective sets"""
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        return {line.strip() for line in file}


def identify_language(word: str, words: dict[str, set[str]]) -> list[str]:
    """Checks all sets and returns a list of language(s) that contains the word"""
    matches = [lang for lang, word_set in words.items() if word in word_set]
    return matches if matches else ["Unknown"]


def check_fuzzy_match(word: str, lang_set: set[str]) -> tuple[str, str, float] | None:
    """Find a fuzzy match for a word"""
    # Only track matches that have an 80% similarity or higher
    # for perfromance reasons and the fact that Iterable is what is accepted, we convert to list here
    matrix = cdist([word], list(lang_set), scorer=fuzz.ratio, score_cutoff=80)

    # Get the index of the highest score in the first row
    best_idx = np.argmax(matrix[0])

    # Check if the best score actually passes the threshold
    if matrix[0][best_idx] > 0:
        # (input word, matched word, similarity score)
        return (word, list(lang_set)[best_idx], round(matrix[0][best_idx], 2))
    else:
        return None
