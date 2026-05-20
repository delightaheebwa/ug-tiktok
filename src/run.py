from .steps import (
    normalize_text_unit,
    load_words_to_set,
    identify_language,
    check_fuzzy_match,
)
from nltk.corpus import words as nltk_words


def run_steps(
    initial_word: str,
) -> tuple[str, tuple[str, str, float]] | list[str] | str:
    """Run the steps for sentiment analysis of the word in question"""

    words_dict: dict[str, set[str]] = {
        "English": set(nltk_words.words()),
        "Luganda": load_words_to_set(
            file_path="C:/Users/HP/Desktop/tiktok-analysis/words/processed/luganda_words.txt"
        ),
        "Swahili": load_words_to_set(
            file_path="C:/Users/HP/Desktop/tiktok-analysis/words/processed/swahili_words.txt"
        ),
    }

    normalized_word = normalize_text_unit(initial_word)

    matches = identify_language(word=normalized_word, words=words_dict)

    fuzzy_list = []

    if matches == ["Unknown"]:
        # since dictionary word wasn't found, fuzzy match is applied
        for lang_name, lang_set in words_dict.items():
            fuzzy_match = check_fuzzy_match(word=normalized_word, lang_set=lang_set)

            if fuzzy_match is not None:
                fuzzy_list.append((lang_name, fuzzy_match))

        # check if it is empty
        if fuzzy_list:
            # pick the tuple best fuzzy match  from the list of tuples
            best_fuzzy_match = max(fuzzy_list, key=lambda x: x[1][2])
        else:
            best_fuzzy_match = "None"

        return best_fuzzy_match

    else:
        # dictionary word was found
        return matches
