from .steps import (
    normalize_text_unit,
    identify_language,
    check_fuzzy_match,
)
from typing import Literal


def run_steps(
    initial_word: str, words_dict_set: dict[str, set[str]], words_dict_list: dict[str, list[str]]
) -> tuple[str, tuple[str, float, int]] | list[str] | Literal["None"]:
    """Run the steps for sentiment analysis of the word in question"""

    
    normalized_word = normalize_text_unit(initial_word)

    matches = identify_language(word=normalized_word, words=words_dict_set)


    if matches == ["Unknown"]:
        # since dictionary word wasn't found, fuzzy match is applied
        fuzzy_list = []
        for lang_name, lang_set in words_dict_list.items():
            fuzzy_match = check_fuzzy_match(word=normalized_word, lang_set=lang_set)
            if fuzzy_match is not None:
                fuzzy_list.append((lang_name, fuzzy_match))
            
        # check if list is empty
        if fuzzy_list:
            # max from list
            best_fuzzy_word = max(fuzzy_list, key=lambda x: x[1][1])
            return best_fuzzy_word
        else:
            best_fuzzy_word = "None"
            return best_fuzzy_word
        
    else:
        # dictionary word was found
        return matches
