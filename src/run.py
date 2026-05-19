from steps import normalize_text_unit, load_words_to_set, identify_language, check_fuzzy_match
from nltk.corpus import words as nltk_words

def run_steps(initial_word: str):
    """Run the steps for sentiment analysis of the concarenated comments string"""

    words_dict: dict[str, set[str]] = {
        "English": set(nltk_words.words()),
        "Luganda": load_words_to_set(file_path="words/processed/luganda_words.txt"),
        "Swahili": load_words_to_set(file_path="words/processed/swahili_words.txt"),
    }

    english_set = words_dict["English"]
    luganda_set = words_dict["Luganda"]
    swahili_set = words_dict["Swahili"]

    combined_set_list = [english_set, luganda_set, swahili_set]

    # maybe go back to repeated calling and atorin wgatever and checkin ifg none

    normalized_comments = normalize_text_unit(initial_word)

    matches = identify_language(word=initial_word, words=words_dict)

    if matches == ["Unknown"]:
        fuzzy_list = []
        for lang_name, lang_set in words_dict.items():
            fuzzy_match = check_fuzzy_match(word=initial_word, lang_set=lang_set)

            if fuzzy_match is not None:
                fuzzy_list.append((lang_name, fuzzy_match))
            pass
        
        # check if it is empty
        if fuzzy_list:
            # pick the tuple best fuzzy match  from the list of tuples
            best_fuzzy_match = max(fuzzy_list, key=lambda x: x[1][2])
        else:
            best_fuzzy_match = None
