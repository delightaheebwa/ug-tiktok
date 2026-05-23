from .steps import (
    normalize_text_unit,
    identify_language,
    check_fuzzy_match,
    get_generation_pipeline,
    label_from_polarity,
    extract_sentiment_label,
    score_from_label,
    generate_text
)
import emoji
import textblob as tb
from textblob import Word



def run_steps(
    initial_word: str, words_dict_set: dict[str, set[str]], words_dict_list: dict[str, list[str]]
) -> dict:
    """
    Analyze one text unit (word/phrase) and return one row-like result object.
    This function is designed to be called inside a for-loop or DataFrame apply.
    """

    normalized_word = normalize_text_unit(initial_word)

    matches = identify_language(word=normalized_word, words=words_dict_set)

    result = {
        "original_text": initial_word,
        "normalized_word_text": normalized_word,
        "dict_word": "",
        "dict_lang": matches,
        "fuzzy_word": "",
        "fuzzy_lang": "",
        "fuzzy_confidence": 0.0,
        "checked_text": "",
        "checked_lang": "",
        "corrected_text": "",
        "lemmatized_text": "",
        "polarity_score": 0.0,
        "polarity_label": "",
        "status": "ok",
        "error_message": "",
    }


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
            result["fuzzy_lang"] = best_fuzzy_word[0]
            result["fuzzy_word"] = best_fuzzy_word[1][0]
            result["fuzzy_confidence"] = best_fuzzy_word[1][1]
            result["checked_text"] = best_fuzzy_word[1][0]
            result["checked_lang"] = best_fuzzy_word[0]


        else:
            best_fuzzy_word = "None"
            result["fuzzy_lang"] = best_fuzzy_word
            result["fuzzy_word"] = best_fuzzy_word
            result["checked_text"] = best_fuzzy_word
            result["checked_lang"] = best_fuzzy_word
            
        
    else:
        # dictionary word was found
        result["dict_word"] = matches[0]
        result["checked_text"] = matches[0]
        result["checked_lang"] = matches[0]
    #    return matches



    # Early return for empty strings to keep loop processing robust.
    if not normalized_word:
        result["status"] = "skipped_empty"
        return result

    try:
        lang = result["fuzzy_lang"]

        if lang == "English":
            # English path: demojize -> spell-correct -> lemmatize -> sentiment.
            demojized = emoji.demojize(normalized_word, delimiters=("", ""))
            corrected = str(tb.TextBlob(demojized).correct()).lower()
            lemma = Word(corrected).lemmatize().lower()
            polarity = float(tb.TextBlob(lemma).sentiment.polarity) # type: ignore

            result["corrected_text"] = corrected
            result["lemmatized_text"] = lemma
            result["polarity_score"] = polarity
            result["polarity_label"] = label_from_polarity(polarity)

        elif lang in {"Luganda", "Swahili"}:
            # Luganda/Swahili path: use your local Gemma models for correction and sentiment.
            gen_pipe = get_generation_pipeline(lang_name=lang)

            correction_prompt = (
                f"Correct spelling and clean this {lang} text. "
                f"Return only the corrected text in lowercase: {normalized_word}"
            )
            corrected = generate_text(gen_pipe, correction_prompt)

            sentiment_prompt = (
                f"Classify the sentiment of this {lang} text. "
                f"Answer with one word only: positive, neutral, or negative. "
                f"Text: {corrected}"
            )
            sentiment_raw = generate_text(gen_pipe, sentiment_prompt)
            sentiment_label = extract_sentiment_label(sentiment_raw)
            sentiment_score = score_from_label(sentiment_label)

            result["corrected_text"] = corrected
            result["lemmatized_text"] = corrected  # Placeholder until a language-specific lemmatizer is added.
            result["polarity_label"] = sentiment_label
            result["polarity_score"] = sentiment_score

        else:
            # Unknown/other language: keep normalized_word text and neutral score.
            result["status"] = "unknown_language"

    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = str(exc)

    return result