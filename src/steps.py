import re
from rapidfuzz.process import extractOne
from rapidfuzz import fuzz
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextGenerationPipeline,
)
import torch


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
    # XX: dependent on dictionary words being correct
    matches = [lang for lang, word_set in words.items() if word in word_set]
    return matches if matches else ["Unknown"]


def check_fuzzy_match(word: str, lang_set: list[str]) -> tuple[str, float, int] | None:
    """Find a fuzzy match for a word"""
    # Only track matches that have an 80% similarity or higher
    best_fuzzy_tuple = extractOne(word, lang_set, scorer=fuzz.ratio, score_cutoff=80)

    if best_fuzzy_tuple is not None:
        # (closest matching word, similarity score, position (index) of that match in the list.)
        return best_fuzzy_tuple
    else:
        return None


# Rewritten multilingual analyzer for token/text-unit processing.
# Key design choices:
# 1) No stopword filtering (per your requirement).
# 2) Returns one structured result object per input text unit.
# 3) Loads heavy models lazily and reuses them across loop iterations.


def get_generation_pipeline(
    lang_name: str,
):  # XXX
    """Lazy-load and cache language-specific generation pipelines."""
    # Reuse generation pipelines so they are not rebuilt inside the loop.
    generation_pipelines: dict = {"English": None, "Luganda": None | TextGenerationPipeline, "Swahili": None | TextGenerationPipeline}

    # Extract language name and word
    model_g_path = "C:/Users/HP/ganda-gemma-1b"
    model_s_path = "C:/Users/HP/swahili-gemma-1b"


    # if lang_name not in generation_pipelines:
    #   raise ValueError(f"Unsupported language code for generation: {lang_name}")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # config to share across models
    load_args = {
        "device_map": "auto",
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
    }

    if generation_pipelines[lang_name] is None:
        model_path = model_g_path if lang_name == "Luganda" else model_s_path
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_args)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        generation_pipelines[lang_name] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    return generation_pipelines[lang_name]


def label_from_polarity(score: float, neutral_band: float = 0.1) -> str:
    """Map numeric polarity to a 3-class label."""
    if score > neutral_band:
        return "positive"
    if score < -neutral_band:
        return "negative"
    return "neutral"


def extract_sentiment_label(raw_text: str):
    """Extract one of: positive, neutral, negative from model text output."""
    lowered = raw_text.lower()
    for label in ("positive", "neutral", "negative"):
        if label in lowered:
            return label
    return "neutral"


def score_from_label(label: str) -> float:
    """Map sentiment label to numeric score."""
    # use of a "non-nuanced" scoring since when the Gemma model outputs "positive", we don't know if it's a strong positive or a weak positive.
    return {"positive": 0.5, "neutral": 0.0, "negative": -0.5}.get(label, 0.0)


def generate_text(gen_pipe, prompt: str, max_new_tokens: int = 5):  # XXX
    """Run deterministic generation for consistency in data processing."""
    output = gen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.1,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()
