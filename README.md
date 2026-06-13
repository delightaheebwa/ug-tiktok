# Ugandan TikTok Sentiment Analysis

**Multilingual sentiment analysis of TikTok comments by Ugandans.**

A two-stage, CPU + GPU pipeline that processes TikTok comments written in a mix of **English**, **Luganda**, and **Swahili** — overcoming the limitations of English-only sentiment tools by combining dictionary-based language identification, fuzzy matching, and fine-tuned Gemma 1B LLMs for African language inference.

---

## The Challenge

Ugandan TikTok comments are naturally multilingual, often mixing English, Luganda, and Swahili within a single comment. Standard sentiment analysis tools (e.g., TextBlob, VADER) only handle English, while language detection models like FastText fail on short, single-word tokens common in comment data.

This project addresses both problems with a custom two-stage architecture.

---

## Architecture

### Stage 1: CPU-bound Preprocessing

```
Raw Comments
    |
    Tokenization (split into words)
    |
    Dictionary Lookup          Exact match against Luganda, Swahili, English word lists
    |
    Fuzzy Match (rapidfuzz)    Fallback for unknown words (80% similarity cutoff)
    |
    English path               Lemmatize (TextBlob) + polarity scoring
    |                          Output: score + label (positive/neutral/negative)
    |
    Luganda/Swahili path       Mark as `pending_gpu` for Stage 2
```

### Stage 2: GPU-bound LLM Inference

```
pending_gpu words (Luganda / Swahili)
    |
    Load fine-tuned Gemma 1B models (4-bit quantized)
    |   - ganda-gemma-1b for Luganda
    |   - swahili-gemma-1b for Swahili
    |
    Batched prompt-based classification
    |
    Output: sentiment label (positive/neutral/negative) per word
```

---

## Repository Structure

```
ug-tiktok/
  src/
    run.py             Per-word processing pipeline (CPU stage)
    steps.py           Helper functions (normalization, language ID, fuzzy match, sentiment)
  notebooks/
    run.ipynb          Main pipeline notebook (CPU + GPU execution)
    experiment.ipynb   Exploratory notebook (FastText-based prototype)
    english_words.csv          Processed English word results
    pending_luganda_words.csv  Luganda words awaiting GPU inference
    pending_swahili_words.csv  Swahili words awaiting GPU inference
  words/
    processed/
      luganda_words.txt        Cleaned Luganda dictionary
      swahili_words.txt        Cleaned Swahili dictionary
    raw/
      English-Luganda_version_2_0.pdf   Source English-Luganda dictionary
      swa_eng_dict_text.pdf             Source Swahili-English dictionary
  img/                   Result screenshots
  learnings.md           Development log with decisions and debugging notes
  dataset_tiktok-comments-scraper_2026-04-22_19-27-52-195.xlsx    Raw comments (11,871 rows)
  requirements.txt       Python dependencies
  .gitignore
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) for Gemma 1B inference
- 8GB+ GPU memory (for 4-bit quantized models)
- Hugging Face access to Gemma models

### Setup

```bash
# Clone the repository
git clone https://github.com/delightaheebwa/ug-tiktok.git
cd ug-tiktok

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# On Windows, use fasttext-wheel instead of fasttext:
#   pip install fasttext-wheel==0.9.2
```

### Download the Gemma Models

The fine-tuned Gemma 1B models must be downloaded separately:

- [ganda-gemma-1b](https://huggingface.co/) (Luganda)
- [swahili-gemma-1b](https://huggingface.co/) (Swahili)

Update the model paths in `notebooks/run.ipynb` to point to your local copies.

---

## Usage

### Running the pipeline

```bash
jupyter notebook notebooks/run.ipynb
```

The notebook executes the full pipeline:
1. Loads and cleans TikTok comments from the XLSX dataset
2. Tokenizes all comments into individual words
3. Runs the CPU-bound preprocessing (`run_steps()` per word, parallelized via ProcessPoolExecutor)
4. Loads Gemma models and runs GPU inference for Luganda/Swahili words
5. Outputs sentiment results per word with language labels

### Using the processing pipeline programmatically

```python
from src.run import run_steps
from src.steps import load_words_to_set

# Load language dictionaries
words_dict = {
    "Luganda": load_words_to_set("words/processed/luganda_words.txt"),
    "Swahili": load_words_to_set("words/processed/swahili_words.txt"),
    "English": load_words_to_set("words/processed/english_words.txt"),
}
words_dict_list = {k: list(v) for k, v in words_dict.items()}

# Process a single word
result = run_steps("wasaze", words_dict_set=words_dict, words_dict_list=words_dict_list)

print(f"Word: {result['original_text']}")
print(f"Language: {result['checked_lang']}")
print(f"Polarity: {result['polarity_label']} ({result['polarity_score']})")
```

### Running GPU inference for Luganda/Swahili

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "path/to/ganda-gemma-1b",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("path/to/ganda-gemma-1b")

# Classify sentiment
prompt = f"Classify the sentiment of this Luganda word as positive, neutral, or negative: 'wasaze'"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=0.1)
label = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `textblob` | English lemmatization and sentiment |
| `rapidfuzz` | Fuzzy string matching for language ID |
| `transformers`, `torch` | Hugging Face LLM inference |
| `bitsandbytes` | 4-bit quantization for GPU memory efficiency |
| `fasttext-wheel` | Language detection (experimental only) |
| `emoji` | Emoji handling |

---

## Known Issues

- **Fuzzy match bias:** The English dictionary is significantly larger than the Luganda/Swahili dictionaries, causing some non-English words to be misidentified as English (e.g., "Asante" → "basanite", "banange" → "bandage")
- **FastText abandoned:** Word-level language detection was abandoned because FastText is optimized for sentence-level detection and fails on single short tokens
- **Work in progress:** The `run_steps()` function intentionally returns an incomplete dict; the GPU pipeline processes words individually for now
- **Missing `requirements.txt`:** (Added in this release — dependencies are now captured for reproducible environments)
- **Missing `.gitignore`:** (Added in this release — excludes caches, logs, and model weights)

---

## License

MIT
