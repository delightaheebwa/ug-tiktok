# Learnings From The Initial Notebook Draft

- I treated the analysis function like a whole-comment pipeline even though the plan shifted to processing one word or text unit at a time. The fix was to make the function accept a single text unit and return one row-like result object per call.

- I loaded the FastText language model inside the processing flow. The fix was to load it once outside the loop so it can be reused for every row.

- I mixed local state lists such as `clean_tokens`, `polarity_list`, and `unknown_lang_list` into a function that was supposed to build one result per input. The fix was to stop accumulating unused internal lists and instead return a structured result object.

- I used stopword filtering in a way that assumed English-only behavior. The fix was to remove stopword handling entirely because the data is multilingual and a single token can still belong to Luganda or Swahili.

- I called `word_tokenize(text)` even though `text` was already meant to be one token or one text unit. The fix was to skip tokenization in that function and operate directly on the incoming unit.

- I applied `textblob.correct()` and lemmatization as if every intermediate value was a string, but some of the code was operating on lists. The fix was to keep the English branch strictly string-based and apply correction and lemmatization in a clear order.

- I tried to get polarity from a list of blob objects and then appended the result without checking the shape. The fix was to compute a single numeric polarity score and map it to a label consistently.

- I used `pipeline("text-classification")` with `AutoModelForCausalLM` for the Luganda and Swahili branches. The fix was to switch to `text-generation` and use the causal language model through a generation pipeline instead.

- I created Hugging Face pipelines inside the per-row logic. The fix was to lazily cache the generation pipelines so they are created once and reused.

- I assumed the language model branch could also perform sentiment directly without a clear output format. The fix was to prompt the model explicitly for either corrected text or a sentiment label and then parse the result.

- I left the function without a return value, which meant the loop would have produced no usable output. The fix was to return a single structured object with fields like `original_text`, `lang`, `corrected_text`, `lemmatized_text`, `polarity_score`, `polarity_label`, `status`, and `error_message`.

- I did not handle empty or malformed inputs. The fix was to normalize the input first and return an explicit skipped or error status when the text is empty or processing fails.

- On Windows, the notebook environment can be sensitive to FastText installation issues. The fix is to use `fasttext-wheel` in the environment while still importing it with `import fasttext`.
