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

- nump 2.x was messing with a fasttext-wheel version. numpy==2.4.4
- fasttext-wheel==0.9.2
- Dependency issues! Always first find a way of installing the latest version that doesn't conflict with all other current package versions.

- fastText’s language ID models are optimized for sentences and longer text, so they often misclassify or fail on very short words like “Asante”. This is a known limitation of n‑gram–based detectors and most simple LID models.

## Session Learnings — Imports, Paths & Environments

- Problem: `ModuleNotFoundError: No module named 'steps'` when importing `src.run` from a notebook inside `notebooks/`.
	- Lesson: Python resolves imports relative to the process working directory (where the notebook runs), not the file location. Adding `sys.path.append('..')` from the notebook makes the repository root visible so `import src.run` works.

- Problem: Using `from steps import ...` inside files in `src/` fails when importing `src.run` from outside `src`.
	- Lesson: Use package-qualified imports (`from src.steps import ...`) or relative imports (`from .steps import ...`) when inside a package. Relative imports require the module to be accessed as part of a package (not as a top-level script).

- Problem: `ModuleNotFoundError: No module named 'src'` after switching to relative imports in the notebook.
	- Lesson: Don't use relative imports (like `from ..src.run`) inside notebooks; instead, add the parent folder to `sys.path` and use absolute package imports (`from src.run import ...`).

- Problem: `ModuleNotFoundError: No module named 'rapidfuzz'` even though the package was installed in the system/terminal environment.
	- Lesson: Notebook kernels can use a different Python environment than the terminal. Verify with `print(sys.executable)`. To install packages into the running kernel, use the kernel's Python executable: `!{sys.executable} -m pip install rapidfuzz` or switch the notebook's kernel in VS Code to the environment where the package is installed.

- Problem: Thinking `__init__.py` would fix import failures.
	- Lesson: `__init__.py` marks a folder as a package but does not change where Python looks for packages. If Python's working directory doesn't include the parent folder, adding `__init__.py` alone won't help.

- Practical tips and good practices:
	- Prefer creating and activating a virtual environment for the project and selecting that interpreter as the notebook kernel in VS Code.
	- Add a `requirements.txt` or `pyproject.toml` to capture dependencies (e.g., `rapidfuzz`, `pandas`, `numpy`, `fasttext-wheel`).
	- When installing inside a notebook, use `!{sys.executable} -m pip install <pkg>` to ensure installation in the active kernel.
	- Keep imports inside package modules as package-relative (`from src.module import ...`) and import from notebooks using the package root (add repo root to `sys.path` if needed).
	- Use `print(sys.path)` and `print(sys.executable)` for quick debugging of import/environment issues.

## Quick Actions Taken This Session

- Updated `src/run.py` imports to use `from src.steps import ...` so module resolution is explicit when importing from project root.
- Adjusted notebook import strategy: added `sys.path.append('..')` in the notebook to expose repository root to the kernel.
- Identified missing dependency (`rapidfuzz`) as an environment/kernel mismatch rather than a code bug.

---
_Session date: 2026-05-20_

## Chat Session Notes — Normalization Fix

- **Problem:** matched dictionary entries contained trailing newline characters, producing outputs like `('rafiki', 'rafiki\n', ...)` when performing comparisons.
- **Fix applied:** updated `load_words_to_set` to use `line.strip()` when reading files so words no longer include `\n` or surrounding whitespace. See [src/steps.py](src/steps.py#L1-L40).
- **Lesson:** prefer `str.strip()` for trimming leading/trailing whitespace (including newlines) instead of regex when the goal is simple trimming; keep both inputs and word lists normalized (collapse internal whitespace, `strip()`, and `lower()`) so comparisons are consistent.

_Added during interactive debugging session: normalized input vs. loaded-word mismatch._

## Chat Session Addendum — Refactor & Performance Learnings (2026-05-21)

- Cache heavy resources once: build `words_dict` (sets) and a parallel `words_dict_list` (lists) outside per-word processing so file I/O and costly conversions run exactly once.
- Avoid repeated `list()` casts on `set` values inside tight loops — sets are unordered and repeated casting is both slow and unsafe for index-based lookups.
- Use `rapidfuzz.process.extractOne(query, choices, ...)` for single-query fuzzy matching instead of `cdist([query], choices, ...)` to avoid allocating a 2D matrix and unnecessary NumPy work.
- Keep fast membership checks on `set` for exact-match lookups and use lists only for fuzzy-search APIs that require ordered sequences.
- When collecting per-language fuzzy candidates, append only non-`None` results and then pick the best match with `max(..., key=lambda x: x[1][1])` where `x[1][1]` is the numeric similarity score returned by `extractOne`.
- Never `return` inside a loop that must evaluate multiple sources; gather candidates first, then decide and return after the loop.
- Validate third-party API return shapes when refactoring (e.g., `extractOne` expects a string query and returns `(match, score, index)`), and update indexing accordingly.
- Notebook imports: either add the project root to `sys.path` in the notebook or keep package-qualified imports (`from src.module import ...`) so kernels resolve modules consistently.

These edits fixed an I/O & CPU bottleneck and corrected several logic errors so the per-word pipeline runs efficiently on large comment datasets.
These edits fixed an I/O & CPU bottleneck and corrected several logic errors so the per-word pipeline runs efficiently on large comment datasets.

## Chat Session — 2026-05-23 Performance & Parallelization Learnings

- Extract unique words upfront using `set(...)` to avoid re-processing duplicates in large comment corpora; process each distinct token once.
- Parallelize CPU-bound per-word processing with `concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())` to utilise all logical cores. Use `ThreadPoolExecutor` only for I/O-bound work or when functions are not picklable.
- Use `functools.partial` to pre-fill static kwargs (e.g., `words_dict_set`, `words_dict_list`) so the mapped function accepts a single argument for `executor.map()`.
- Build a `word_results_cache = dict(zip(unique_words, results_list))` after parallel processing, then reconstruct final outputs by O(1) lookups for every original token.
- Fix: when computing percentages, divide by the total number of processed tokens (e.g., `len(percent_list)`), not `len(result)` which is incorrect.
- Notebook caveats: `ProcessPoolExecutor` in Windows or in-notebook contexts may require guarding with `if __name__ == '__main__'` or running the workload as a script to avoid child-process import issues. If running inside a notebook, consider running the heavy job as a separate script or use thread-based pools as a fallback.
- Picklability and large objects: ensure `run_steps` and passed structures are picklable. Avoid passing heavy non-picklable objects (open handles, model instances) into worker calls; instead, load heavy models inside worker processes using an `initializer` or lazily on first call.

_Added during interactive chat session: cache + parallelization pattern, Windows/notebook multiprocessing caveats, and the minor percentage-bug fix._

## Chat Session — 2026-05-23: DataFrame creation & column mapping

- Replaced slow row-by-row DataFrame appends with a single-call construction from a list of dicts: `pd.DataFrame(data=results_list, columns=[...])` — much faster and avoids repeated reallocations.
- Use `columns` to select only the dictionary keys you want (for example: `['checked_text','checked_lang','polarity_score','polarity_label']`) so extra keys are ignored.
- Rename columns using the actual dictionary keys: `words.rename(columns={"checked_text": "Word", "checked_lang": "Language", ...}, inplace=True)` — use the `columns` param (not `index`) and ensure keys match `run_steps` output.
- Compute term frequency with `words.groupby([...]).size().reset_index(name='Term Frequency')` after renaming.
- Performance note: building the DataFrame from list-of-dicts is O(n) and far superior to repeated `words.loc[len(words)] = ...` in a Python loop.
- Small correctness check: confirm the exact key name in `results_list` (e.g., `corrected_text` vs `checked_text`) before passing keys into `pd.DataFrame(columns=...)` or into the `rename` mapping.

_Session date: 2026-05-23_

## Chat Session — 2026-05-26: Two-Stage CPU/GPU Pipeline Learnings

- Separate the workflow by cost: keep dictionary lookup, fuzzy matching, and English processing in the CPU stage, and defer Luganda/Swahili Gemma inference to the notebook GPU stage.
- Do not try to share a live GPU model across `ProcessPoolExecutor` workers; each worker would need its own copy and that can trigger OOM or pickling problems.
- Store full result dictionaries for pending GPU work, not just text strings, so the notebook can update the original `results_list` entries in place after inference.
- Use `checked_lang` and `status == "pending_gpu"` to filter the GPU queue in the notebook; `checked_lang` is the key that tells you whether a pending item belongs to Luganda or Swahili.
- Never use `append()` inside a list comprehension for building a list; `append()` returns `None`, so the comprehension would produce a list of `None` values.
- Build batched prompts with a list comprehension, one prompt per word, and pass the prompt list directly to the Hugging Face pipeline instead of stuffing all words into one giant prompt string.
- When the pipeline returns batched outputs, pair them back to the original dictionaries with `zip(pending_words, outputs)` and mutate each dictionary’s `polarity_label`, `polarity_score`, and `status`.
- For a smoke test, process only a tiny sample from each pending language first so you can verify the output shape before running the full batch.
- Be careful to loop over the filtered pending list when generating prompts; looping over the full `results_list` defeats the batching/filtering logic.

_Added during the chat session about moving to a two-step CPU/GPU architecture and fixing the notebook batching logic._

## Chat Session — 2026-05-30: Ordered-dedupe, CPU parallelism & GPU-aware batching

- Use ordered deduplication (`unique_words = list(dict.fromkeys(all_words))`) to avoid reprocessing duplicate tokens while preserving the original token ordering for easy mapping back into results.
- Parallelise CPU-bound preprocessing with `concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())` and a tuned `chunksize` to reduce IPC overhead and improve throughput for large token sets.
- Detect GPU with `torch.cuda.is_available()` and avoid performing GPU model inference inside worker processes; instead, collect `pending_gpu` items and run batched inference in a single GPU-backed process to prevent repeated model loads and CUDA context contention.
- Set `TOKENIZERS_PARALLELISM='false'` when using multiprocessing plus Hugging Face tokenizers to avoid contention and noisy logs.
- Build a `word_results_lookup = dict(zip(unique_words, results_unique))` and reconstruct results with `[word_results_lookup[w] for w in all_words]` so the CPU stage runs in O(U) (unique tokens) while final output preserves original order and frequency counts.
- Caveat: passing very large lookup structures into `ProcessPoolExecutor` will pickle and transfer them to workers. For large `words_dict_set` consider using a worker `initializer` to load read-only shared data or run the preprocessing as a separate script to avoid excessive pickling.
- Tuning note: the `chunksize` heuristic (`len(unique_words)//(max_cores*4)`) is a good starting point but may require adjustment depending on dataset size and the cost of `run_steps`.

_Session date: 2026-05-30_
