# P02 · Preprocessing, on Your Corpus

**Meetings:** D25 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 1 Preprocessing, the rest of it, about 15 minutes.

**During the chapter**

**Keep the list.** In `scratch/p02-order.md` in the rag repo, write every operation the chapter applies to text, in the order it applies them. The walkthrough has you write the same pipeline for your corpus, and the chapter's order is the one to copy or argue with.

**Notes**

**The order of operations changes the output, silently.** Stem before removing stopwords and your stopword list stops matching (`"was"` stems to `"wa"`). Split into tokens before stripping punctuation and `"dog."` and `"dog"` become two terms, and every score after that is wrong. Lowercasing and stripping punctuation can go in either order.

**`string.punctuation` is ASCII only.** It does not contain the curly apostrophe `’`, the em dash `—` or curly quotes, and a lot of real text is full of them. If your corpus came from a modern source, `“Ithaca”` with curly quotes survives an ASCII strip as a different term from `ithaca`. The walkthrough strips by Unicode category instead, which catches all of them.

**`-m rag.preprocess`, not `rag/preprocess.py`.** Running a file as a module is what lets `rag/tfidf.py` say `from rag.corpus import ...` on Friday. Every command that runs something in `rag/` uses `-m`; scripts in `scratch/` run by path.

**Walkthrough — the same pipeline on your corpus**

Boot.dev grades its version on its dataset. This runs yours on your corpus, the one you locked in A05b. Everything from here to December runs on that file.

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P01 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/preprocess
```

*If P01 is not approved yet:* `git switch dev/setup && git switch -c dev/preprocess`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 1 finished, every lesson green, scratch/p02-order.md written
- [ ] Step 3: rag.corpus run, passages counted
- [ ] Step 4: tokenize and preprocess typed into rag/preprocess.py, the two lines match
- [ ] Extension: prediction written first, three sentences from my corpus through the pipeline
- [ ] Fill every slot in evidence/P02.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Finish chapter 1.**

In `~/version_control/rag-search-engine/`, lessons 6 to 9, each run locally before it is submitted. Every one of them edits `cli/lib/keyword_search.py`, and by lesson 7 the processing lives in one helper, `tokenize_text`, that the rest of the course reuses. When the chapter page shows every lesson complete, save it as `evidence/p02-bootdev-ch1.png` in the rag repo, and check `scratch/p02-order.md` against what you built: lowercase, strip punctuation, tokenize, drop stopwords, stem.

**Lesson 6: punctuation.** Strip every punctuation mark from both the query and the title before comparing, so `"magic charlie"` finds `It's Magic, Charlie Brown`. The course points at three standard-library pieces: `string.punctuation` is the ASCII marks as one string, `str.maketrans("", "", string.punctuation)` builds a table that maps each of them to nothing, and `.translate(table)` applies it.

1. At the top of `cli/lib/keyword_search.py`:

```python
import string

PUNCT = str.maketrans("", "", string.punctuation)

# In: text.  Out: the same text lowercased with every ASCII punctuation mark deleted.
def normalize(text):
    return text.lower().translate(PUNCT)
```

2. Use it on both sides in `search_titles`:

```python
def search_titles(query):
    query = normalize(query)
    results = []
    for movie in load_movies():
        if query in normalize(movie["title"]):
            results.append(movie)
    return results[:5]
```

3. Run, then submit:

```bash
uv run cli/keyword_search_cli.py search "magic charlie"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `It's Magic, Charlie Brown` in the list. Deleting the marks, not replacing them with spaces, is the course's choice: `sci-fi` becomes `scifi` and `It's` becomes `Its`. P02 Step 4 makes the other choice on your corpus, and Reflection 1 asks you to say what each one does to `well-known`.

**Lesson 7: tokenization.** `"Great Bear"` should match `Big Bear`, so the match has to be token against token, not whole query against whole title. The course says to wrap the processing in one helper, because every later lesson reuses it.

1. In `cli/lib/keyword_search.py`, the helper, with `normalize` folded in:

```python
# In: text.  Out: its tokens: lowercased, punctuation deleted, split on any whitespace.
def tokenize_text(text):
    return text.lower().translate(PUNCT).split()
```

`.split()` with no argument splits on any run of whitespace, so two spaces or a tab do not make an empty token.

2. The match: at least one query token appears in some title token. "Appears in" is still substring, so `fast` matches the title token `faster`:

```python
def search_titles(query):
    query_tokens = tokenize_text(query)
    results = []
    for movie in load_movies():
        title_tokens = tokenize_text(movie["title"])
        if any(q in t for q in query_tokens for t in title_tokens):
            results.append(movie)
    return results[:5]
```

`any(...)` with two `for`s tries every query token against every title token and stops at the first hit.

3. Run, then submit:

```bash
uv run cli/keyword_search_cli.py search "furious fast"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Furious Seven`, `Fast and Furious` and `Faster, Pussycat! Kill! Kill!`. The last one is there because `fast` is inside `faster`; that is substring matching on tokens, which lesson 9 replaces with stems.

**Lesson 8: stop words.** `"the bear"` should not match `The Terminator`. Drop the common words from both sides.

1. Make `data/stopwords.txt` from the list at the bottom of the lesson page: select it, paste it into the new file, one word per line. It is in `data/`, which is gitignored, like the dataset.

2. Load it once, at import time, through the same processing as everything else, so `aren't` in the file becomes `arent` and matches the token `arent`:

```python
# In: nothing.  Out: the set of stop words from data/stopwords.txt, processed the way tokens are.
def load_stopwords():
    with open("data/stopwords.txt", encoding="utf-8") as f:
        return {w.lower().translate(PUNCT) for w in f.read().splitlines() if w.strip()}

STOPWORDS = load_stopwords()
```

A set, not a list: `token in STOPWORDS` is one lookup instead of a scan through 170 words for every token.

3. Filter inside the helper, after splitting:

```python
def tokenize_text(text):
    tokens = text.lower().translate(PUNCT).split()
    return [t for t in tokens if t not in STOPWORDS]
```

`search_titles` does not change; both sides go through `tokenize_text`.

4. Run, then submit:

```bash
uv run cli/keyword_search_cli.py search "the hot shot"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Hot Potato`, `Hotel Chevalier` and `Killshot`, and no title that matched only on `the`. *If it broke* with `FileNotFoundError: data/stopwords.txt`, the file is in the wrong folder or has a different name. *If it broke* with the stop words not filtering, you loaded them raw; `aren't` with its apostrophe never equals the token `arent`.

**Lesson 9: stemming.** `"running"` should find `Run Baby Run`. Writing a stemmer is a project of its own, so the course installs NLTK's Porter stemmer.

1. Add the dependency and import it:

```bash
uv add nltk==3.9.1
```

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
```

2. Stem each token as the last step of the helper, after the stop words are gone:

```python
def tokenize_text(text):
    tokens = text.lower().translate(PUNCT).split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return [stemmer.stem(t) for t in tokens]
```

Order matters here, and it is Reflection 3: stem first and `was` becomes `wa`, which is not in the stop word list, so it survives.

3. Run, then submit:

```bash
uv run cli/keyword_search_cli.py search "running"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Virginia's Run`, `Take the Money and Run` and `Woman on the Run`: `running` stems to `run`, and so does `run`. *If it broke* with `ModuleNotFoundError: No module named 'nltk'`, the `uv add` did not run in the project folder, or you ran the script with `python` instead of `uv run`.

That is the whole pipeline, in the course's order: lowercase, delete punctuation, split, drop stop words, stem. Steps 3 and 4 below build the same five steps on your corpus, with one decision made differently.

**Step 3. The corpus as passages.**

TF-IDF scores terms per document, and your corpus is one file, so for the next few chapters a "document" is a passage: the blank-line blocks of your file, merged forward until each one is at least `MIN_CHARS` long. Open `rag/corpus.py`, read it, and set `MIN_CHARS` to the minimum you settled on in A06. Then:

```bash
uv run python -m rag.corpus
```

*You should see* a passage count in the hundreds or low thousands, a `shortest` that is at least `MIN_CHARS`, and first and last lines that match the first and last words of `data/corpus.txt`. On Homer with `MIN_CHARS = 200`: `2118 passages`, `shortest 200 median 596 longest 3415`, a first passage that begins `[Illustration] The Odyssey by Homer` and a last one that ends `the funeral of Hector tamer of horses.` The leftover at the end is merged backward into the last passage rather than dropped, which is why nothing is shorter than the minimum. Paste the four lines under `## Step 3` in `evidence/P02.md`.

**Step 4. `rag/preprocess.py`.**

NLTK is already installed by `setup.sh`; it is here only for its Porter stemmer, which is the one the file uses whatever the chapter used, so that the test lines below come out the same for everyone. If the chapter's stemmer differs, say so in `evidence/P02.md` and keep Porter here. Open `rag/preprocess.py`. The top half is shipped: the docstring that states the order of operations, the stopword list, and `strip_punctuation`, which turns every punctuation mark and symbol into a space by its Unicode category rather than an ASCII list. Read those lines. The two functions that are the pipeline you type yourself, under the `# ---- P02 Step 4` line, because the pipeline is the lesson:

```python
# In: text.  Out: lowercased words of two or more characters, punctuation gone.
def tokenize(text):
    return [t for t in strip_punctuation(text.lower()).split() if len(t) > 1]

# In: text, a stopword set, whether to stem, a set of stems to drop afterwards.  Out: the list of search terms.
def preprocess(text, stopwords=STOPWORDS, stem=True, drop_stems=frozenset()):
    tokens = [t for t in tokenize(text) if t not in stopwords]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]       # "ships" -> "ship", "sailing" -> "sail"
    return [t for t in tokens if t not in drop_stems]     # drop_stems is empty until the P04 extension fills it; it runs after stemming on purpose

if __name__ == "__main__":
    print(tokenize("Don’t stop—well-known Ulysses's ships."))
    print(preprocess("The ships were sailing to Ithaca, and the sailors rowed."))
```

The first test line has a curly apostrophe and an em dash on purpose. On a Mac, `’` is option-shift-`]` and `—` is option-shift-hyphen; on Windows, copy that one line from here. A straight `'` and a `-` print the same two lists, and then nothing about this file has tested the Unicode strip.

Punctuation becomes a space rather than vanishing, so `well-known` is two terms and the em dash does not glue two words together. The possessive `'s` leaves a stray `s`, which the one-letter rule drops. That is a decision, and the docstring states it. If the chapter's order differs from the docstring's, change the docstring to the chapter's order or keep yours, and say which in `evidence/P02.md`. The deliverable is the docstring as much as the code.

```bash
uv run python -m rag.preprocess
```

*You should see* exactly this, because it does not depend on your corpus:

```text
['don', 'stop', 'well', 'known', 'ulysses', 'ships']
['ship', 'sail', 'ithaca', 'sailor', 'row']
```

Paste both lines under `## Step 4`. *If it broke:* `don’t` surviving with its curly apostrophe means your strip is ASCII only. `ulyssess` means you deleted punctuation instead of turning it into a space.

**Extension — three sentences from your corpus (ASSIGNED)**

Pick three sentences from `data/corpus.txt`, select-and-copy, never retyped: one with a name, one with a number, and one with punctuation the chapter's examples did not have. Put them in `scratch/p02-sentences.txt` with a blank line between them; the corpus file wraps its lines, so a copied sentence arrives on two or three lines, and that is fine, the blank line is what separates one sentence from the next. Before you run anything, write under `## Extension` in `evidence/P02.md` which of the three you expect the pipeline to damage most, and how. Commit that file. I will check your commit timestamps. Then:

```bash
uv run python scratch/p02-sentences.py
```

*You should see* each sentence followed by its tokens and its terms. On Homer, `Euryclea, daughter of Ops, the son of Pisenor` comes out as `['euryclea', 'daughter', 'op', 'son', 'pisenor', ...]`: the stemmer took the `s` off `Ops`, so a search for `ops` finds it and a search for the name as written does not. Paste the three pairs under your prediction and name the term that came out wrong or lost meaning, and which operation did it.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P02.md` (the questions are below) and run `uv run python scripts/check_evidence.py P02` until it says `all slots filled`. Then:

```bash
git add rag/ scratch/ evidence/
git commit -m "P02: corpus passages and the preprocessing pipeline"
git push -u origin dev/preprocess
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the one operation in your docstring you would argue with.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P02.md` with every slot filled + `rag/corpus.py` with your `MIN_CHARS` + `rag/preprocess.py` (the docstring states your order) + `scratch/p02-order.md` + `scratch/p02-sentences.txt` + `evidence/p02-bootdev-ch1.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P02.md`)

1. Paste your `tokenize` line. Say what happened to `Don’t`, to `well-known` and to `Ulysses's`, and which line of `strip_punctuation` or `tokenize` did each. Then say what the course's lesson 6, which deletes the marks instead of replacing them, would have made of the same three.

   *How to get it:* the curly apostrophe is punctuation by Unicode category, so `strip_punctuation` turned it into a space and `t` was dropped by `len(t) > 1`; the hyphen became a space, so two tokens; the possessive left an `s` that the same length rule dropped. Lesson 6's `translate` deletes: `dont`, `wellknown`, `ulyssess`, and it would leave the curly apostrophe alone, because `string.punctuation` is ASCII.

2. Your `shortest` passage is exactly `MIN_CHARS` or longer. Say why the loop in `load_passages` cannot produce a shorter one, and what happens to the text left over at the end of the file.

   *How to get it:* read the `if len(buf) >= min_chars` line and the `if buf and passages:` / `elif buf:` lines after the loop in `rag/corpus.py`.

3. The chapter removes stop words (lesson 8) before it stems (lesson 9). Say what goes wrong if you stem first, with one stopword from the list that would stop matching.

   *How to get it:* `uv run python -c "from nltk.stem import PorterStemmer; print(PorterStemmer().stem('was'), PorterStemmer().stem('this'))"` prints what those two become.
