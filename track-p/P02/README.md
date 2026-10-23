# P02 · Preprocessing, on Your Corpus

**Meetings:** D25 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 1 Preprocessing, the rest of it, about 15 minutes.

**During the chapter**

**Keep the list.** In `scratch/p02-order.md` in the rag repo, write every operation the chapter applies to text, in the order it applies them. The walkthrough has you write the same pipeline for your corpus, and the chapter's order is the one to copy or argue with.

**Notes**

**The order of operations changes the output, silently.** Stem before removing stopwords and your stopword list stops matching (`"was"` stems to `"wa"`). Split into tokens before stripping punctuation and `"dog."` and `"dog"` become two terms, and every score after that is wrong. Lowercasing and stripping punctuation can go in either order.

**`string.punctuation` is ASCII only.** It does not contain the curly apostrophe `’`, the em dash `—` or curly quotes, and a lot of real text is full of them. If your corpus came from a modern source, `“Ithaca”` with curly quotes survives an ASCII strip as a different term from `ithaca`. The walkthrough strips by Unicode category instead, which catches all of them.

**`-m rag.preprocess`, not `rag/preprocess.py`.** Running a file as a module is what lets `rag/tfidf.py` say `from rag.corpus import ...` on Friday. Every command in this track uses `-m`.

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
- [ ] Step 4: rag/preprocess.py pasted and run, the two lines match
- [ ] Extension: prediction in the log, five sentences from my corpus through the pipeline
- [ ] Fill every slot in evidence/P02.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Finish chapter 1.**

In `~/version_control/bootdev-rag/`, the remaining lessons, each run locally before it is submitted. When the chapter page shows every lesson complete, save it as `evidence/p02-bootdev-ch1.png` in the rag repo, and finish `scratch/p02-order.md`.

**Step 3. The corpus as passages.**

TF-IDF scores terms per document, and your corpus is one file, so for the next few chapters a "document" is a passage: the blank-line blocks of your file, merged forward until each one is at least `MIN_CHARS` long. Open `rag/corpus.py`, read it, and set `MIN_CHARS` to the minimum you settled on in A06. Then:

```bash
uv run python -m rag.corpus
```

*You should see* a passage count in the hundreds or low thousands, a `shortest` that is at least `MIN_CHARS`, and first and last lines that match the first and last words of `data/corpus.txt`. On Homer with `MIN_CHARS = 200`: `2118 passages`, `shortest 200 median 596 longest 3415`, a first passage that begins `[Illustration] The Odyssey by Homer` and a last one that ends `the funeral of Hector tamer of horses.` The leftover at the end is merged backward into the last passage rather than dropped, which is why nothing is shorter than the minimum. Paste the four lines under `## Step 3` in `evidence/P02.md`.

**Step 4. `rag/preprocess.py`.**

```bash
uv add nltk
```

NLTK is here only for its Porter stemmer. If the chapter had you use a different stemmer, use that one instead, so the two implementations agree. This is the file you type rather than run, because the pipeline is the lesson. Create `rag/preprocess.py` and type it:

```python
# rag/preprocess.py  (P02)
"""Turn text into search terms.

Order: lowercase -> punctuation and symbols to spaces -> split on whitespace
-> drop one-letter tokens -> drop stopwords -> stem -> drop corpus stems.
"""
import unicodedata

from nltk.stem import PorterStemmer

STOPWORDS = frozenset("""
a about after all also an and any are as at be been but by can could did do does
for from had has have he her him his how i if in into is it its me my no not of on
or our out she so than that the their them then there these they this those to up
was we were what when where which who why will with would you your
""".split())

_stemmer = PorterStemmer()

# In: text.  Out: the same text with every punctuation mark and symbol replaced by a space.
def strip_punctuation(text):
    # unicodedata.category names what a character is: "P..." is punctuation, "S..." is a symbol. Curly quotes and em dashes count; an ASCII list would miss them.
    return "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in text)

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

**Extension — five sentences from your corpus (ASSIGNED)**

Pick five sentences from `data/corpus.txt`, select-and-copy, never retyped: one with a name, one with a number, one with punctuation the chapter's examples did not have, and two you like. Before you run anything, write in your log which of the five you expect the pipeline to damage most, and how. Commit the log. I will check your commit timestamps. Then put the five in `scratch/p02-sentences.txt`, one per line, and:

```bash
uv run python scratch/p02-sentences.py
```

*You should see* each sentence followed by its tokens and its terms. On Homer, `Euryclea, daughter of Ops, the son of Pisenor` comes out as `['euryclea', 'daughter', 'op', 'son', 'pisenor', ...]`: the stemmer took the `s` off `Ops`, so a search for `ops` finds it and a search for the name as written does not. Paste the five pairs under `## Extension` in `evidence/P02.md` and name the term that came out wrong or lost meaning, and which operation did it.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P02.md` (the questions are below) and run `uv run python scripts/check_evidence.py P02` until it says `all slots filled`. Then:

```bash
git add rag/ scratch/ evidence/ pyproject.toml uv.lock
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

1. Paste your `tokenize` line. Say what happened to `Don’t`, to `well-known` and to `Ulysses's`, and which line of `strip_punctuation` or `tokenize` did each.

   *How to get it:* the curly apostrophe is punctuation by Unicode category, so `strip_punctuation` turned it into a space and `t` was dropped by `len(t) > 1`; the hyphen became a space, so two tokens; the possessive left an `s` that the same length rule dropped.

2. Your `shortest` passage is exactly `MIN_CHARS` or longer. Say why the loop in `load_passages` cannot produce a shorter one, and what happens to the text left over at the end of the file.

   *How to get it:* read the `if len(buf) >= min_chars` line and the two `if buf` lines after the loop in `rag/corpus.py`.

3. The chapter stems after it removes stopwords. Say what goes wrong if you stem first, with one stopword from the list that would stop matching.

   *How to get it:* `uv run python -c "from nltk.stem import PorterStemmer; print(PorterStemmer().stem('was'), PorterStemmer().stem('this'))"` prints what those two become.
