# A13 · Text In, Numbers Out: Preprocessing and TF-IDF

**Meetings:** D27–D28 · **Points:** 15 pts


**Watch**
Day 1 — [Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) ch. 1 Preprocessing, about 25 minutes of lessons
Day 2 — Boot.dev ch. 2 TF-IDF, about 25 minutes of lessons

Boot.dev has no video: short readings, a function to write, tests run on your machine by its CLI.

**During the video**

Boot.dev's course builds its own project on its own dataset. Keep that project in its own folder, `~/version_control/bootdev-rag/`, outside every course repo, so none of its files end up in a course commit.

**Type every solution yourself.** Do not paste from the hint panel or the solution tab. The walkthrough below has you write the same function a second time on your corpus, and the second time only goes quickly if the first time went through your fingers.

**Run before you submit.** Every lesson, run `bootdev run <lesson-id>` first, without `-s`. That runs the tests locally and shows you what they check. Read the output, fix, and only then `bootdev run <lesson-id> -s` to submit.

**Ch. 1 · keep a list.** In `scratch/a13-order.md` in your rag repo, write down every operation the chapter applies to text, in the order it applies them. Step 4 asks you to defend an order, and the chapter's is the one to argue with.

**Ch. 2 · copy the formula exactly.** When the chapter gives you the IDF formula, copy it into `scratch/a13-order.md` character for character, including every `+ 1`. Step 7 has a line marked `# CHAPTER FORMULA` and that is where it goes.

**Notes**

**Install the Boot.dev CLI before you touch Chapter 1.** It is a Go program, so Go has to be installed for it to build:

```bash
brew install go
go install github.com/bootdotdev/bootdev@latest
bootdev login
```

If `bootdev` is not found after the install, `$(go env GOPATH)/bin` is not on your PATH. Add `export PATH="$PATH:$(go env GOPATH)/bin"` to your shell profile and open a new terminal.

Where the course calls a model, it uses free models through OpenRouter. A `429` means you are throttled, not broken. Wait sixty seconds; hammering Run makes the cooldown longer.

**The order of preprocessing operations changes the output, silently.** If you stem before removing stopwords, your stopword list stops matching (`"was"` stems to `"wa"`). If you split into tokens before stripping punctuation, `"dog."` and `"dog"` become two different terms and every score after that is wrong. Lowercasing and stripping punctuation can go in either order.

**`string.punctuation` is ASCII only.** It does not contain the curly apostrophe `’`, the em dash `—` or curly quotes, and a lot of real text is full of them. If your corpus came from a modern source, `“Ithaca”` with curly quotes survives an ASCII strip as a different term from `ithaca`. The walkthrough strips by Unicode category instead, which catches all of them.

**Use the formula the chapter gives you, not one you remember.** There are at least four common IDF variants, differing in where the `+ 1`s go. The grader compares floats. A different smoothing choice is not close enough.

**Walkthrough — the same pipeline on your corpus**

Boot.dev grades its version on its dataset. This runs yours on your corpus, the one you locked in A05b. Everything from A13 to A22 runs on that file.

**Step 1. Accept the repo, open the track log.**

Track A lives in a new repo. Accept it once, here: **[RAG](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/rag/accept)**

```bash
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-rag-<your-username>.git
cd hse-2026-2027-rag-<your-username>
./setup.sh
git switch -c dev/tfidf
```

`setup.sh` again: hooks do not travel with a clone.

Your Unit 0 log branch merged with A12. The track gets a new one, open until the end of the fall:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git switch main && git pull
git switch -c <your-username>-track
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Boot.dev CLI installed, `bootdev login` done
- [ ] Boot.dev ch. 1 Preprocessing, every lesson green
- [ ] `scratch/a13-order.md` with the chapter's operation order
- [ ] Corpus copied in, sha256 matches `data/SOURCE.md` (Steps 2–3)
- [ ] `rag/corpus.py` and `rag/preprocess.py` running (Steps 4–6)
- [ ] Push `dev/tfidf` and open the PR
```

**Step 2. Bring your corpus over, byte for byte.**

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
cp ../hse-2026-2027-foundations-<your-username>/scripts/fetch_corpus.sh scripts/
cp ../hse-2026-2027-foundations-<your-username>/data/SOURCE.md data/
bash scripts/fetch_corpus.sh
shasum -a 256 data/corpus.txt
grep -i sha256 data/SOURCE.md
```

*You should see* the same 64-character hash twice. Not a similar one: the same one. Every number you produce this semester is a number about that exact file, and a hash that differs means you are measuring a different corpus from the one you described.

*If it broke:* a different hash usually means the source changed since A05b or your script fetches "latest". Come and find me before you go on; do not edit `SOURCE.md` to match.

**Step 3. `rag/corpus.py`: the corpus as passages.**

TF-IDF scores terms per document. Your corpus is one file, so for A13 to A15 a "document" is a passage: the blank-line blocks of your file, merged forward until each one is at least `MIN_CHARS` long. Set `MIN_CHARS` to the minimum you settled on in A06. (A16 replaces passages with real chunks.)

```python
# rag/corpus.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.txt"
MIN_CHARS = 200          # your A06 minimum

def load_text():
    return CORPUS.read_text(encoding="utf-8")

def load_passages(min_chars=MIN_CHARS):
    passages, buf = [], ""
    for block in load_text().split("\n\n"):
        block = " ".join(block.split())
        if not block:
            continue
        buf = f"{buf} {block}" if buf else block
        if len(buf) >= min_chars:
            passages.append(buf)
            buf = ""
    if buf and passages:
        passages[-1] += " " + buf
    elif buf:
        passages.append(buf)
    return [{"id": f"p{i:04d}", "text": t} for i, t in enumerate(passages)]

if __name__ == "__main__":
    ps = load_passages()
    lengths = sorted(len(p["text"]) for p in ps)
    print(len(ps), "passages")
    print("shortest", lengths[0], "median", lengths[len(lengths) // 2], "longest", lengths[-1])
    print(ps[0]["id"], ps[0]["text"][:80])
    print(ps[-1]["id"], ps[-1]["text"][-80:])
```

```bash
uv run python -m rag.corpus
```

*You should see* a passage count in the hundreds or low thousands, a `shortest` that is at least `MIN_CHARS`, and first and last lines that match the first and last words of `data/corpus.txt`. The leftover at the end is merged backward into the last passage rather than dropped, which is why nothing is shorter than the minimum.

`-m rag.corpus` and not `rag/corpus.py`: running it as a module is what lets `rag/tfidf.py` say `from rag.corpus import ...` in Step 7. Every command this semester uses `-m`.

**Step 4. `rag/preprocess.py`.**

```bash
uv add nltk
```

NLTK is here only for its Porter stemmer. If Boot.dev's chapter had you use a different stemmer, use that one instead, so the two implementations agree.

```python
# rag/preprocess.py
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

def strip_punctuation(text):
    return "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in text)

def tokenize(text):
    return [t for t in strip_punctuation(text.lower()).split() if len(t) > 1]

def preprocess(text, stopwords=STOPWORDS, stem=True, drop_stems=frozenset()):
    tokens = [t for t in tokenize(text) if t not in stopwords]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return [t for t in tokens if t not in drop_stems]

if __name__ == "__main__":
    print(tokenize("Don’t stop—well-known Ulysses's ships."))
    print(preprocess("The ships were sailing to Ithaca, and the sailors rowed."))
```

Punctuation becomes a space rather than vanishing, so `well-known` is two terms and the em dash does not glue two words together. The possessive `'s` leaves a stray `s`, which the one-letter rule drops. That is a decision, and the docstring states it. If you choose differently from Boot.dev's chapter, change the docstring and say why in one line under it. The deliverable is the docstring as much as the code.

`drop_stems` is empty for now. The extension fills it, and it runs **after** stemming on purpose.

**Step 5. Run it.**

```bash
uv run python -m rag.preprocess
```

*You should see* exactly this, because it does not depend on your corpus:

```
['don', 'stop', 'well', 'known', 'ulysses', 'ships']
['ship', 'sail', 'ithaca', 'sailor', 'row']
```

*If it broke:* `don’t` surviving with its curly apostrophe means your strip is ASCII only. `ulyssess` means you deleted punctuation instead of turning it into a space.

**Step 6. Push the first day.**

```bash
git add rag/corpus.py rag/preprocess.py scripts/fetch_corpus.sh data/SOURCE.md scratch/a13-order.md pyproject.toml uv.lock
git commit -m "A13: corpus passages and preprocessing pipeline"
git push -u origin dev/tfidf
```

Open the pull request now, not at the end: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Then close the log with `bash scripts/sign-off.sh` and `git add logs && git commit && git push` in the log repo.

**Day 2 starts here.** In the rag repo: `git switch dev/tfidf && git pull`. In the log repo: `bash scripts/start-entry.sh`, then:

```markdown
- [ ] Boot.dev ch. 2 TF-IDF, every lesson green
- [ ] Formula copied into `scratch/a13-order.md`
- [ ] `rag/tfidf.py` with the chapter's formula (Steps 7–8)
- [ ] Extension: corpus stopwords, before/after numbers
- [ ] Push, sign off
```

**Step 7. `rag/tfidf.py`.**

```python
# rag/tfidf.py
import math
from collections import Counter

from rag.corpus import ROOT, load_passages
from rag.preprocess import preprocess

class TfidfIndex:
    def __init__(self, units, **pp):
        self.units = units
        self.pp = pp
        self.docs = [Counter(preprocess(u["text"], **pp)) for u in units]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.N = len(units)
        self.df = Counter()
        for counts in self.docs:
            self.df.update(counts.keys())

    def tf(self, term, i):
        return self.docs[i][term] / max(self.lengths[i], 1)

    def idf(self, term):
        return math.log((self.N + 1) / (self.df[term] + 1))   # CHAPTER FORMULA

    def tfidf(self, term, i):
        return self.tf(term, i) * self.idf(term)

    def top_terms(self, i, n=5):
        return sorted(self.docs[i], key=lambda t: (-self.tfidf(t, i), t))[:n]

if __name__ == "__main__":
    passages = load_passages()
    ix = TfidfIndex(passages)
    raw = TfidfIndex(passages, stopwords=frozenset())
    print("passages:", ix.N, " vocabulary:", len(ix.df))
    once = sum(1 for n in ix.df.values() if n == 1)
    print("terms in exactly one passage:", once, f"({once / len(ix.df):.0%})")
    print("idf('the') with stopwords kept:", round(raw.idf("the"), 4), " df:", raw.df["the"])
    print("largest possible idf:", round(math.log((ix.N + 1) / 2), 4))
    for i in (0, ix.N // 2, ix.N - 1):
        print(passages[i]["id"], ix.top_terms(i))
```

The `idf` line is one common variant. Replace it with the formula from `scratch/a13-order.md`, and if the chapter's term frequency is a raw count rather than a count divided by length, change `tf` to match. If your `largest possible idf` line no longer matches your formula for a term in one passage, fix that line too.

**Step 8. Run it and read it.**

```bash
uv run python -m rag.tfidf
```

*You should see* four things:

| Line | Shape it should have |
|---|---|
| `terms in exactly one passage` | a large share of the vocabulary, often a third to over half. Real vocabularies are mostly words that appear once. |
| `idf('the')` | tiny: under a tenth of the largest possible idf. `the` is in most passages, so it carries almost no information about which one you want. Its tf-idf in any passage is near zero even though its count is the highest. |
| `largest possible idf` | the idf of every term in exactly one passage. Those are your rarest words. |
| the three `top_terms` lists | names, rare nouns and distinctive words from that passage. No function words. If one list is numbers or fragments (`['183', 'carp', 'er', '182', '273']`), you landed in a footnote block or a table of contents; that is a real artifact of your file, and Reflection 3 wants it. |

If a top-terms list is full of words like `said` or your main character's name, that is not a bug. It is the extension.

*If it broke:* `idf('the')` larger than the largest possible idf means `raw.df["the"]` is 0: the `stopwords=frozenset()` argument never reached `preprocess`. A typo in a term gives the same symptom and no error, because a `Counter` returns 0 for anything it has not seen.

**Extension — your corpus's own stopwords (ASSIGNED)**

A generic stopword list does not know your corpus. In a corpus like the Odyssey, the hero's name is in a large share of the passages, so it separates few of them from each other, and no English stopword list includes it. Your corpus has words like that too. Find them, measure what removing them does, and name what it cost.

Add this to the bottom of `rag/tfidf.py`, inside the `__main__` block:

```python
    THRESHOLD = 0.20
    common = sorted(t for t, n in ix.df.items() if n / ix.N >= THRESHOLD)
    for th in (0.10, 0.20, 0.30):
        print(f"df >= {th:.0%}:", sum(1 for n in ix.df.values() if n / ix.N >= th), "terms")
    print("corpus stopwords:", common)
    (ROOT / "rag" / "corpus_stopwords.txt").write_text("\n".join(common) + "\n", encoding="utf-8")

    cut = TfidfIndex(passages, drop_stems=frozenset(common))
    print("vocabulary before/after:", len(ix.df), len(cut.df))
    before, after = sum(ix.lengths), sum(cut.lengths)
    print(f"term occurrences removed: {1 - after / before:.1%}")
    changed = sum(len(set(ix.top_terms(i)) ^ set(cut.top_terms(i))) // 2 for i in range(ix.N))
    print(f"top-5 slots changed: {changed} of {5 * ix.N}")
```

Run it. The list is stems, which is why `drop_stems` filters **after** stemming. Filter it against raw tokens and nothing matches.

Then, in `rag/corpus_notes.md`, write:

1. The three threshold counts and which threshold you kept. 20% is a starting point, not an answer.
2. Your list, and next to each word: noise or meaning.
3. Vocabulary before and after, the share of term occurrences removed, and top-5 slots changed.
4. **Where it got worse.** Pick one word on your list that carries meaning for a question someone would ask about your corpus, write that question, and say what removing the word does to it. A list with nothing worth keeping on it means your threshold is too low.

Commit `rag/corpus_stopwords.txt` and `rag/corpus_notes.md` separately from the code, so the list and the decision are their own commit:

```bash
git add rag/tfidf.py
git commit -m "A13: TF-IDF index with the chapter's formula"
git add rag/corpus_stopwords.txt rag/corpus_notes.md
git commit -m "A13: corpus stopwords and what they changed"
git push
```

Save a screenshot of each chapter's page on Boot.dev with every lesson complete as `evidence/a13-bootdev-ch1.png` and `evidence/a13-bootdev-ch2.png`, commit them, push. One PR, not two; push again each day. Then `bash scripts/sign-off.sh` in the log repo, and `git add logs && git commit && git push`.

**Deliverable**
`rag/corpus.py` + `rag/preprocess.py` (docstring states your operation order) + `rag/tfidf.py` + `rag/corpus_stopwords.txt` + `rag/corpus_notes.md` + `scripts/fetch_corpus.sh` + `data/SOURCE.md` + two Boot.dev screenshots in `evidence/`.

**Reflection Questions**

1. Paste your `idf('the')` line and your `largest possible idf` line. Then pick one term from your own `top_terms` output and compute its tf-idf by hand from its count in that passage, the passage's length, and its df: show the arithmetic and the value `ix.tfidf` gives. If they differ, say which part of your formula or your code accounts for the difference.
2. Paste your corpus stopword list with the noise/meaning label on each word. For the one you marked meaning, paste the question you wrote and say what happens to its terms under `preprocess(question, drop_stems=...)`: paste both outputs, with and without the list. Did you keep the word out anyway, and why?
3. Before you ran Step 8, did you expect the share of terms appearing in exactly one passage to be higher or lower than it was? Write the number you got and pick three of those one-passage terms from your corpus. For each, say whether it is a real word, a name, a typo, or an artifact of your preprocessing, and what your `strip_punctuation` or stemmer did to produce the artifacts.
