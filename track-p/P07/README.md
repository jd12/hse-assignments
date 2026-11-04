# P07 · Vectors, by Hand, Then the Model

**Meetings:** D30 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 4 Semantic Search, lessons 1 to 9, about 25 minutes.

**During the chapter**

**Write the model down.** Lesson 3 names the embedding model and lesson 3's check prints one number about it. Write the model's name, that number, and the vector dimension once lesson 10 prints it (P08) into `scratch/p07-model.md` in the rag repo. Lesson 9 says why the chapter uses cosine similarity and not the dot product; write that reason under the model in one sentence of your own.

**Predict the paraphrase queries.** Your P06 scoreboard has two `paraphrase` queries that keyword search missed. Before P08 runs semantic search on them, write under `## Predictions` in `evidence/P07.md` which of your five queries you expect semantic search to find in the top 3 and which you expect it to miss, one sentence each. Commit the file; I check the timestamps. P08 checks the predictions.

**Notes**

**Lessons 5, 7 and 8 are typed in the browser, not in your project.** They have a `main.py` on the page and **Run** and **Submit** buttons; there is no `bootdev run` for them. Each is one short function with a loop, and the chapter has you write them by hand once before NumPy does them for you in lesson 14.

**Lesson 3 downloads about 80 MB of model and a much larger library behind it.** `uv add sentence-transformers` pulls in PyTorch; on school wifi that is several minutes. Start it, then read lessons 4 to 9 while it runs. The model files land in a cache in your home folder the first time `verify` runs, and every later run loads them from there.

**A 0.9 and a 0.99 are not the same answer.** A06 taught you that cosine similarity between unrelated English sentences sits well above zero. Lesson 8's example makes the same point with magnitudes: two vectors can point the same way and differ in length by a factor of five, and the dot product sees the length while cosine does not. P08's walkthrough prints real scores on your corpus.

**Walkthrough — the chapter**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P05 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/semantic
```

*If P05 is not approved yet:* `git switch dev/keyword && git switch -c dev/semantic`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 4, lessons 1 to 9, every lesson green (three in the browser, two with bootdev run)
- [ ] scratch/p07-model.md: the model, its sequence limit, why cosine; my five predictions in evidence/P07.md, committed
- [ ] Step 3: dot and cosine on two of my corpus sentences, by hand, in evidence/P07.md
- [ ] Fill every slot in evidence/P07.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 4, lessons 1 to 9.**

In `~/version_control/rag-search-engine/`. Stop after lesson 9; lessons 10 to 16 are P08. Only lessons 1 and 3 touch the project; 5, 7 and 8 are in the browser; 2, 4, 6 and 9 are readings with a question at the end.

**Lesson 1: a second CLI.** Semantic search gets its own entry point, `cli/semantic_search_cli.py`, beside the keyword one. The course gives you the skeleton to paste. Its `match args.command` needs a `command` attribute to exist, so add the subparsers line now rather than in lesson 3:

```python
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
```

Run, then submit:

```bash
uv run cli/semantic_search_cli.py --help
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `usage: semantic_search_cli.py`; that is the check.

**Lesson 2: embeddings.** A reading: a vector is a list of numbers, and the model puts similar meanings at nearby points. One multiple-choice question.

**Lesson 3: the model.** The chapter uses `all-MiniLM-L6-v2`, a small model that runs on your own machine with no key and no network after the first download.

1. Add the library. This is the slow step; start it and keep reading:

```bash
uv add sentence-transformers
```

2. Create `cli/lib/semantic_search.py`. The constructor takes the model name with the chapter's as the default, which chapter 5 relies on:

```python
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")


# In: nothing.  Out: nothing; prints the model and its sequence limit.
def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
```

`device="cpu"` is the line the lesson says to add if you get an error about your GPU; adding it up front costs nothing on a laptop.

3. In the CLI, the import, a `verify` subcommand and its case:

```python
from lib.semantic_search import verify_model
```

```python
    subparsers.add_parser("verify", help="Load the embedding model and print its limits")
```

```python
        case "verify":
            verify_model()
```

4. Run, then submit:

```bash
uv run cli/semantic_search_cli.py verify
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* a download progress bar the first time, then a description of the model and `Max sequence length: 256`; that is the check, and the number goes into `scratch/p07-model.md`. It is the most tokens the model reads from one text; chapter 5 is about what to do with text longer than that. *If it broke* with `ModuleNotFoundError: No module named 'sentence_transformers'`, the `uv add` did not finish, or you ran with `python` instead of `uv run`. *If it broke* with a long error mentioning `MPS` or `CUDA`, `device="cpu"` is missing.

**Lesson 4: model selection.** A reading on general, domain and multilingual models. One multiple-choice question.

**Lesson 5: vector operations.** In the browser. `add_vectors` and `subtract_vectors` each check the two lengths match, raise `ValueError` if not, and loop over the positions:

```python
def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(f"vectors have different lengths: {len(vec1)} and {len(vec2)}")
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(f"vectors have different lengths: {len(vec1)} and {len(vec2)}")
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] - vec2[i])
    return result
```

**Run** then **Submit** on the page. *You should see* the tests in `main_test.py` pass.

**Lesson 6: dimensions.** A reading: the chapter's vectors have hundreds of numbers, and distance still means similarity. A Boots question; answer in the chat box until it is satisfied.

**Lesson 7: the dot product.** In the browser. Multiply position by position and add up:

```python
def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"vectors have different lengths: {len(vec1)} and {len(vec2)}")
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total
```

The lesson's example, `[.8, .5, .5]` against `[.5, .4, .6]`, is `0.4 + 0.2 + 0.3`, which is `0.9`. Check your function gives it.

**Lesson 8: cosine similarity.** In the browser, with `dot` and `euclidean_norm` handed to you from `lib`. The dot product divided by the two lengths, with a guard for a zero-length vector:

```python
from lib import dot, euclidean_norm


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"vectors have different lengths: {len(vec1)} and {len(vec2)}")
    magnitude1 = euclidean_norm(vec1)
    magnitude2 = euclidean_norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot(vec1, vec2) / (magnitude1 * magnitude2)
```

The lesson's example: `[0.6, 0.8]` and `[3.0, 4.0]` have a dot product of `5.0` and a cosine of `1.0`, because they point the same way and only differ in length. That is the whole reason for the division.

**Lesson 9: why cosine.** A reading: the model was trained with cosine similarity, so search with it. A Boots question. The one-sentence answer goes in `scratch/p07-model.md`.

**Step 3. Dot and cosine on your corpus, by hand.**

Two sentences from your corpus, scored the lesson 7 and 8 way, on tiny vectors you make yourself so the arithmetic is visible. Count three words in each sentence, so each sentence becomes a 3-number vector of counts; on Homer, the words `ship`, `sea` and `god` in two sentences from book 9:

```bash
uv run python -c "
from rag.preprocess import preprocess
words = ['ship', 'sea', 'god']
a = 'We then sailed on with heavy hearts, glad to have escaped death though we had lost our comrades, and our ships went over the sea.'
b = 'The gods sent us a fair wind, and the ships ran before it over the sea.'
va = [preprocess(a).count(w) for w in words]
vb = [preprocess(b).count(w) for w in words]
dot = sum(x * y for x, y in zip(va, vb))
na = sum(x * x for x in va) ** 0.5
nb = sum(x * x for x in vb) ** 0.5
print(va, vb, 'dot', dot, 'cosine', round(dot / (na * nb), 3))
"
```

*You should see* two short vectors, their dot product and their cosine. On Homer: `[1, 1, 0] [1, 1, 1] dot 2 cosine 0.816`. Replace the three words and the two sentences with your own, picking words that are in at least one of the sentences, and paste the line under `## Step 3` in `evidence/P07.md`. Then double every count in `va` by hand and say what happens to the dot product and to the cosine. *If it broke* with `cosine` dividing by zero, one of your vectors is all zeros: none of the three words is in that sentence after `preprocess`, which stems, so `ships` and `ship` both count as `ship` but `gods` and `god` need checking.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P07.md` (the questions are below) and run `uv run python scripts/check_evidence.py P07` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/
git commit -m "P07: chapter 4 started, the model written down, predictions for semantic search"
git push -u origin dev/semantic
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P08.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P07.md` with every slot filled + `scratch/p07-model.md`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P07.md`)

1. Paste your Step 3 line and the doubled version. Say why the dot product changed and the cosine did not, and which of the two the chapter's search uses, with the lesson 9 reason.

   *How to get it:* doubling every number in `va` doubles the dot product and doubles `na`, and the two cancel in the division. Lesson 8's `[0.6, 0.8]` against `[3.0, 4.0]` is the same fact with a factor of five.

2. Lesson 3's check printed `Max sequence length: 256`. Say what the number is a limit on, and what the model does with your longest passage from P02 (its length is in your P02 Step 3 paste) if that passage is longer than the limit.

   *How to get it:* the lesson says tokens, not characters, and the model's own tokenizer decides how many a text has; roughly one token per short word. A text past the limit is cut off at the limit, and the rest is never embedded, which is the problem chapter 5 solves.

3. Paste the predictions you wrote for your five queries. For one `paraphrase` query, say in one sentence what you expect the vector to match on that the stems could not.

   *How to get it:* the answer passage and the query share no stems, by construction. If semantic search finds it, something other than the words matched: the lesson 2 reading says what.
