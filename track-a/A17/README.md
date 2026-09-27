# A17 · Query Expansion and Reranking

**Meetings:** D32–D33 · **Points:** 15 pts


**Watch — 23 min**
Day 1 — 9 min
[DeepLearning.AI, *Advanced Retrieval for AI with Chroma*](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) lesson 4, Query Expansion (9m)
Day 2 — 14 min
Lesson 5, Cross-encoder re-ranking (6m) · lesson 6, Embedding adaptors (8m)

Both days are short on video on purpose. The walkthrough is long, and Day 2's install takes time.

*Your Boot.dev subscription ends after A16. From here the retrieval arc runs on DeepLearning.AI, which is free.*

**During the video**

These are notebook lessons, and the notebook runs on the DeepLearning.AI page. **Type along in the page, not by pressing Shift-Enter on his cells.** Under each of his code cells, add a cell of your own (the **+** button), type his code into it, and run yours. When a cell of his only loads a file or prints a banner, run it as it is.

Before you close the tab each day, copy the functions you typed into `scratch/a17-video.py` in your rag repo. It does not have to run on your machine. It is committed, and it is how I see you typed along.

**Lesson 4 · type and run.** He writes two kinds of expansion: one asks the model for a hypothetical answer and searches with the question plus that answer, the other asks for several related questions and searches with all of them. Type both. When he plots the queries against the dataset, pause on the plot and write one sentence in `scratch/a17-video.py`, as a comment, on where the expanded query landed relative to the original.

**Lesson 5 · type and run.** The cross-encoder cells: loading the model, scoring `(query, document)` pairs, and re-sorting. Type the scoring loop yourself; it is five lines, and Day 2's Step 6 is the same five lines on your chunks.

**Lesson 6 · watch; typing along is optional.** An embedding adaptor is a matrix trained to bend query vectors toward the documents that answered them. It needs labeled relevance data, and you will not have that until A19. Watch for what data he trains it on and where that data comes from; the extension asks you one paragraph about it.

**Notes**

**Expansion calls a chat model, so it costs money and it is not deterministic.** The walkthrough caches every chat response in `data/llm_cache.json`, keyed by the model and the exact messages. Run a query twice and the second run is free and identical. When a later assignment needs to measure run-to-run spread, it turns the cache off; today it stays on.

**A hypothetical answer can be confidently wrong, and it still helps.** The expansion is searched, not shown. A made-up paragraph in the style of your corpus lands near real paragraphs in the style of your corpus. The risk is a query where the made-up answer is about something else entirely and drags the search with it. That is one of the queries the extension asks you to find.

**Reranking is two stages, and the point is cost.** The retriever is cheap: pull 30 candidates. The cross-encoder is expensive: read each `(query, chunk)` pair jointly and score it. Never rerank the whole corpus. A cross-encoder cannot be precomputed the way embeddings can, because it needs the query, which is why it is slow.

**The cross-encoder reads about 512 tokens and silently ignores the rest.** A 200-word chunk fits. A 400-word chunk does not, and the answer in its second half is invisible to the reranker. If you froze a large chunk size in A16, this is where it costs you.

**Walkthrough — expansion and reranking on your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/hybrid && git pull       # skip this line if A16 has merged
git switch -c dev/rerank
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Lesson 4 Query Expansion, typed along in the page
- [ ] `scratch/a17-video.py` with both expansion functions
- [ ] `rag/llm.py` with the cache (Step 2)
- [ ] `rag/expand.py`, both methods (Steps 3–4)
- [ ] Expansion predictions committed, then scored (Step 5)
- [ ] Push, PR, sign off
```

**Step 2. `rag/llm.py`.**

The chat endpoint, called the way you have called the embeddings endpoint since A05, with a cache in front of it.

```python
# rag/llm.py
import hashlib
import json
import os
import urllib.request

from rag.corpus import ROOT

CHAT_MODEL = "gpt-4o-mini"
CACHE = ROOT / "data" / "llm_cache.json"

def chat(messages, model=CHAT_MODEL, temperature=0, cache=True):
    key = hashlib.sha256(json.dumps([model, messages]).encode()).hexdigest()
    store = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if cache and key in store:
        return store[key]
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    if cache:
        store[key] = text
        CACHE.write_text(json.dumps(store), encoding="utf-8")
    return text
```

```bash
uv run python -c "from rag.llm import chat; print(chat([{'role': 'user', 'content': 'Say ready.'}]))"
```

*You should see* a one-word reply. Run it again: the same reply, instantly, because it came from the cache.

**Step 3. `rag/expand.py`.**

Put the title from your `data/SOURCE.md` in `TITLE`. The model writes better expansions when it knows what it is imitating.

```python
# rag/expand.py
from rag import search
from rag.hybrid import rrf
from rag.llm import chat

TITLE = "<the title from data/SOURCE.md>"

def hypothetical(query):
    return chat([
        {"role": "system", "content": f"You are an expert on {TITLE}. Write one short paragraph, "
                                      "in its style, that answers the question. Do not hedge."},
        {"role": "user", "content": query}])

def related(query, n=3):
    text = chat([
        {"role": "system", "content": f"You help search {TITLE}. Write {n} different short search "
                                      "queries that would find passages answering the question. "
                                      "One per line, no numbering."},
        {"role": "user", "content": query}])
    return [line.strip() for line in text.splitlines() if line.strip()][:n]

def expand_search(query, k=5, pool=50):
    units, kw, sem = search.engines()
    by_id = {u["id"]: u for u in units}
    rankings = [[r["id"] for r in kw.search(query, pool)],
                [r["id"] for r in sem.search(query + "\n" + hypothetical(query), pool)]]
    return [by_id[i] for i in rrf(rankings, top=k)]

def multi_search(query, k=5, pool=50):
    units = search.engines()[0]
    by_id = {u["id"]: u for u in units}
    rankings = [[r["id"] for r in search.hybrid(q, k=pool)] for q in [query] + related(query)]
    return [by_id[i] for i in rrf(rankings, top=k)]
```

`expand_search` is lesson 4's first method inside your hybrid search: the keyword side still gets the original query, and only the semantic side sees the hypothetical answer. `multi_search` is the second method: hybrid-search each query, then fuse all the rankings.

**Step 4. Look at what the model wrote before you trust it.**

```bash
uv run python -c "
from rag.expand import hypothetical, related
q = '<one of your paraphrase queries>'
print(hypothetical(q)); print('---'); print(related(q))
"
```

*You should see* a paragraph that sounds like your corpus and three short queries. Read the paragraph for facts. If it states something your corpus does not say, that is normal, and it is worth noticing which words it used that the real passage uses too.

**Step 5. Predict, then score.** In `scratch/a17-predictions.md`, write the ids of the queries you expect `expand` to improve over `hybrid`, and the ones you expect it to make worse. Commit it before you run anything:

```bash
git add rag/llm.py rag/expand.py scratch/a17-video.py scratch/a17-predictions.md
git commit -m "A17: query expansion, predictions before the run"
git push -u origin dev/rerank
```

Then add both methods to the dictionary in `rag/scoreboard.py`:

```python
from rag import expand
```

```python
        "expand": lambda q: expand.expand_search(q, 5),
        "multi": lambda q: expand.multi_search(q, 5),
```

```bash
uv run python -m rag.scoreboard
```

*You should see* five columns. The first three must match your frozen A16 row exactly. Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Sign off the log.

**Day 2 starts here.** In the rag repo: `git switch dev/rerank && git pull`. In the log repo: `bash scripts/start-entry.sh`, then:

```markdown
- [ ] Lesson 5 Cross-encoder re-ranking, typed along in the page
- [ ] Lesson 6 Embedding adaptors, watched
- [ ] `sentence-transformers` installed, model downloaded (Step 6)
- [ ] `rag/rerank.py`, reranker on the scoreboard (Steps 6–7)
- [ ] Extension: `rag/rerank_deltas.md`, latency, adaptor paragraph
- [ ] Push, sign off
```

**Step 6. `rag/rerank.py`.**

```bash
uv add sentence-transformers
```

It pulls PyTorch. Let it finish.

```python
# rag/rerank.py
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import search

@lru_cache(maxsize=1)
def model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, k=5):
    scores = model().predict([(query, c["text"]) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [dict(candidates[i], score=float(scores[i]), first_stage=i + 1) for i in order[:k]]

def retrieve(query, k=5, pool=30):
    return rerank(query, search.hybrid(query, k=pool), k)
```

It is the model lesson 5 uses. `first_stage` records where hybrid search had each chunk before the reranker moved it, which is the column `rerank_deltas.md` is built from. `retrieve` is the function A18, A19 and A20 call.

```bash
uv run python -c "
from rag.rerank import retrieve
for r in retrieve('<one of your queries>'):
    print(round(r['score'], 2), r['first_stage'], r['id'], r['text'][:60])
"
```

*You should see* a one-time model download, then five rows. The scores are not similarities: they are raw model outputs, positive for pairs the model thinks match and negative for pairs it thinks do not, with no fixed upper or lower bound. `first_stage` values are between 1 and 30. Usually at least one is above 5, a chunk hybrid search had outside its top five; if all five came from 1–5, try one of your paraphrase queries.

**Step 7. Put it on the scoreboard, with hit@1.** Add two entries to the dictionary:

```python
from rag import rerank
```

```python
        "rerank": lambda q: rerank.retrieve(q, 5),
        "exp+rr": lambda q: rerank.rerank(q, expand.expand_search(q, 30), 5),
```

Reranking is about the top slot, so count it. In `__main__`, next to `totals`, add a `firsts` counter:

```python
    firsts = dict.fromkeys(ms, 0)
```

inside the loop, under the `totals` line:

```python
            firsts[name] += rank == 1
```

and after the `hit@3` line:

```python
    print("hit@1 ", *(f"{firsts[n]}/{len(QUERIES)}".rjust(8) for n in ms))
```

```bash
uv run python -m rag.scoreboard
```

*You should see* seven columns and two total rows.

**Extension — what the reranker moved (ASSIGNED)**

**Part 1. The scoreboard.** Paste the seven-column table into `rag/SCOREBOARD.md` under `## A17`, with both total rows. Under it, go back to `scratch/a17-predictions.md` and mark each prediction right or wrong.

**Part 2. `rag/rerank_deltas.md`.** Five queries. For each, the top 5 from `search.hybrid` and the top 5 from `rerank.retrieve`, side by side, with the answer chunk marked. Pick them so that at least one is a query where reranking moved the answer up from below rank 5, and at least one where it moved the answer down or changed nothing useful. For each, one sentence on what the promoted chunk has that the demoted one does not.

**Part 3. Latency.** Time 10 queries through each:

```bash
uv run python -c "
import time
from rag import search, rerank
from rag.scoreboard import QUERIES
rerank.retrieve('warm up')
for name, fn in (('hybrid', search.hybrid), ('rerank', rerank.retrieve)):
    t0 = time.perf_counter()
    for q in QUERIES: fn(q['query'])
    print(name, round((time.perf_counter() - t0) * 100), 'ms per query')
"
```

*You should see* reranking cost more per query than hybrid, by an amount that depends on your laptop and your chunk size. The first call loads the model, which is why there is a warm-up.

**Part 4. Embedding adaptors, one paragraph.** What lesson 6 trained the adaptor on, where those labels came from, and whether your corpus would benefit: how many labeled queries you would need to have, and whether A19's golden set could be used for it without ruining A19's numbers.

**Part 5. Where it got worse.** Name the method with the best hit@3 and give the query it lost that `hybrid` alone got. If expansion caused it, paste the hypothetical answer from `data/llm_cache.json`, or re-run Step 4 on that query and paste it.

```bash
git add rag/rerank.py rag/scoreboard.py rag/SCOREBOARD.md rag/rerank_deltas.md scratch/ pyproject.toml uv.lock
git commit -m "A17: cross-encoder reranking, deltas and latency"
git push
```

One PR, not two; push again each day. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`rag/llm.py` + `rag/expand.py` + `rag/rerank.py` + `rag/scoreboard.py` (seven methods) + `rag/SCOREBOARD.md` (A17 section) + `rag/rerank_deltas.md` (five queries, latency, adaptor paragraph) + `scratch/a17-video.py` + `scratch/a17-predictions.md`.

**Reflection Questions**

1. From `rerank_deltas.md`, take the query where the reranker promoted the answer from furthest down. Paste its `first_stage` number, the chunk's first 200 characters, and the hybrid rank-1 chunk's first 200 characters. What did BM25 and the embedding both reward in the wrong chunk that the cross-encoder did not?
2. Paste your committed predictions from `scratch/a17-predictions.md` and the `hybrid` and `expand` columns for every query you got wrong. For one of them, paste the hypothetical answer the model wrote and say which words in it pulled the search toward or away from the answer chunk.
3. Paste your two latency numbers and your hit@1 row. If your agent answers questions in a chat where people wait for a reply, is the difference in hit@1 worth the difference in milliseconds? Answer with the numbers, and name the kind of query in your ten where you would skip the reranker.
