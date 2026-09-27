# A19 · Measuring Retrieval: Recall, Precision, and the Golden Set

**Meetings:** D36–D37 · **Points:** 15 pts

**Watch — 42 min**
Day 1 — about 21 min
[DeepLearning.AI, *Building and Evaluating Advanced RAG*](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) lesson 3, RAG Triad of metrics (42m), first half: stop halfway through the lesson's running time
Day 2 — about 21 min
Lesson 3, second half, to the end

**During the video**

A notebook lesson on the DeepLearning.AI page. Same rule: a cell under each of his code cells, his code typed into it, yours run. Copy what you typed into `scratch/a19-video.py` before you close the tab each day.

**Write each metric as a comparison.** The triad is three scores, and each one compares two of three things: the question, the retrieved context, and the answer. Each time he defines one, pause and write one line in `scratch/a19-video.py` as a comment: the metric's name, which two of the three it compares, and what a low score means. By the end of the second half you have three lines, and the extension's judges are those three lines turned into prompts.

**Type the feedback-function cells.** Each metric is a feedback function built from a provider and a selector that says which part of the pipeline's record to read. Type them. Where he chains a method that picks out the context from the record, write a comment naming what it selects.

**Day 1 · labeling comes first.** Watch the first half, then spend the rest of the period on Steps 2–4. The golden set is committed before Day 2's runner exists.

**Notes**

**A retrieval eval needs no judge.** For each question, is a chunk you labeled relevant in the top k, yes or no? That is a set membership test, and it is why you build it before anything fancier. The triad is the opposite instrument: three model-graded scores. Running both on the same questions is the assignment, and you already know from A12 that a judge is believed only after it is checked against labels.

**Labels are made from the corpus, not from your retriever.** If you find the answer chunk by searching with your own pipeline, your golden set contains only answers your pipeline can find, and it scores itself perfectly. The walkthrough gives you an exact-phrase finder that bypasses every ranking you have built. Label with that.

**The golden set is committed before the runner runs.** I will check your commit timestamps. A label written after you have seen what the retriever returned is a label the retriever helped write.

**Small eval sets lie.** With 20 questions, one question flipping moves a score by 0.05. The runner prints `n` above every table. A difference of one question is not a finding.

**Measure this week; do not fix.** You will see questions your pipeline misses. Write them down. A20 is where changes happen, and a change made to fix a specific golden question gets named as one.

`recall@k` here means: at least one labeled chunk is in the top k. `precision@k`: the share of the top k that is labeled relevant. MRR: one over the rank of the first relevant chunk, averaged, with 0 when none is in the top 5. They answer different questions, and a system can look good on one and bad on another.

**Walkthrough — a golden set for your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/rag-pipeline && git pull     # skip this line if A18 has merged
git switch -c dev/retrieval-eval
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Lesson 3, first half, typed along in the page
- [ ] `scratch/a19-video.py` with the metrics written as comparisons so far
- [ ] `rag/eval/find.py` (Step 2)
- [ ] Twenty questions labeled in `rag/eval/golden.json` (Step 3)
- [ ] `rag/eval/check_golden.py` passes (Step 4)
- [ ] Golden set and predictions committed and pushed before Day 2
- [ ] Sign off
```

**Step 2. `rag/eval/find.py`: an exact-phrase finder.**

```python
# rag/eval/find.py
import sys

from rag.chunking import load_chunks
from rag.scoreboard import norm

phrase = norm(" ".join(sys.argv[1:]))
for c in load_chunks():
    if phrase in norm(c["text"]):
        print(c["id"], c["char_start"], c["text"][:100].replace("\n", " "))
```

```bash
uv run python -m rag.eval.find <a short phrase you know is in your corpus>
```

*You should see* one line per chunk containing the phrase. With overlap, a phrase near a chunk boundary is in two chunks. Both are correct labels, and both go in `chunk_ids`.

**Step 3. Write `rag/eval/golden.json`.**

Twenty entries. Open your corpus in an editor, pick the passage first, then write the question a reader would ask about it. Then find the chunk ids with `find.py` on a phrase from that passage.

```json
[
  {"id": "g01", "question": "What does the hero call himself to the one-eyed giant?",
   "chunk_ids": ["c00298"], "quote": "my name is Noman",
   "answer": "Noman", "kind": "paraphrase"}
]
```

That example is the Odyssey. `quote` is the sentence that answers the question, copied exactly. `answer` is the shortest string any correct answer must contain; A20 checks answers with it, so pick something that has one spelling. The mix:

| How many | `kind` | What it is |
|---|---|---|
| 6 | `paraphrase` | no content words shared with the answer chunk |
| 4 | `exact` | turns on a name, number or term that appears in few chunks |
| 6 | `ordinary` | what a reader of your corpus would actually ask |
| 4 | `two-chunk` | the answer needs two chunks that are not neighbors; list both |

Do not reuse your ten `queries.json` queries. Those have been tuned against for three weeks.

**Step 4. `rag/eval/check_golden.py`.**

```python
# rag/eval/check_golden.py
import json
from collections import Counter

from rag.chunking import OVERLAP, SIZE, load_chunks
from rag.corpus import ROOT
from rag.scoreboard import norm

GOLD = json.loads((ROOT / "rag" / "eval" / "golden.json").read_text(encoding="utf-8"))
chunks = {c["id"]: c for c in load_chunks()}
print(f"chunking {SIZE}/{OVERLAP}, {len(chunks)} chunks, {len(GOLD)} questions")
for g in GOLD:
    missing = [c for c in g["chunk_ids"] if c not in chunks]
    assert not missing, f"{g['id']}: no such chunk {missing}"
    assert any(norm(g["quote"]) in norm(chunks[c]["text"]) for c in g["chunk_ids"]), f"{g['id']}: quote not in its chunks"
    assert any(norm(g["answer"]) in norm(chunks[c]["text"]) for c in g["chunk_ids"]), f"{g['id']}: answer not in its chunks"
print("kinds:", dict(Counter(g["kind"] for g in GOLD)))
print("mean labeled chunks per question:", sum(len(g["chunk_ids"]) for g in GOLD) / len(GOLD))
```

```bash
uv run python -m rag.eval.check_golden
```

*You should see* your frozen chunk size, `20 questions`, the four kinds with counts 6, 4, 6, 4, and a mean between 1 and about 2. The first line proves the labels are on the chunking you froze in A16. If you ever change `SIZE` or `OVERLAP`, this script fails, which is the point of it.

**Step 5. Predict, then commit, before any runner exists.** In `rag/eval/predictions.md`, write what you expect recall@5 and MRR to be for `bm25`, `semantic`, `hybrid` and `rerank` on these twenty. Numbers, not words.

```bash
git add rag/eval/find.py rag/eval/golden.json rag/eval/check_golden.py rag/eval/predictions.md scratch/a19-video.py
git commit -m "A19: golden set of 20 and predictions, before any run"
git push -u origin dev/retrieval-eval
```

I will check your commit timestamps. Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Sign off the log.

**Day 2 starts here.** In the rag repo: `git switch dev/retrieval-eval && git pull`. In the log repo: `bash scripts/start-entry.sh`, then:

```markdown
- [ ] Lesson 3, second half, typed along in the page
- [ ] `rag/eval/metrics.py` and `rag/eval/run.py` (Steps 6–7)
- [ ] `rag/eval/triad.py` run over the twenty (extension)
- [ ] `rag/eval/triad_vs_labels.md`
- [ ] Push, sign off
```

**Step 6. `rag/eval/metrics.py`.**

```python
# rag/eval/metrics.py
def score(ranked_ids, relevant, k=5):
    relevant = set(relevant)
    first = next((r for r, cid in enumerate(ranked_ids, 1) if cid in relevant), None)
    return {
        "recall@1": 1.0 if first == 1 else 0.0,
        "recall@5": 1.0 if first is not None and first <= k else 0.0,
        "precision@5": len(relevant & set(ranked_ids[:k])) / k,
        "rr": 1.0 / first if first else 0.0,
    }
```

**Step 7. `rag/eval/run.py`.**

```python
# rag/eval/run.py
import json

from rag import rerank, search
from rag.corpus import ROOT
from rag.eval.metrics import score

GOLD = json.loads((ROOT / "rag" / "eval" / "golden.json").read_text(encoding="utf-8"))
METHODS = {"bm25": search.keyword, "semantic": search.semantic,
           "hybrid": search.hybrid, "rerank": rerank.retrieve}

print(f"n = {len(GOLD)} questions; one question moves a score by {1 / len(GOLD):.2f}")
print("method     recall@1 recall@5 prec@5   MRR")
runs = {}
for name, fn in METHODS.items():
    per = []
    for g in GOLD:
        ids = [r["id"] for r in fn(g["question"], 5)]
        per.append(dict(score(ids, g["chunk_ids"]), id=g["id"], kind=g["kind"], ranked=ids))
    runs[name] = per
    mean = lambda m: sum(p[m] for p in per) / len(per)
    print(f"{name:10} {mean('recall@1'):8.2f} {mean('recall@5'):8.2f} {mean('precision@5'):6.2f} {mean('rr'):6.2f}")
ceiling = sum(min(len(g["chunk_ids"]), 5) for g in GOLD) / (5 * len(GOLD))
print(f"highest precision@5 this set allows: {ceiling:.2f}")
(ROOT / "rag" / "eval" / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
```

```bash
uv run python -m rag.eval.run
```

*You should see* four rows where, in every row, `recall@1 ≤ MRR ≤ recall@5`. That is not a property of your pipeline, it is arithmetic: a question found at rank 1 adds 1 to all three, one found at rank 3 adds a third to MRR and 1 to recall@5. If a row breaks it, the metric code is wrong. Every `prec@5` is at or below the ceiling on the last line, which is low because most of your questions have one relevant chunk and the other four slots cannot count. Precision@5 is the wrong headline for a set like this, and the reflection asks why.

Paste the table into `rag/eval/RESULTS.md` next to your committed predictions.

**Extension — the triad against your labels (ASSIGNED)**

The triad's context-relevance judge and your labels answer the same question about the same pairs: is this retrieved chunk relevant to this question? Run the judge and count where they disagree.

**Part 1. `rag/eval/triad.py`.**

```python
# rag/eval/triad.py
import json
import re

from rag.answer import answer
from rag.chunking import load_chunks
from rag.corpus import ROOT
from rag.llm import chat

GOLD = json.loads((ROOT / "rag" / "eval" / "golden.json").read_text(encoding="utf-8"))
chunks = {c["id"]: c for c in load_chunks()}
TAIL = " Reply with a single digit 0, 1, 2 or 3 on the first line, then one sentence of reasons."
CONTEXT = "Rate how relevant the PASSAGE is to answering the QUESTION. 3 = contains the answer, 0 = unrelated." + TAIL
GROUNDED = "Rate how well every claim in the ANSWER is supported by the PASSAGES. 3 = all supported, 0 = none." + TAIL
ANSWER = "Rate how well the ANSWER addresses the QUESTION, ignoring whether it is true. 3 = fully, 0 = not at all." + TAIL

def grade(instruction, content):
    out = chat([{"role": "system", "content": instruction}, {"role": "user", "content": content}])
    m = re.search(r"[0-3]", out.splitlines()[0] if out else "")
    return int(m.group()) if m else None

rows = []
for g in GOLD:
    r = answer(g["question"])
    ids = [cid for cid, _ in r["retrieved"]]
    passages = "\n\n".join(f"[{c}] {chunks[c]['text']}" for c in ids)
    rows.append({
        "id": g["id"], "retrieved": ids, "labels": [c in g["chunk_ids"] for c in ids],
        "context": [grade(CONTEXT, f"QUESTION: {g['question']}\n\nPASSAGE: {chunks[c]['text']}") for c in ids],
        "grounded": grade(GROUNDED, f"PASSAGES:\n{passages}\n\nANSWER: {r['answer']}"),
        "answer_rel": grade(ANSWER, f"QUESTION: {g['question']}\n\nANSWER: {r['answer']}"),
        "answer": r["answer"]})
    print(g["id"], rows[-1]["labels"], rows[-1]["context"], rows[-1]["grounded"], rows[-1]["answer_rel"])

(ROOT / "rag" / "eval" / "triad.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
pairs = [(lab, (c or 0) >= 2) for row in rows for lab, c in zip(row["labels"], row["context"])]
table = {(l, j): sum(1 for p in pairs if p == (l, j)) for l in (True, False) for j in (True, False)}
print("label relevant,     judge relevant:", table[(True, True)])
print("label relevant,     judge not:     ", table[(True, False)])
print("label not relevant, judge relevant:", table[(False, True)])
print("label not relevant, judge not:     ", table[(False, False)])
print(f"agreement: {(table[(True, True)] + table[(False, False)]) / len(pairs):.0%} of {len(pairs)} pairs")
```

```bash
uv run python -m rag.eval.triad
```

*You should see* twenty lines, then a two-by-two table whose four counts add up to about 100 (twenty questions, five chunks each, fewer where the pipeline refused at the gate). Expect the largest disagreement cell to be "label not relevant, judge relevant": you labeled the chunk that answers, and the judge also credits chunks that are merely on topic.

**Part 2. `rag/eval/triad_vs_labels.md`.**

1. The four counts and the agreement rate.
2. Five disagreements, read by you: for each, the question, the chunk's first 200 characters, your label, the judge's score, and who was right. If the judge was right, your golden set gets a new chunk id, in a **separate commit** whose message says which question and why. A label changed after a judge disagreed with it is a changed label, and the history should say so.
3. The groundedness and answer-relevance scores for every question where the answer was wrong or refused. Did the judges notice?
4. **Where it got worse.** From `runs.json`, the `kind` on which `rerank` does worse than `hybrid`, with the per-kind recall@5 for both. If there is no such kind, the kind where the gap is smallest, and the question that causes it.
5. Which instrument you believe for which purpose, in three sentences, with a number in each.

```bash
git add rag/eval/metrics.py rag/eval/run.py rag/eval/runs.json rag/eval/RESULTS.md rag/eval/triad.py rag/eval/triad.json rag/eval/triad_vs_labels.md scratch/a19-video.py
git commit -m "A19: retrieval metrics on the golden set, triad vs hand labels"
git push
```

One PR, not two; push again each day. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`rag/eval/golden.json` (20 questions, committed before any run) + `rag/eval/predictions.md` + `rag/eval/find.py` + `rag/eval/check_golden.py` + `rag/eval/metrics.py` + `rag/eval/run.py` + `rag/eval/RESULTS.md` + `rag/eval/triad.py` + `rag/eval/triad_vs_labels.md` + `scratch/a19-video.py`.

**Reflection Questions**

1. Paste your committed predictions and your `run.py` table. Which prediction was furthest off, and in which direction? Name the question in `runs.json` that contributes most to that gap and paste its `ranked` list next to its `chunk_ids`.
2. Paste your highest-precision@5-allowed line and your hybrid `prec@5`. Construct, from two of your own golden questions, a case where a change to the pipeline would raise recall@5 and lower precision@5, or the reverse, and say which of the two numbers you would want to go up for the agent in A20.
3. From `triad_vs_labels.md`, take the disagreement you were least sure how to settle. Paste the question, the chunk, your label and the judge's score with its one sentence of reasons (from `data/llm_cache.json`, or by re-running `grade` on it). What rule did you use to label that chunk on Day 1, and would a classmate labeling your corpus have made the same call?
