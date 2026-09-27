# A18 · End-to-End RAG with Citations and a Refusal Policy

**Meetings:** D34–D35 · **Points:** 15 pts

**Watch — 15 min**
Day 1 — 15 min
[DeepLearning.AI, *Building and Evaluating Advanced RAG*](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) lesson 2, Advanced RAG Pipeline (15m)
Day 2 — no video. The refusal policy and the extension are the day.

**During the video**

A notebook lesson, run on the DeepLearning.AI page. Same rule as A17: add a cell under each of his code cells, type his code into it, run yours. Copy what you typed into `scratch/a18-video.py` in your rag repo before you close the tab.

He builds a basic pipeline over one document with LlamaIndex, then two advanced versions of it, and scores all three with an evaluation library. **Type the basic pipeline and run it.** It is five or six calls: load, index, make a query engine, ask. Every one of those calls hides a step you wrote by hand between A13 and A17, and the walkthrough asks you to name which.

**Pause on the evaluation results table** when he shows it. Write down in `scratch/a18-video.py`, as a comment, the name of each metric in the table and the numbers for the basic pipeline and the best one. A19 is those metrics, and you will want to know what a good number looked like on his data.

The two advanced pipelines (sentence-window and auto-merging) are **watch only**. Their own lessons are not assigned; it is enough to know what they change.

**Notes**

**Citations have to survive chunking, and yours already do.** Every chunk since A16 is a record with an `id` and the character positions it came from. The prompt gives the model the id at the start of every chunk and requires it at the end of every sentence. Hand it bare strings and it cites anyway, with numbers it invented. Step 5 shows you.

**A citation is only real if it resolves.** The pipeline checks every id in the answer against the ids it actually retrieved, and any that do not match are logged as `invalid`. An answer with an invalid citation is worse than one with no citation, because it looks checked.

**The refusal policy is a written rule with a number in it.** "Try not to hallucinate" is not a policy. "If the reranker's best score is below X, the pipeline says `I don't know` without calling the model; otherwise the model answers only from the passages and says `I don't know` if they do not contain it" is a policy. You pick X from your own scores in Step 4.

**The near-miss is the hard case.** A question whose answer is *almost* in your corpus: the right topic, a different detail. The retriever finds a relevant chunk, so the score gate passes, and the model has to notice the specific thing is not there. Build one deliberately.

**Walkthrough — the pipeline on your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/rerank && git pull       # skip this line if A17 has merged
git switch -c dev/rag-pipeline
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Lesson 2 Advanced RAG Pipeline, typed along in the page
- [ ] `scratch/a18-video.py` with the metrics table noted
- [ ] `rag/answer.py` answering with resolving citations (Steps 2–3)
- [ ] Refusal threshold chosen from my own scores (Step 4)
- [ ] The bare-string trap (Step 5)
- [ ] Push, PR, sign off
```

**Step 2. `rag/answer.py`.**

```python
# rag/answer.py
import json
import re
import sys
import time

from rag.corpus import ROOT
from rag.expand import TITLE
from rag.llm import chat
from rag.rerank import retrieve

REFUSE_BELOW = 0.0          # Step 4 sets this from your own numbers
LOG = ROOT / "data" / "answers.jsonl"

SYSTEM = f"""You answer questions about {TITLE} using ONLY the passages below.
Each passage starts with its id in square brackets.
End every sentence of your answer with the id of the passage it came from, like [c00412].
If the passages do not contain the answer, reply with exactly one line:
I don't know. Searched for: <the question, rephrased as a search>
Never cite an id that is not in the passages. Never use outside knowledge."""

def answer(question, k=5, cache=True):
    hits = retrieve(question, k=k)
    record = {"time": time.time(), "question": question,
              "retrieved": [(h["id"], round(h["score"], 3)) for h in hits]}
    if not hits or hits[0]["score"] < REFUSE_BELOW:
        text = f"I don't know. Searched for: {question}"
        record["gate"] = "refused before generation"
    else:
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        text = chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"{context}\n\nQuestion: {question}"}], cache=cache)
        record["gate"] = "passed"
    cited = sorted(set(re.findall(r"\[(c\d{5})\]", text)))
    retrieved_ids = {h["id"] for h in hits}
    record.update(answer=text, cited=cited, invalid=[c for c in cited if c not in retrieved_ids])
    with LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record

if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]))
    print(r["answer"])
    print("retrieved:", r["retrieved"])
    print("cited:", r["cited"], " invalid:", r["invalid"], " gate:", r["gate"])
```

That is the whole pipeline: chunk (A16), hybrid retrieve (A16), rerank (A17), gate, generate, check. `data/answers.jsonl` gets one line per question, with everything you would need to debug an answer a week from now.

**Step 3. Ask it something your corpus answers.**

```bash
uv run python -m rag.answer <a question from your queries.json, phrased as a question>
```

*You should see* an answer where every sentence ends in an id like `[c00412]`, a `retrieved:` list of five ids with scores, `cited:` a subset of those five, and `invalid: []`. On the Odyssey, the right citation for a question about the scar the old nurse recognizes is a chunk from Book XIX, where she washes his feet. Yours should be the chunk your needle is in.

Check one citation by hand: print the cited chunk and read it.

```bash
uv run python -c "
from rag.chunking import load_chunks
c = {c['id']: c for c in load_chunks()}['<a cited id>']
print(c['char_start'], c['char_end']); print(c['text'])
"
```

*If it broke:* `cited: []` with sentences that plainly came from the passages means the model used a different bracket style; paste one answer into your log and tighten the instruction in `SYSTEM`. `ModuleNotFoundError: sentence_transformers` means you branched from `main` before A17 merged.

**Step 4. Choose the refusal threshold from your own scores.**

Write three questions your corpus cannot answer, on topics it never mentions. Then:

```bash
uv run python -c "
from rag.rerank import retrieve
from rag.scoreboard import QUERIES
for q in QUERIES:
    print('in  ', q['id'], round(retrieve(q['query'], k=1)[0]['score'], 2))
for q in ['<out-of-corpus 1>', '<out-of-corpus 2>', '<out-of-corpus 3>']:
    print('out ', round(retrieve(q, k=1)[0]['score'], 2), q)
"
```

*You should see* the in-corpus top scores mostly positive and the out-of-corpus ones mostly negative, with some overlap near zero. The cross-encoder's scores are not probabilities and they are not comparable across models, so the line is yours to draw. Set `REFUSE_BELOW` in `rag/answer.py` between the two groups, and write the ten in-scores, the three out-scores and your threshold into `rag/refusal_policy.md` under `## Calibration`. If the groups overlap, say how many in-corpus queries the threshold refuses. That is the cost of the gate.

**Step 5. The bare-string trap.** Once, on purpose: in `answer`, change the context line to drop the ids,

```python
        context = "\n\n".join(h["text"] for h in hits)
```

and ask the Step 3 question again with the cache off:

```bash
uv run python -c "from rag.answer import answer; r = answer('<the Step 3 question>', cache=False); print(r['answer']); print('invalid:', r['invalid'])"
```

*You should see* citations anyway: numbers like `[1]`, or ids in the right format that do not exist. Paste the answer and its `invalid:` line into `rag/refusal_policy.md` under `## Bare strings`. Then put the ids back.

```bash
git add rag/answer.py rag/refusal_policy.md scratch/a18-video.py
git commit -m "A18: end-to-end answer with resolving citations and a score gate"
git push -u origin dev/rag-pipeline
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Sign off the log.

**Day 2 starts here.** In the rag repo: `git switch dev/rag-pipeline && git pull`. In the log repo: `bash scripts/start-entry.sh`, then:

```markdown
- [ ] Fifteen questions with expected behavior, committed before any run
- [ ] `rag/answer_eval.py` run, every answer marked by hand
- [ ] `rag/refusal_policy.md`: the rule, three transcripts, the numbers
- [ ] Push, sign off
```

**Extension — does it refuse the right things? (ASSIGNED)**

**Part 1. Fifteen questions, expected behavior first.** `rag/answer_questions.json`:

```json
[
  {"id": "a01", "question": "...", "expect": "answer", "needle": "..."},
  {"id": "a11", "question": "...", "expect": "refuse"},
  {"id": "a14", "question": "...", "expect": "refuse", "near": "what the corpus does say instead"}
]
```

Ten `answer` questions: your ten queries from `queries.json`, rephrased as questions, same needles. Three `refuse` questions: the three from Step 4. Two `refuse` near-misses: the right topic, a detail your corpus does not contain, with `near` saying what it does contain. On the Odyssey, "What color was the scar on Ulysses' leg?" is a near-miss: the scar and the boar are there, a color is not.

```bash
git add rag/answer_questions.json
git commit -m "A18: fifteen questions and expected behavior, before any run"
git push
```

I will check your commit timestamps.

**Part 2. Run them.**

```python
# rag/answer_eval.py
import json

from rag.answer import answer
from rag.chunking import load_chunks
from rag.corpus import ROOT
from rag.scoreboard import norm

QS = json.loads((ROOT / "rag" / "answer_questions.json").read_text(encoding="utf-8"))
chunks = {c["id"]: c for c in load_chunks()}

for q in QS:
    r = answer(q["question"])
    refused = r["answer"].startswith("I don't know")
    grounded = any(norm(q.get("needle", "~")) in norm(chunks[c]["text"]) for c in r["cited"])
    print(f"{q['id']} expect={q['expect']:6} got={'refuse' if refused else 'answer':6} "
          f"gate={r['gate'][:7]:7} cited={len(r['cited'])} invalid={len(r['invalid'])} "
          f"needle_cited={grounded}")
    print("   ", r["answer"][:160].replace("\n", " "))
```

```bash
uv run python -m rag.answer_eval
```

*You should see* fifteen pairs of lines. `invalid=0` on every row. Most `answer` rows with `needle_cited=True`, meaning at least one cited chunk contains the answer's needle. Your three out-of-corpus questions refused, some of them at the gate.

**Part 3. Mark every answer by hand.** Add a column to your notes: correct, wrong, or refused. `needle_cited` tells you the right chunk was cited; it does not tell you the sentence citing it is true. Read each one.

**Part 4. `rag/refusal_policy.md`.** Under `## Policy`, the rule in two sentences, with your number in it. Under `## Transcripts`, three from `data/answers.jsonl`, pasted whole: a normal answer, a clean refusal, and your worse near-miss. Under `## Numbers`:

| | count |
|---|---|
| in-corpus answered correctly | / 10 |
| in-corpus refused (false refusals) | / 10 |
| in-corpus answered wrongly | / 10 |
| out-of-corpus refused | / 3 |
| near-misses refused | / 2 |
| answers with an invalid citation | / 15 |

**Part 5. Where it got worse.** The gate refuses some questions the model would have answered well. Take your lowest-scoring in-corpus question: set `REFUSE_BELOW` far below it for one run, paste the answer the model gives, and say whether the gate was right to block it. Put the threshold back.

```bash
git add rag/answer_eval.py rag/refusal_policy.md
git commit -m "A18: refusal policy, fifteen-question run, hand marks"
git push
```

One PR, not two; push again each day. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`rag/answer.py` + `rag/answer_questions.json` (committed before the run) + `rag/answer_eval.py` + `rag/refusal_policy.md` (calibration, bare strings, policy, three transcripts, numbers) + `scratch/a18-video.py`.

**Reflection Questions**

1. Paste your `## Calibration` scores and your threshold. How many of your ten in-corpus queries does it refuse, and which out-of-corpus question came closest to passing? Paste that question's top retrieved chunk and say what in it the cross-encoder matched.
2. Paste your worse near-miss transcript: the question, the retrieved ids with scores, the gate result and the answer. Did it refuse? If it answered, underline (in markdown, with `**`) the claim it added that the cited chunk does not contain, and paste the sentence of the chunk it was leaning on.
3. Pick one line of the basic LlamaIndex pipeline you typed in `scratch/a18-video.py`. Name which of your own files and functions from A13–A17 does the same job, with file and function name, and one thing yours does that the library call hid (or the reverse).
