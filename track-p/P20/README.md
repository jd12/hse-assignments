# P20 · ★ Milestone A1, Day 2: The Delta, Including Where It Got Worse

**Meetings:** D43 · **Points:** 10 pts

**Watch — none**

No Boot.dev today. The eval runs, the report gets written, and the defense of it is the practical on D45.

**Notes**

**Sixty requests.** Twenty questions in `none` mode is twenty requests; twenty in `tool` mode is about forty, two per question when the model searches once. That is more than the free tier's day, and `rag/llm.py` caches every one, so a run that dies at request fifty finishes tomorrow where it stopped. I have put credit on the key for this week; the limit is still twenty a minute, and the script already waits.

**Routing and retrieval fail differently.** A `tool`-mode miss is one of three things: the model did not call the tool; it called it with a query that found the wrong chunks; or it got the right chunk and still answered wrong. The results file has the model's query and the ids the tool returned for every question, so each miss can be sorted into one of the three. P21 can fix the second kind. Nothing in `rag/` can fix the first.

**One run is a coin flip, so two questions get three.** The model is not deterministic and the router changes models between requests. Running the whole eval three times is a hundred and eighty requests; running two questions three times is twelve, and it is enough to show the spread the report has to admit.

**Walkthrough — the eval, the sort, the spread, the report**

**Step 1. Branch, log.**
Same branch as P19:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/milestone-a1 && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: rag.eval.grounded, both modes, the table in evidence/P20.md, P19 predictions checked
- [ ] Step 3: every tool-mode miss sorted into routing or retrieval
- [ ] Step 4: two questions, three runs each
- [ ] Extension: rag/MILESTONE.md with Where It Got Worse
- [ ] Fill every slot in evidence/P20.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. The eval.**

Open `rag/eval/grounded.py` and read `grade`: the three verdicts and the one rule for out-of-corpus questions. Then:

```bash
uv run python -m rag.eval.grounded
```

*You should see* twenty rows for `none`, twenty for `tool`, and two summary lines. It takes a few minutes; the script waits between requests. On Homer:

```text
{{LLM:P20-eval}}
```

Paste the two summary lines and the forty rows under `## Step 2` and check your P19 predictions, number by number. `rag/eval/grounded_results.json` has every answer, every tool query and the ids the tool returned; Step 3 reads it.

*If it broke* partway with `the model did not answer after several tries`, the day's limit is reached or the router is empty; run the same command later and the cache skips everything already answered. *If it broke* with the `tool` mode's `calls` column all `0`, the model never used the tool; that is a routing result, not a crash, and Step 3 is where it goes.

**Step 3. Routing against retrieval.**

For every in-corpus question the `tool` mode did not get right, sort it:

```bash
uv run python -c "
import json
rows = json.load(open('rag/eval/grounded_results.json', encoding='utf-8'))['tool']
qs = {q['id']: q for q in json.load(open('rag/eval/grounded_questions.json', encoding='utf-8'))}
for r in rows:
    q = qs[r['id']]
    if q['answer'] is None or r['verdict'] == 'correct':
        continue
    print(r['id'], r['verdict'], '| called the tool:', 'yes' if r['calls'] else 'no', '| queries:', r['queries'], '| cited labeled:', r['cited_labeled'])
"
```

For each line: no tool call is routing; a tool call whose chunks did not include a labeled one is retrieval (run the model's query through `uv run python -m rag.rerank "<its query>"` to see what it got); a tool call that returned a labeled chunk and still went wrong is generation. Fill the table under `## Step 3` with a `lost at` of `routing`, `retrieval` or `generation` per miss.

**Step 4. Spread, on two questions.**

Pick one question the `tool` mode got right and one it got wrong. Run each three times in `tool` mode with the cache bypassed, so each run is a fresh request:

```bash
uv run python -c "
from rag.agent import run
from rag.eval.grounded import grade
import json
qs = {q['id']: q for q in json.load(open('rag/eval/grounded_questions.json', encoding='utf-8'))}
for gid in ['<right id>', '<wrong id>']:
    for i in range(3):
        r = run('tool', qs[gid]['question'] + ' ' * i)       # i trailing spaces: a new cache key, the same question
        print(gid, 'run', i + 1, grade(qs[gid], r['answer']), 'calls', len(r['calls']))
"
```

The trailing spaces change the fingerprint `rag/llm.py` caches under without changing what the model reads. *You should see* six verdicts. On Homer:

```text
{{LLM:P20-spread}}
```

Paste the verdicts under `## Step 4`. If the three runs of one question disagree, that question's verdict in Step 2 was one of three possible, and the report has to say so.

**Extension — The report** *(assigned)*

`rag/MILESTONE.md`, five numbered sections, one page:

1. **The table.** Both summary lines from Step 2, pasted. Under them, your headline in one sentence with two numbers in it: `correct` for `none` against `tool` out of twenty.
2. **Routing.** From Step 3: how many `tool`-mode misses were routing, how many retrieval, how many generation, and one question from each kind with the model's query where there was one.
3. **Where it got worse.** At least two categories where `tool` did worse than `none`, or where the tool made the agent slower or wronger: an out-of-corpus question answered instead of refused, a question the model knew from memory and got wrong after searching, a question the tool returned the right chunk for and the model answered from the wrong one. For each: the count, one question pasted with its query, its ids and its answer from the results file, and the mechanism in two sentences. If `tool` beat `none` on every row, look at the `tool_ms`, at questions with two or more calls, and at answers that cite a chunk and are wrong; the regressions are there.
4. **The spread.** Step 4's six verdicts, and what they mean for the headline: a range, not a point.
5. **The number I trust least.** One number from this report, and why.

Paste the headline, the two categories and the least-trusted number under `## Extension` in `evidence/P20.md`.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P20.md` (the questions are below) and run `uv run python scripts/check_evidence.py P20` until it says `all slots filled`. Then:

```bash
git add rag/MILESTONE.md rag/eval/grounded_results.json evidence/
git commit -m "P20: Milestone A1, the grounded agent against the baseline on twenty questions, the report"
git push
```

The P19 pull request is still open and now has both days on it. The PR body is your headline sentence; edit the description on GitHub to add it, then leave it for me.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`rag/MILESTONE.md` + `rag/eval/grounded_results.json` + `evidence/P20.md` with every slot filled. The report is what you read from on D45.

**Reflection Questions** (answer under `## Reflection` in `evidence/P20.md`)

1. Paste your two summary lines. Say what the delta is in `correct`, what it is in `wrong`, and why the second matters more for a reader who will act on the answers.

   *How to get it:* a refusal costs the reader a lookup; a wrong answer costs them a false belief. A tool that turns three refusals into three correct answers and one refusal into one wrong answer has a positive delta in `correct` and a worse `wrong`, and the reader who acts on answers feels the second.

2. Paste your Step 3 table. Say how many of the `tool` mode's misses were routing (the tool not called, or called with a bad query) and how many were retrieval (called, the chunk not returned), and which of the two P21 can fix.

   *How to get it:* P21 traces the pipeline from the query onward; a question the model never sent to the pipeline is not in the trace. Routing is fixed in the prompt strings, retrieval in `rag/`.

3. Paste your Step 4 verdicts. Say what three runs told you that one run could not, and what it would take in requests to run the whole eval three times, against the free tier's daily limit.

   *How to get it:* three runs show whether a verdict is stable; one run cannot. The full eval three times is about a hundred and eighty requests, more than three days on the free tier, which is why the report shows a spread on two questions and not on twenty.
