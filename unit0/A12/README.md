# A12 · Evaluating AI Agents 07–14 + Unit 0 Capstone & Track Election

**Meetings:** D24–D26 · **Points:** 15 pts

**Watch — 75 min**

**Day 1 — 29 min**
[Evaluating AI Agents, DeepLearning.AI / Arize](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) · lesson 7, Adding router and skill evaluations (12m) · lesson 8, Lab 3: Adding router and skill evaluations (17m, notebook)
 · A full section, over 25 minutes to finish it.

**Day 2 — 14 min**
Same course · lesson 9, Adding trajectory evaluations (5m) · lesson 10, Lab 4: Adding trajectory evaluations (9m, notebook)
 · The rest of the period is capstone build: the judge.

**Day 3 — 32 min**
Same course · lesson 11, Adding structure to your evaluations (7m) · lesson 12, Lab 5: Adding structure to your evaluations (15m, notebook) · lesson 13, Improving your LLM-as-a-judge (4m) · lesson 14, Monitoring agents (6m)
 · Lesson 15, the conclusion, is one minute and optional. The rest of the period is the measured improvement and the track election.

**During the video**

The notebooks are the code-along again: run every cell in the DLAI page as he does. Then the walkthrough builds the same evaluation against your own agent, on your own 25 questions.

**Lesson 7 and Lab 3 · find the ground truth.** When the lab builds its router eval, write down where its correct answers come from and who wrote them. Yours come from `expected_tool` in `evals/questions.jsonl`, which you wrote on A11. When it builds a skill eval, write down whether a person, a rule or a model decides pass and fail.

**Lesson 9 and Lab 4 · write down the division.** When the lab scores a trajectory, pause and write exactly what it divides by what, and where the "best" path length comes from. Step 5 does the same thing with your `min_steps`.

**Lesson 11 and Lab 5 · what stays fixed.** When the lab compares two experiments, write down everything it holds the same between them. The measured improvement at the end of this assignment is worth nothing if any of those move.

**Lesson 13 · list the fixes.** Write down each technique he gives for improving a judge. Step 8 uses exactly one of them, and you name which.

**Lesson 14 · one monitor.** Write one number about your agent you would want on a dashboard in production, and which span attribute in your `spans.jsonl` it would come from.

**Notes**

**Labels before the judge. The order is not negotiable.** Write a one-line pass rule, label the 25 runs by hand, commit the labels, and only then write the judge. A judge that exists first pulls your labels toward it without your meaning it to, and the agreement number becomes a measurement of nothing. I will check your commit timestamps. Labels committed after the judge cap Evidence at 3.

**A judge that passes everything is the default failure.** If 20 of your 25 are passes, a judge that says "pass" to everything agrees with you 80% of the time and is useless. Overall agreement hides it. Report the two per-class numbers: of the runs you failed, how many did the judge pass; of the runs you passed, how many did it fail.

**The judge sees the quote.** Every question has the passage from your corpus that answers it. Without it, the judge is grading whether an answer sounds right.

**You are tuning the judge on the set you measure it on.** Twenty-five is all you have. Change the judge prompt once, not five times, and say in the report that the after-number is optimistic for that reason.

**One run is not a measurement.** Your agent samples, so the same question can go two ways. Before and after are three runs each, and the spread goes in the report.

**Spend.** One agent run is several model calls. Six runs of 25 is 150 agent runs, plus the judge on each. Check the remaining balance on your key before Day 3; it is a hard cap and it does not warn you.

**Walkthrough — router, skill and trajectory evals, a calibrated judge, one measured improvement**

**Step 1. Merge, branch, log.**
Open the foundations repo on GitHub and merge the pull request for A10 if I have approved it, and A11b's if you have not already; click **Delete branch** on each. A11's `dev/instrumentation` stays open; this work continues on it:

```bash
cd ~/version_control/<your-v2-agent-repo>
git switch dev/instrumentation && git pull
git branch --show-current     # dev/instrumentation, not main
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist. Day 1:

```markdown
- [ ] Lesson 7 and Lab 3, ground truth written down
- [ ] Re-read evals/questions.jsonl and STEPS.md before anything else
- [ ] Step 2, router and skill numbers for base-1
- [ ] Steps 3–4, pass rule, 25 hand labels, committed alone
```

Day 2: lessons 9–10 and Lab 4, Step 5 (trajectory eval), Steps 6–7 (judge and agreement). Day 3: lessons 11–14 and Lab 5, Step 8 (one judge fix), the extension, the track election, and closing the Unit 0 log.

Three meetings, one branch, the same pull request as A11. Push again each day; one log entry per meeting.

**Step 2. Router and skill, as two numbers (Day 1).**

A week of fall break is long enough that your own questions stop being obvious. Read `evals/questions.jsonl` and `evals/STEPS.md` before you write anything.

Create `evals/evals.py`:

```python
# evals/evals.py
import json, sys
from pathlib import Path

def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def norm(s):
    return " ".join(str(s).lower().split())

qs = {q["id"]: q for q in load("evals/questions.jsonl")}
tools_by_trace = {}
for s in sorted(load("evals/traces/spans.jsonl"), key=lambda s: s["start_ms"]):
    if s["kind"] == "TOOL":
        tools_by_trace.setdefault(s["trace_id"], []).append(s)

def router(run, q):
    calls = tools_by_trace.get(run["trace_id"], [])
    return bool(calls) and calls[0]["attributes"].get("tool") == q["expected_tool"]

def skill(run, q):
    if q["answer"] == "none" or not q["quote"]:
        return None
    needle = norm(q["quote"])[:50]
    return any(needle in norm(c["attributes"].get("output", ""))
               for c in tools_by_trace.get(run["trace_id"], []))

if __name__ == "__main__":
    for tag in sys.argv[1:]:
        runs = load(f"evals/runs/{tag}.jsonl")
        r = [router(x, qs[x["id"]]) for x in runs]
        k = [v for x, ok in zip(runs, r) if ok and (v := skill(x, qs[x["id"]])) is not None]
        print(f"{tag}: router {sum(r)}/{len(r)}  skill {sum(k)}/{len(k)}")
```

Router asks whether the first tool your agent reached for is the one you said it should. Skill asks, only for the runs where it was, whether any tool call actually brought back the passage that holds the answer. The no-answer questions have no passage, so they have no skill score.

```bash
uv run --no-project python evals/evals.py base-1
```

*You should see* one line with two fractions, and they will not be the same fraction. A low router number is fixed in tool descriptions and the system prompt. A low skill number is fixed in the tool itself: a search pattern that is too literal, a read that stops before the answer, an output cut at 20,000 characters. Write both numbers in `evals/JUDGE_REPORT.md` under a heading **Baseline**.

**Step 3. Write the pass rule.**

Create `evals/judge_labels.json` with the rule and nothing else yet:

```json
{"rule": "PASS if ... ; otherwise FAIL.", "labels": []}
```

One sentence, precise enough that a stranger applying it to your 25 would agree with you. It must say something about the answer and something about the path. For the no-answer questions it must say what a passing answer looks like.

**Step 4. Label the 25 by hand, and commit them alone.**

For each run in `evals/runs/base-1.jsonl`, print its tree with `show_trace.py <trace_id>`, read the answer against the quote, and add a label:

```json
{"id": "e07", "run": "base-1", "label": "fail", "why": "right name, but read the whole file three times first"}
```

```bash
git add evals/judge_labels.json evals/evals.py evals/JUDGE_REPORT.md
git commit -m "A12: pass rule and 25 hand labels, before any judge exists"
git push
```

`git log --oneline -- evals/judge_labels.json evals/judge.py` must show this commit before `judge.py` exists. I will check your commit timestamps.

*You should see*, counting your labels, some of each. If all 25 are passes or all are fails, your questions were too easy or too hard to calibrate anything, and the judge report will say nothing. Your A11 no-answer and two-passage questions are where the fails should be.

**Step 5. The trajectory eval (Day 2).**

Add to `evals/evals.py`, above `__main__`:

```python
def trajectory(run, q):
    calls = tools_by_trace.get(run["trace_id"], [])
    keys = [(c["attributes"].get("tool"), json.dumps(c["attributes"].get("args"), sort_keys=True))
            for c in calls]
    repeats = len(keys) - len(set(keys))
    convergence = min(1.0, q["min_steps"] / max(len(calls), 1))
    return len(calls), repeats, convergence
```

and extend the print in `__main__`:

```python
        t = [trajectory(x, qs[x["id"]]) for x in runs]
        print(f"   repeats {sum(x[1] for x in t)}  convergence {sum(x[2] for x in t) / len(t):.2f}")
```

`repeats` counts tool calls that exactly repeat an earlier call in the same run: same tool, same arguments. Convergence is your shortest path divided by the path it took, capped at 1: 1.00 means every run was as short as you said it could be.

*You should see* convergence somewhere below 1.00 and above zero. Now find a run you labeled **pass** whose convergence is under 0.5 or whose repeats are above zero. That is a correct answer reached by a bad path, and your final-answer evals from the spring could not see it.

**Step 6. The judge.**

```python
# evals/judge.py
import json, os, sys, urllib.request
from pathlib import Path
from evals import load, qs, tools_by_trace

MODEL = "gpt-4o-mini"   # the model from A09 and A10
RULE = json.loads(Path("evals/judge_labels.json").read_text(encoding="utf-8"))["rule"]
PROMPT = """You are grading one run of an agent that answers questions from a text file.
Pass rule: {rule}

Question: {question}
The passage from the file that answers it, or "none" if the file does not answer it: {quote}
Expected answer: {answer}
The tool calls the agent made, in order:
{calls}
The agent's final answer: {final}

Reply with JSON: {{"label": "pass" or "fail", "reason": "<one sentence>"}}"""

def judge(run):
    q = qs[run["id"]]
    calls = "\n".join(f'{i + 1}. {c["attributes"].get("tool")} {json.dumps(c["attributes"].get("args"))[:200]}'
                      for i, c in enumerate(tools_by_trace.get(run["trace_id"], []))) or "(none)"
    content = PROMPT.format(rule=RULE, question=q["question"], answer=q["answer"],
                            quote=q["quote"] if q["answer"] != "none" else "none",
                            calls=calls, final=run.get("answer"))
    body = json.dumps({"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(json.load(r)["choices"][0]["message"]["content"])

if __name__ == "__main__":
    tag = sys.argv[1]
    out = Path(f"evals/judge_out/{tag}.jsonl")
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for run in load(f"evals/runs/{tag}.jsonl"):
            v = judge(run)
            f.write(json.dumps({"id": run["id"], "run": tag, **v}) + "\n")
            print(run["id"], v["label"], v["reason"][:70])
```

The judge reads your rule and never your labels. `from evals import ...` works because Python puts the script's own folder, `evals/`, first on the path.

```bash
uv run --no-project python evals/judge.py base-1
```

*You should see* 25 lines, each `pass` or `fail` with a reason.

**Step 7. Agreement, per class.**

```python
# evals/agreement.py
import json, sys
from pathlib import Path

labels = {l["id"]: l["label"] for l in json.loads(Path("evals/judge_labels.json").read_text(encoding="utf-8"))["labels"]}
judged = {j["id"]: j["label"] for j in map(json.loads, Path(f"evals/judge_out/{sys.argv[1]}.jsonl").read_text(encoding="utf-8").splitlines())}
pairs = [(labels[i], judged[i]) for i in labels]
fails = [j for mine, j in pairs if mine == "fail"]
passes = [j for mine, j in pairs if mine == "pass"]
print(f"overall agreement {sum(a == b for a, b in pairs)}/{len(pairs)}")
print(f"you FAIL, judge PASS (false pass) {fails.count('pass')}/{len(fails)}")
print(f"you PASS, judge FAIL (false fail) {passes.count('fail')}/{len(passes)}")
print("disagree:", [i for i in labels if labels[i] != judged[i]])
```

```bash
uv run --no-project python evals/agreement.py base-1
```

*You should see* three fractions and a list of ids. Paste them into `JUDGE_REPORT.md` under **Judge v1**. Then, for every id in the disagree list, decide who was right, you or the judge, and write one line each. Some of them will be the judge. That is not a failure of the assignment; it is the assignment.

**Step 8. Fix the judge once (Day 3).**

Pick one technique from lesson 13 and apply it to `PROMPT` or `RULE`. Save the old output first so the comparison survives:

```bash
cp evals/judge_out/base-1.jsonl evals/judge_out/base-1.v1.jsonl
uv run --no-project python evals/judge.py base-1
uv run --no-project python evals/agreement.py base-1
```

Put the three fractions under **Judge v2** in the report, next to v1, and name the technique. If the false-pass rate went up, the fix made the judge more lenient; say so. Do not change your labels to match the judge. If you now think a label was wrong, say which and why, and leave it.

**Extension — one improvement, measured (ASSIGNED)**

**X1. Baseline, three runs.** You have `base-1`. Make two more, unchanged:

```bash
uv run --no-project python evals/run_all.py base-2
uv run --no-project python evals/run_all.py base-3
```

**X2. Choose one change and predict.** Choose the one change your Step 2 and Step 5 numbers point at: a tool description (router), a tool's search behaviour (skill), or a stopping instruction for the no-answer case (trajectory). One change. In `JUDGE_REPORT.md`, under **Improvement**, write what you are changing, which of your four numbers (router, skill, convergence, judge pass rate) it should move and by how much, and which one it might make worse. Commit that before you touch the agent:

```bash
git add evals/
git commit -m "A12: baseline x3, judge v2, predicted effect of one change (before the change)"
```

**X3. Make the change, then three more runs.**

```bash
git add -u && git commit -m "A12: <the change, in words>"
for i in 1 2 3; do uv run --no-project python evals/run_all.py fix-$i; done
for t in base-2 base-3 fix-1 fix-2 fix-3; do uv run --no-project python evals/judge.py $t; done
uv run --no-project python evals/evals.py base-1 base-2 base-3 fix-1 fix-2 fix-3
```

The spans file keeps growing; every eval looks runs up by trace id, so old runs stay readable. Do not change `PREFIX`, the questions, the model, the judge or the rule between the first `base` run and the last `fix` run.

**X4. Report.** In `JUDGE_REPORT.md`:

| run | router /25 | skill | repeats | convergence | judge pass /25 |
|---|---|---|---|---|---|

Six rows, then the mean and the min–max for base and for fix. The judge-pass column is from judge v2. Then answer, in a short paragraph: is the difference between the means bigger than the spread inside each group? Which number got worse? If the change made things worse, the report says so, and a clean negative result gets full credit.

**X5. Commit, push.**

```bash
git add evals/ && git commit -m "A12: before/after x3 each, one change, judge v2"
git push
```

The v2 pull request from A11 is already open; these commits land on it.

**Step 9. The track election.**

It goes in the foundations repo, because it is course business rather than agent work:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/track-election
```

Create `TRACK_ELECTION.md`. Rank both, 1 and 2; you may not get your first choice.

**Track A, applied.** Production AI engineering: retrieval, memory, the Model Context Protocol, durable execution, orchestration, deployment. Your corpus becomes the thing your agent retrieves from.

**Track B, theory.** LLM internals: linear algebra, autograd from scratch, attention derived rather than described, and a GPT built from an empty file. Your corpus becomes what that GPT trains on.

One paragraph, based on evidence from these eight weeks rather than on which sounds better. Name the assignment from A01 through A12 you lost track of time on, with the file in your repo that shows it, and one thing about the track you did not rank first that you will miss.

```bash
git add TRACK_ELECTION.md
git commit -m "A12: track election"
git push -u origin dev/track-election
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop.

**Step 10. Close the Unit 0 log.**

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

Then, on GitHub, merge the `<your-username>-unit0` pull request and delete the branch. Eight weeks of entries go in at once. Your track branch opens at A13 or B13.

Both the v2 pull request and `dev/track-election` merge when the work is finished and I have approved it.

**Deliverable**
In the v2 repo: `evals/evals.py` (router, skill, trajectory) + `evals/judge_labels.json` (committed before `judge.py`) + `evals/judge.py` + `evals/agreement.py` + `evals/JUDGE_REPORT.md` (baseline, judge v1 and v2 per class, improvement table over 3 + 3 runs) + `evals/runs/` + `evals/judge_out/`. In the foundations repo: `TRACK_ELECTION.md`.

**Reflection Questions**

1. Paste `git log --oneline -- evals/judge_labels.json evals/judge.py`. Then give the three fractions from `agreement.py` for judge v1 and v2, and say which class is worse and whether your one fix helped it or only helped overall agreement.

2. Pick one id where you and the judge disagreed. Paste the question, your label and `why`, and the judge's `reason`. Who was right? If the judge was, quote the part of your pass rule that did not cover the case. Then give one run you labeled **pass** that `evals.py` scores badly on repeats or convergence, and say what the agent did along the way.

3. From your improvement table: the base mean and range and the fix mean and range for the number you predicted would move, next to the prediction in the commit before the change (give its hash). Is the difference bigger than the spread? Name the number that got worse, and the entry in your A10 catalog that this change addresses or does not touch.
