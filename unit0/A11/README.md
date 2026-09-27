# A11 · Evaluating AI Agents 02–06: Build and Trace Your Own Agent

**Meetings:** D22–D23 · **Points:** 15 pts


**Watch — 49 min**

**Day 1 — 29 min**
[Evaluating AI Agents, DeepLearning.AI / Arize](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) · lesson 2, Evaluation in the time of LLMs (7m) · lesson 3, Decomposing agents (6m) · lesson 4, Lab 1: Building your agent (16m, notebook)
 · Your v2 agent repo open in VS Code beside the browser.

**Day 2 — 20 min**
Same course · lesson 5, Tracing agents (4m) · lesson 6, Lab 2: Tracing your agent (16m, notebook)

**During the video**

The notebooks are the code-along. They run in the DLAI page with everything installed; run every cell, in order, as he does. You do not install Phoenix or any of the lab's packages on your machine.

**Lesson 2 · write, do not type.** No code. When he lists what makes an LLM's output hard to test compared with a function's, write each item in your log, and beside each, one behavior of your v2 agent that has that problem.

**Lesson 3 · map your own agent.** No code. When he names the parts an agent decomposes into, pause and write the same parts for your v2 agent: which call decides what to do next, which tools it can call, and what it carries between steps. You turn this into `evals/AGENT_MAP.md` in Step 3, with file and line numbers.

**Lab 1 · run it, then compare the tool lists.** When the lab defines its tools, copy their names and one-line descriptions into your log next to your own agent's tools. The lab's router is the model reading those descriptions and choosing one. So is yours.

**Lab 2 · write down the tree.** When the first trace opens in the Phoenix UI, click into it and write the span tree in your log: every span's name and kind, indented under its parent, with its duration. Step 7 prints the same tree for your agent, and `evals/TRACE.md` holds both.

Two notebook failures, both in the lab: re-running the cell that registers the tracer registers it twice, and every span appears twice (restart the kernel rather than debugging it); and an empty Phoenix UI right after a run usually means the spans have not been flushed yet, not that nothing was traced. Wait, then refresh.

**Notes**

**The work is in your v2 agent repo,** the one you cloned in A01. It is yours, not from Classroom 50, so it has no secret guard until Step 1 installs one. Traces contain prompts. Install the guard before you generate anything.

**A span is a record, not a product.** An id, the id of its parent, a start time, an end time, and some attributes. Phoenix draws them nicely; `src/trace.ts` in Step 4 writes them to a file, one line each, in about forty lines of TypeScript. Keep the span names and kinds exactly as given; A12's evals read them.

**A global "current span" breaks on parallel tool calls.** If your agent runs two tools at once, a single variable holding the current parent hands both children the wrong parent, and the tree comes out tangled with no error. `AsyncLocalStorage` keeps the parent per chain of `await`s, which is why the tracer uses it.

**Approvals hang batch runs.** A v2 agent that asks before running a shell command will sit waiting at the first question of a 25-question run. Step 6 sets `EVAL_MODE=1`; make your approval gate auto-approve read-only tools when it is set, and nothing else.

**Your agent may not be allowed outside its own folder.** If its file tools are sandboxed to the repo, a path into your foundations repo fails. Step 2 copies the corpus into `data/`, git-ignored.

**Walkthrough — trace your own agent on your own corpus**

**Step 1. Merge, branch, log.**
Open the foundations repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Today's work is in the v2 repo:

```bash
cd ~/version_control/<your-v2-agent-repo>
git switch main && git pull
git switch -c dev/instrumentation
bash ~/version_control/hse-2026-2027-foundations-<your-username>/scripts/install_secret_guard.sh
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: lessons 2–3, my agent's parts written down
- [ ] Day 1: Lab 1 run, tool lists compared
- [ ] Day 1: Steps 2–3, corpus copied, AGENT_MAP.md with file:line
- [ ] Day 2: lesson 5, Lab 2 run, Phoenix tree written down
- [ ] Day 2: Steps 4–7, trace.ts, loop wrapped, one tree printed
- [ ] Extension: 25 questions and min_steps committed, then run
- [ ] Push and open the PR
```

On GitHub, add **jd12** as a collaborator on the v2 repo if I am not one already. It is your repository, so I cannot review it otherwise.

Two meetings, one branch, and it stays open through A12. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Bring the corpus.**

```bash
mkdir -p data evals/traces scripts
cp ~/version_control/hse-2026-2027-foundations-<your-username>/data/corpus.txt data/
printf 'data/\n' >> .gitignore
git status --short
```

*You should see* `.gitignore` modified and nothing under `data/`.

**Step 3. Map the agent.**

```bash
grep -rn "tool_calls\|toolCalls\|tools:" --include=*.ts . | grep -v node_modules | head -20
```

Create `evals/AGENT_MAP.md` with this table, filled in from your code:

| Part | Where (file:line) | What it does in my agent |
|---|---|---|
| Router | | The model call whose response says which tool to use next |
| Loop | | Where it decides to call the model again or stop |
| Skill: each tool | | One row per tool |
| Approval gate | | Where it asks before running something |

*You should see* one row per tool, matching the list you wrote during Lab 1.

**Step 4. The tracer.**

Create `src/trace.ts`. Type it.

```ts
// src/trace.ts
import { appendFileSync, mkdirSync } from "node:fs";
import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";

type Kind = "AGENT" | "LLM" | "TOOL";
const FILE = "evals/traces/spans.jsonl";
const MAX = 20_000;
const current = new AsyncLocalStorage<{ traceId: string; spanId: string }>();

export const currentTraceId = () => current.getStore()?.traceId;

export async function span<T>(name: string, kind: Kind,
    attributes: Record<string, unknown>, fn: () => Promise<T>): Promise<T> {
  const parent = current.getStore();
  const s = {
    trace_id: parent?.traceId ?? randomUUID(),
    span_id: randomUUID(),
    parent_id: parent?.spanId ?? null,
    name, kind, attributes: { ...attributes } as Record<string, unknown>,
    start_ms: Date.now(), end_ms: 0, status: "ok",
  };
  try {
    const out = await current.run({ traceId: s.trace_id, spanId: s.span_id }, fn);
    s.attributes.output = String(typeof out === "string" ? out : JSON.stringify(out)).slice(0, MAX);
    return out;
  } catch (e) {
    s.status = "error";
    s.attributes.error = String(e);
    throw e;
  } finally {
    s.end_ms = Date.now();
    mkdirSync("evals/traces", { recursive: true });
    appendFileSync(FILE, JSON.stringify(s) + "\n");
  }
}
```

A span with no parent starts a new trace. Every span started inside `fn` finds its parent through `current`. A span is written when it ends, so children land in the file before their parent. Tool outputs over 20,000 characters are cut; A12 counts that as a finding, not a bug.

**Step 5. Wrap the router and the skills.**

Open the loop at the file and line in your map. Wrap the model call and each tool call. Your function names will differ; keep the span names and kinds:

```ts
import { span } from "./trace";

const response = await span("llm", "LLM", { step, n_messages: messages.length },
  () => callModel(messages));

const result = await span(`tool:${call.name}`, "TOOL", { tool: call.name, args: call.args },
  () => runTool(call));
```

*If your loop is inside a library call* (one function that runs several steps for you), wrap each tool's own execute function in a `TOOL` span instead, and wrap the library call in one `LLM` span. You lose one `LLM` span per step and keep everything A12 needs.

**Step 6. One entry point that answers and exits.**

```ts
// scripts/ask.ts
import { span, currentTraceId } from "../src/trace";
import { runAgent } from "../src/agent";   // your loop's entry point; the name is yours

const PREFIX = "Answer from the file data/corpus.txt. ";
const question = process.argv.slice(2).join(" ");
let traceId = "";
const answer = await span("agent_run", "AGENT", { question }, async () => {
  traceId = currentTraceId()!;
  return runAgent(PREFIX + question);
});
console.log(JSON.stringify({ trace_id: traceId, question, answer }));
```

`PREFIX` is part of every experiment from here to the end of A12. Do not change it between runs you compare. Run it the way your repo runs TypeScript:

```bash
EVAL_MODE=1 npx tsx scripts/ask.ts "Who is the first person named in the file?"
```

*You should see* a single JSON line last, with a `trace_id`, and new lines at the end of `evals/traces/spans.jsonl`.

*If it broke:* no new lines in `spans.jsonl` means `span` never ran, so the edit in Step 5 is on a path your entry point does not take. A hang means your approval gate is still asking. An error about top-level `await` means your repo compiles to CommonJS; put everything after the imports inside `async function main() { ... }` and call `main()`.

**Step 7. Print the tree.**

```python
# evals/show_trace.py
import json, sys
from pathlib import Path

spans = [json.loads(l) for l in Path("evals/traces/spans.jsonl").read_text(encoding="utf-8").splitlines()]
tid = sys.argv[1] if len(sys.argv) > 1 else spans[-1]["trace_id"]
kids = {}
for s in spans:
    if s["trace_id"] == tid:
        kids.setdefault(s["parent_id"], []).append(s)

def show(s, depth=0):
    ms = s["end_ms"] - s["start_ms"]
    print(f'{"  " * depth}{s["kind"]:<5} {s["name"]:<24} {ms:>6} ms  {str(s["attributes"].get("args", ""))[:60]}')
    for c in sorted(kids.get(s["span_id"], []), key=lambda c: c["start_ms"]):
        show(c, depth + 1)

for root in kids.get(None, []):
    show(root)
```

```bash
uv run --no-project python evals/show_trace.py
```

It is standard library only; `--no-project` tells `uv` not to look for a Python project in a TypeScript repo.

*You should see* one `AGENT` line at the left, then `LLM` and `TOOL` lines indented under it, in time order, alternating: a model call, the tools it asked for, a model call, and a final `LLM` with no tool after it. The `LLM` count is one more than the number of rounds of tool calls. The root's time is at least the sum of its children's.

Create `evals/TRACE.md` with the Lab 2 tree from your log at the top and your agent's tree under it.

**Extension — twenty-five questions and the shortest path (ASSIGNED)**

**X1. Write the suite, and your prediction, first.** Create `evals/questions.jsonl`, 25 lines, over your corpus:

```json
{"id": "e01", "question": "What false name does Ulysses give the Cyclops?", "answer": "Noman", "quote": "my name is Noman", "expected_tool": "grep", "min_steps": 1}
```

`expected_tool` is the tool in *your* agent that should be called first. `min_steps` is the fewest tool calls you think a careful agent needs. You may reuse ten of your A10 questions. Of the 25, three have no answer in the corpus (`"answer": "none"`), three need two passages from different parts of the file, and the rest range from easy to near-miss. A12 labels these runs pass or fail, and twenty-five easy questions calibrate nothing.

```bash
git add src/trace.ts scripts/ask.ts evals/AGENT_MAP.md evals/show_trace.py evals/TRACE.md evals/questions.jsonl .gitignore
git add -u
git commit -m "A11: tracer, 25 corpus questions with min_steps, before the first run"
```

I will check your commit timestamps.

**X2. Run all 25.**

```python
# evals/run_all.py
import json, os, subprocess, sys
from pathlib import Path

ASK = ["npx", "tsx", "scripts/ask.ts"]          # however your repo runs a TypeScript file
tag = sys.argv[1]
out = Path(f"evals/runs/{tag}.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for line in Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        r = subprocess.run(ASK + [q["question"]], capture_output=True, text=True,
                           timeout=300, env={**os.environ, "EVAL_MODE": "1"})
        try:
            rec = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            rec = {"trace_id": None, "answer": None, "error": r.stderr[-300:]}
        f.write(json.dumps({"id": q["id"], "run": tag, **rec}) + "\n")
        print(q["id"], rec.get("trace_id"), str(rec.get("answer"))[:60])
```

```bash
uv run --no-project python evals/run_all.py base-1
```

*You should see* 25 lines, each with a trace id. A `None` is a crash; its `error` field says why, and it stays in the file as a result.

**X3. Count the path.**

```python
# evals/steps.py
import json, sys
from collections import Counter
from pathlib import Path

spans = [json.loads(l) for l in Path("evals/traces/spans.jsonl").read_text(encoding="utf-8").splitlines()]
tools = Counter(s["trace_id"] for s in spans if s["kind"] == "TOOL")
qs = {q["id"]: q for q in map(json.loads, Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines())}
for line in Path(f"evals/runs/{sys.argv[1]}.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    n, m = tools[r["trace_id"]], qs[r["id"]]["min_steps"]
    print(f'{r["id"]}  tools {n:>2}  min {m}  excess {n - m:+d}')
```

```bash
uv run --no-project python evals/steps.py base-1 | tee evals/STEPS.md
```

At the bottom of `evals/STEPS.md` write four numbers: the mean excess over your 25, how many questions took more than twice your `min_steps`, how many took fewer (you were wrong about the shortest path), and how many crashed. Then print the tree of the question with the largest excess, paste it, and say in two sentences what the extra calls were doing.

*You should see* excess mostly zero or positive. Every agent wastes calls somewhere; the no-answer questions are usually where, because nothing tells the agent to stop looking.

**X4. Commit, push, PR, sign off.**

```bash
git add evals/ scripts/ src/
git commit -m "A11: base-1 run of 25, tool calls vs predicted minimum"
git push -u origin dev/instrumentation
```

Open the pull request in the v2 repo: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through A12 and merges once, at the end of A12. The PR body names the span you trust least.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
In the v2 repo: `src/trace.ts` + the wrapped loop + `scripts/ask.ts` + `evals/AGENT_MAP.md` + `evals/TRACE.md` (Lab 2's tree and yours) + `evals/questions.jsonl` + `evals/run_all.py` + `evals/steps.py` + `evals/STEPS.md` + `evals/runs/base-1.jsonl` + `evals/traces/spans.jsonl`.

**Reflection Questions**

1. Paste your agent's tree from `evals/TRACE.md`. How many `LLM` spans and how many `TOOL` spans, and which single span took the most milliseconds? Put the Lab 2 tree beside it and name one span kind or level of nesting the lab has that yours does not, and whether your agent needs it.

2. From `evals/AGENT_MAP.md`, give the file and line of your router. Paste `git diff main --stat -- src/` and name the one change that was not a `span(...)` wrapper, if there was one (an approval flag, a path, an export), and why the trace needed it. If any span came out with the wrong parent or not at all on the first try, say which and what fixed it.

3. From `evals/STEPS.md`: your mean excess, the count over twice your `min_steps`, and the commit hash of the questions file. Name the question where you most underestimated the path, paste its tree, and say whether the extra calls were the agent's fault or a sign that your `min_steps` was wrong.
