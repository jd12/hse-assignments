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

**Lesson 3 · map your own agent.** No code. When he names the parts an agent decomposes into, pause and write the same parts for your v2 agent: which call decides what to do next, which tools it can call, and what it carries between steps. Step 3 turns this into the table under `## Step 3` in `evidence/A11.md`, with file and line numbers.

**Lab 1 · run it, then compare the tool lists.** When the lab defines its tools, copy their names and one-line descriptions into your log next to your own agent's tools. The lab's router is the model reading those descriptions and choosing one. So is yours.

**Lab 2 · write down the tree.** When the first trace opens in the Phoenix UI, click into it and write the span tree under `## Lab 2` in `evidence/A11.md`: every span's name and kind, indented under its parent, with its duration. Step 8 prints the same tree for your agent.

Two notebook failures, both in the lab: re-running the cell that registers the tracer registers it twice, and every span appears twice (restart the kernel rather than debugging it); and an empty Phoenix UI right after a run usually means the spans have not been flushed yet, not that nothing was traced. Wait, then refresh.

**Notes**

**The work is in your v2 agent repo,** the one you cloned in A01. It is yours, not from Classroom 50, so it has no secret guard until Step 1 installs one. Traces contain prompts. Install the guard before you generate anything. The evidence goes in the foundations repo, in `evidence/A11.md`, as always.

**A span is a record, not a product.** An id, the id of its parent, a start time, an end time, and some attributes. Phoenix draws them nicely; `src/trace.ts` writes them to a file, one line each, in forty lines of TypeScript. Keep the span names and kinds exactly as Step 6 gives them; A12's evals read them.

**A global "current span" breaks on parallel tool calls.** If your agent runs two tools at once, a single variable holding the current parent hands both children the wrong parent, and the tree comes out tangled with no error. `AsyncLocalStorage` keeps the parent per chain of `await`s, which is why the tracer uses it.

**Approvals hang batch runs, and refusals empty them.** A v2 agent that asks before running a shell command sits waiting at the first question of a 25-question run, so `EVAL_MODE=1` makes the approval gate auto-approve read-only tools and refuse everything else without asking. That also means `run_shell` is refused in a batch: the agent cannot `grep` your corpus with it, and `read_file` is cut at 20,000 characters, which on a one-megabyte corpus is the first one percent of the file. Step 4 gives the agent a read-only `grep_file` tool so it can find things in a file it cannot read whole.

**Your agent may not be allowed outside its own folder.** If its file tools are sandboxed to the repo, a path into your foundations repo fails. Step 2 copies the corpus into `data/`, git-ignored.

**Walkthrough — trace your own agent on your own corpus**

**Step 1. Merge, branch, log.**
Open the foundations repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. This assignment and A12 share one evidence branch in the foundations repo and one work branch in the v2 repo; both stay open through A12:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/agent-evals
ls evals evidence
```

*You should see* the nine files the template-update pull request put in `evals/` (`trace.ts`, `grepFile.ts`, `ask.ts`, `show_trace.py`, `run_all.py`, `steps.py`, `evals.py`, `judge.py`, `agreement.py`) and `evidence/A11.md` and `evidence/A12.md` among the evidence files. Then the v2 repo:

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
- [ ] Day 1: lessons 2–3, my agent's parts written down; Lab 1 run, tool lists compared
- [ ] Day 1: Steps 2–4, corpus and files copied, map filled in evidence/A11.md, grep_file registered
- [ ] Day 2: lesson 5, Lab 2 run, Phoenix tree in evidence/A11.md
- [ ] Day 2: Steps 5–8, trace.ts read, loop wrapped, one tree printed and pasted
- [ ] Extension: 25 questions and the prediction committed, then base-1 run, STEPS.md pasted
- [ ] Fill every slot in evidence/A11.md (check_evidence.py: all slots filled); push both repos, open both PRs, jd12 as collaborator on the v2 repo
```

On GitHub, add **jd12** as a collaborator on the v2 repo if I am not one already. It is your repository, so I cannot review it otherwise.

Two meetings, one branch in each repo. Push again each day; one PR per repo, not two; one log entry per meeting.

**Step 2. Bring the corpus and the files.**

```bash
cd ~/version_control/<your-v2-agent-repo>
F=~/version_control/hse-2026-2027-foundations-<your-username>
mkdir -p data evals/traces scripts src/tools
cp $F/data/corpus.txt data/
cp $F/evals/trace.ts src/trace.ts
cp $F/evals/grepFile.ts src/tools/grepFile.ts
cp $F/evals/ask.ts scripts/ask.ts
cp $F/evals/show_trace.py $F/evals/run_all.py $F/evals/steps.py $F/evals/evals.py $F/evals/judge.py $F/evals/agreement.py evals/
```

Then open `.gitignore`. `data/` is already in it, so the corpus stays out of git (it is in the foundations repo already). Delete the two lines `evals/runs/` and `evals/traces/`: this assignment and A12 commit both folders, and git would otherwise drop them from every commit without a word. Then:

```bash
git status --short
```

*You should see* `.gitignore` modified, the new files as `??` (`scripts/` shows as one folder line, since it is new), and nothing under `data/`. On Windows, run these in Git Bash, where `$F` and `~` work as written.

**Step 3. Map the agent.**

```bash
grep -rn "tool_calls\|toolCalls\|tools:" --include=*.ts . | grep -v node_modules | head -20
```

*You should see* the lines where your loop reads the model's tool calls and where the tool list is built; on the reference agent, five lines across `src/agent.ts`, `src/llm.ts`, `src/tools/index.ts` and `evals/run.ts`. Open those files and fill the table under `## Step 3` in `evidence/A11.md`: the router (the model call whose response says which tool to use next), the loop (where it decides to call the model again or stop), the approval gate, and one row per tool with its file and line. On the reference agent the router is the `callModel` line in `src/agent.ts`, the loop is the `for (let step = 0; ...)` above it, the gate is `approve()` in `src/approval.ts`, and the tools are the five in `src/tools/index.ts`. Add a row for each tool the Lab 1 agent has that yours does not, or say there is none.

**Step 4. Give the agent a grep tool.**

`src/tools/grepFile.ts` is a read-only tool: a pattern and a file path in, up to 50 matching lines with their line numbers out, the same shape as the tools in `src/tools/fs.ts`. Register it where your tool list is built. On the reference agent that is two lines in `src/tools/index.ts`: an import under the others,

```ts
import { grepFile } from "./grepFile.js";
```

and `grepFile` added to the end of the `TOOLS` list. If your agent's tool interface differs, adapt the three fields that matter: `name`, `description`, and `run`, which takes the arguments and returns a string.

```bash
npx tsc --noEmit
```

*You should see* nothing: a clean typecheck. *If it broke* with `Cannot find module './index.js'`, your tool registry is not at `src/tools/index.ts`; change the `import type { Tool }` line at the top of `grepFile.ts` to point at the file that defines your tools.

**Step 5. Read the tracer.**

Open `src/trace.ts` and read it top to bottom with its comments; it is one function. Three things to find before you wrap anything: the line that decides whether a span starts a new trace or joins its parent's (`parent?.traceId ?? randomUUID()`), the line that makes children find their parent (`current.run(...)`), and the `finally` block that writes the span when it ends, which is why children land in the file before their parent. Tool outputs over 20,000 characters are cut; A12 counts that as a finding, not a bug.

**Step 6. Wrap the router and the skills.**

Open the loop at the file and line in your map. Wrap the model call and each tool call. Your function names will differ; keep the span names and kinds exactly:

```ts
import { span } from "./trace.js";

const response = await span("llm", "LLM", { step, n_messages: messages.length },
  () => callModel(messages, tools));

const result = await span(`tool:${name}`, "TOOL", { tool: name, args },
  () => runTool({ name, args }));
```

On the reference agent, `const { message } = await callModel(messages, tools);` becomes `const { message } = await span("llm", "LLM", { step, n_messages: messages.length }, () => callModel(messages, tools));` and `result = await runTool({ name, args });` becomes `` result = await span(`tool:${name}`, "TOOL", { tool: name, args }, () => runTool({ name, args })); ``. *If your loop is inside a library call* (one function that runs several steps for you), wrap each tool's own execute function in a `TOOL` span instead, and wrap the library call in one `LLM` span. You lose one `LLM` span per step and keep everything A12 needs.

**Step 7. One question, one trace.**

`scripts/ask.ts` answers one question inside one `AGENT` span named `agent_run` and prints a JSON line. Open it: its import of `runAgent` is the name and path of your loop's entry point, and `.answer` at the end is the field of its result that holds the answer text; change either if yours differ. `PREFIX` is part of every experiment from here to the end of A12. Do not change it between runs you compare.

```bash
rm -f evals/traces/spans.jsonl
EVAL_MODE=1 npx tsx scripts/ask.ts "Who is the first person named in the file?"
wc -l evals/traces/spans.jsonl
```

`spans.jsonl` only grows: every run appends, a failed one too, which is what the `rm -f` is for; run it again before any rerun, or the count below means nothing. *You should see* possibly a `DeprecationWarning` about `punycode` (harmless; the reference agent prints nothing else in this mode), then a single JSON line with a `trace_id`, the question and the answer, and then the line count: with one tool call, `4 evals/traces/spans.jsonl` (an `LLM` span, a `TOOL` span, the final `LLM` span, then the `AGENT` span that wraps them). Paste the JSON line and the `wc -l` line under `## Step 7` in `evidence/A11.md`. *If it broke:* a count of 1 means only `agent_run` was written, so the edit in Step 6 is on a path your entry point does not take. A count of 2 with `"status": "error"` on both lines means the model call itself failed (key, network); the error is above the JSON line. A hang means your approval gate is still asking. An error about top-level `await` means your repo compiles to CommonJS; put everything in `ask.ts` after the imports inside `async function main() { ... }` and call `main()`.

**Step 8. Print the tree.**

```bash
uv run --no-project python evals/show_trace.py
```

`--no-project` tells `uv` not to look for a Python project in a TypeScript repo; the script is standard library only. *You should see* one `AGENT` line at the left, then `LLM` and `TOOL` lines indented under it, in time order, alternating: a model call, the tools it asked for, a model call, and a final `LLM` with no tool after it. On the reference agent with one grep, the shape is this (the milliseconds are yours; the two model calls take most of the time):

```text
AGENT agent_run                   .... ms
  LLM   llm                        .... ms
  TOOL  tool:grep_file               13 ms  {'pattern': 'Homer', 'path': 'data/corpus.txt'}
  LLM   llm                        .... ms
```

The `LLM` count is one more than the number of rounds of tool calls, and the root's time is at least the sum of its children's. Paste the tree under `## Step 8` in `evidence/A11.md` and fill the counts line under it.

**Extension — twenty-five questions and the shortest path (ASSIGNED)**

**X1. Write the suite, and your prediction, first.** In the v2 repo, replace the eight questions already in `evals/questions.jsonl` (they belong to `evals/run.ts`, which nothing from here on uses; `npm run eval` stops working, and that is fine) with 25 lines over your corpus:

```json
{"id": "e01", "question": "What false name does Ulysses give the Cyclops?", "answer": "Noman", "quote": "my name is Noman", "expected_tool": "grep_file", "min_steps": 1}
```

`expected_tool` is the tool in *your* agent that should be called first; for a question answered by finding a line in the corpus that is `grep_file`. `min_steps` is the fewest tool calls you think a careful agent needs. You may reuse ten of your A10 questions. Of the 25, three have no answer in the corpus (`"answer": "none"`, `"quote": ""`), three need two passages from different parts of the file (`min_steps` at least 2), and the rest range from easy to near-miss. A12 labels these runs pass or fail, and twenty-five easy questions calibrate nothing.

Then fill `## Prediction` in `evidence/A11.md`: the `wc -l` line, the no-answer and two-passage ids, and your predicted mean number of tool calls over the 25. Commit both repos before the first run:

```bash
git add src/ scripts/ evals/ .gitignore
git commit -m "A11: tracer, grep_file, 25 corpus questions with min_steps, before the first run"
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git add evidence/A11.md
git commit -m "A11: prediction before the first run"
```

I will check your commit timestamps.

**X2. Run all 25.**

```bash
cd ~/version_control/<your-v2-agent-repo>
uv run --no-project python evals/run_all.py base-1
```

*You should see* 25 lines, each an id, a trace id and the first sixty characters of the answer, and `evals/runs/base-1.jsonl` with 25 lines. It is 25 agent runs of several model calls each; a few minutes. A `None` in the trace id column is a crash; its `error` field in the file says why, and it stays in the file as a result. `ASK` at the top of `run_all.py` is `npx tsx scripts/ask.ts`; change it only if your repo runs TypeScript some other way.

**X3. Count the path.**

```bash
uv run --no-project python evals/steps.py base-1 | tee evals/STEPS.md
```

*You should see* one line per question, `e01  tools  1  min 1  excess +0`, then four lines: the mean excess over the questions that ran, how many took more than twice your `min_steps`, how many took fewer (you were wrong about the shortest path), and how many crashed. Excess is mostly zero or positive. Every agent wastes calls somewhere; the no-answer questions are usually where, because nothing tells the agent to stop looking. Paste the four lines under `## Extension X3` in `evidence/A11.md`. Then print the tree of the question with the largest excess, with its trace id from `evals/runs/base-1.jsonl`:

```bash
uv run --no-project python evals/show_trace.py <trace_id>
```

Paste it in the second fence under `## Extension X3` and say in two sentences, from the `args` column, what the extra calls were doing.

**X4. Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/A11.md` (the questions are below) and run `uv run python scripts/check_evidence.py A11` in the foundations repo until it says `all slots filled`. Then, in the v2 repo:

```bash
git add evals/ scripts/ src/
git commit -m "A11: base-1 run of 25, tool calls vs predicted minimum"
git push -u origin dev/instrumentation
```

Open the pull request in the v2 repo: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through A12 and merges once, at the end of A12. The PR body names the span you trust least. Then the foundations repo:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git add evidence/A11.md
git commit -m "A11: evidence"
git push -u origin dev/agent-evals
```

Open that pull request the same way; it also stays open through A12.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
In the v2 repo: `src/trace.ts` + `src/tools/grepFile.ts` registered + the wrapped loop + `scripts/ask.ts` + `evals/questions.jsonl` + the six Python files in `evals/` + `evals/STEPS.md` + `evals/runs/base-1.jsonl` + `evals/traces/spans.jsonl`. In the foundations repo: `evidence/A11.md` with every slot filled (the prediction committed before the first run, the Lab 2 tree, the map, the Step 7 lines, your tree with its counts, the STEPS summary and the largest-excess tree, the three reflection answers).

**Reflection Questions** (answer under `## Reflection` in `evidence/A11.md`)

1. Paste your agent's tree from Step 8. How many `LLM` spans and how many `TOOL` spans, and which single span took the most milliseconds? Put the Lab 2 tree beside it and name one span kind or level of nesting the lab has that yours does not, and whether your agent needs it.

   *How to get it:* both trees are already in the file, under `## Step 8` and `## Lab 2`; count the `LLM` and `TOOL` lines and read the milliseconds column for the biggest. The lab's tree usually has a kind yours does not (a `CHAIN` or `RETRIEVER` span, or a tool nested inside another tool); say whether anything in your agent does that job.

2. From the Step 3 map, give the file and line of your router. Paste `git diff main --stat -- src/` from the v2 repo and name the one change that was not a `span(...)` wrapper, if there was one (a tool registration, an approval flag, a path, an export), and why the trace needed it. If any span came out with the wrong parent or not at all on the first try, say which and what fixed it.

   *How to get it:* `git diff main --stat -- src/` in the v2 repo, on `dev/instrumentation`, lists every file under `src/` that changed with how many lines. On the reference agent that is `src/agent.ts` (the two wraps and the import), `src/trace.ts` and `src/tools/grepFile.ts` (new) and `src/tools/index.ts` (the registration, which is the change that was not a wrapper: without `grep_file` the batch run has nothing to trace but a cut `read_file`).

3. From `evals/STEPS.md`: your mean excess, the count over twice your `min_steps`, and the commit hash of the questions file. Name the question where you most underestimated the path, paste its tree, and say whether the extra calls were the agent's fault or a sign that your `min_steps` was wrong.

   *How to get it:* the mean and the count are the four summary lines pasted under `## Extension X3`; the hash is the commit whose message says before the first run: `git log --oneline --grep="before the first run" -- evals/questions.jsonl | tail -1` in the v2 repo. The question is the one with the largest `excess` in `STEPS.md`, and its tree is already pasted; the agent's fault looks like the same `grep_file` pattern three times in the `args` column, a wrong `min_steps` looks like three different patterns that each found something.
