# A20 · ★ MILESTONE A1: Grounded Agent with a Measured Delta

**Meetings:** D38–D39 · **Points:** 15 pts

**Watch**
Day 1 — no video. Milestone build.
Day 2 — no video. Milestone build.

**During the video**

No video. Before you write any code, open your v2 agent repo and find the web search tool: its name, its description string, the function that runs it, and the place it is registered with the model. Paste all four, with file and line numbers, into `scratch/a20-web-tool.md` in your rag repo. Step 3 copies that shape exactly, and Step 4 changes that function.

**Notes**

★ This is Milestone A1, the first of two gates in Track A. Everything after it assumes your agent can retrieve.

The job: give your v2 agent a `search_knowledge_base` tool backed by your A18 retriever, next to the web search tool it already has. Then prove with numbers whether it got better, **and where it got worse.** There is no version of this where adding retrieval improves everything. Local retrieval should beat web search on your corpus and lose on anything recent, anything outside it, and anything that needs breadth. A report with only wins has not been measured, and it comes back.

**You are crossing a language boundary.** The agent is TypeScript; the retriever is Python. Pick one and commit: expose the Python retriever behind a tiny local HTTP endpoint that the TypeScript tool calls, or port retrieval to TypeScript. The walkthrough builds the endpoint. A port is allowed if it returns the same top-5 chunk ids as `rag.rerank.retrieve` on all twenty golden questions, and you show that check. Either way, write down which you chose and why; the choice comes back when you meet MCP.

**The model picks between the two tools using only the strings you wrote.** A vague description and it picks wrong, and you will blame the retriever for a routing problem. The eval measures routing separately from retrieval so you can tell which one failed.

**Environment variables are read when a module loads.** `KB_ENABLED` decides whether the new tool is registered. Set it on the command line, before the process starts. Setting `process.env.KB_ENABLED` inside the eval script does nothing, because the agent module was imported first.

**Run each configuration three times.** The agent's model is not deterministic, so one run of each is a coin flip. The rubric's top Evidence score needs the spread, and the report shows the lowest and highest of three runs next to every mean.

**Walkthrough — the grounded agent**

**Step 1. Merge, branch, log.**
Open both repos on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. This milestone has a branch in each repo and a pull request in each.

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/retrieval-eval && git pull   # skip this line if A19 has merged
git switch -c dev/kb-server
```

```bash
cd ~/version_control/<your v2 agent repo>
git switch main && git pull
git switch -c dev/grounded-retrieval
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] `scratch/a20-web-tool.md`: name, description, function, registration
- [ ] `rag/serve.py` answering on 127.0.0.1:8765 (Step 2)
- [ ] `search_knowledge_base` registered next to web search (Step 3)
- [ ] Web search output as one block per result, both tools recording calls (Step 4)
- [ ] One question through each configuration by hand (Step 5)
- [ ] Thirty questions written and committed before any run (Step 6)
- [ ] Push both branches, open both PRs, sign off
```

**Step 2. `rag/serve.py`: the retriever behind a local endpoint.**

```python
# rag/serve.py
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from rag.rerank import retrieve

PORT = 8765

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/search":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        t0 = time.perf_counter()
        hits = retrieve(body["query"], k=int(body.get("k", 5)))
        ms = (time.perf_counter() - t0) * 1000
        payload = {"ms": round(ms), "results": [
            {"id": h["id"], "score": round(float(h["score"]), 4), "text": h["text"]} for h in hits]}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

if __name__ == "__main__":
    retrieve("warm up")
    print(f"knowledge base on http://127.0.0.1:{PORT}/search")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
```

The server loads the chunks, the vectors and the cross-encoder once, at start, and keeps them in memory. That is the argument for a server over starting Python on every tool call: the cross-encoder alone takes seconds to load. `127.0.0.1` means only your own laptop can reach it.

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
uv run python -m rag.serve
```

Leave it running in its own terminal. In another:

```bash
curl -s -X POST http://127.0.0.1:8765/search -d '{"query": "<golden question g01>", "k": 5}' | head -c 400   # pick a question with no apostrophe in it; one would end the quoted string early
uv run python -c "from rag.rerank import retrieve; print([h['id'] for h in retrieve('<golden question g01>')])"
```

*You should see* JSON with `ms` and five results, and the five ids in the JSON in the same order as the ids the second line prints. The same function answers both, so any difference means the server is running old code; restart it.

**Step 3. The tool, in your v2 agent repo.**

A call recorder first, so the eval can see which tools ran without depending on how your agent reports them:

```ts
// src/tools/calls.ts
export type Call = { tool: string; query: string; output: string; ms: number };
export const CALLS: Call[] = [];
```

Then the tool:

```ts
// src/tools/knowledgeBase.ts
import { CALLS } from "./calls";

const KB_URL = process.env.KB_URL ?? "http://127.0.0.1:8765/search";

export const KB_DESCRIPTION =
  "Search <your corpus title> for passages. Use this for any question about <what your corpus " +
  "covers>. Returns up to five passages, each starting with an id in square brackets; cite those ids. " +
  "Do NOT use this for recent events, facts from after <your corpus's date>, or anything not in <title>.";

export async function searchKnowledgeBase(query: string, k = 5): Promise<string> {
  const t0 = Date.now();
  let output: string;
  try {
    const res = await fetch(KB_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { results } = (await res.json()) as { results: { id: string; text: string }[] };
    output = results.map((r) => `[${r.id}] ${r.text.replace(/\s+/g, " ")}`).join("\n\n");
  } catch (err) {
    output = `knowledge base unavailable: ${(err as Error).message}`;
  }
  CALLS.push({ tool: "search_knowledge_base", query, output, ms: Date.now() - t0 });
  return output;
}
```

Fill in the three `<...>` in `KB_DESCRIPTION` for your corpus. The "Do NOT" sentence is the one that steers out-of-corpus questions to web search. Each result becomes one line, and results are separated by a blank line, which is how the scorer in Step 8 finds the rank.

Now register it, in the file you pasted into `scratch/a20-web-tool.md`, next to the web search tool and in exactly the same shape: name `search_knowledge_base`, description `KB_DESCRIPTION`, one required string parameter `query`, and a handler that returns `await searchKnowledgeBase(args.query)`. Wrap the registration in `if (process.env.KB_ENABLED !== "0")`, so `KB_ENABLED=0` gives you back the old agent exactly.

*If it broke:* `knowledge base unavailable: fetch failed` is the server not running, or running on another port. The agent gets that string back as the tool result instead of crashing, which is deliberate: Step 5 checks it.

**Step 4. Make web search comparable.** In the web search function, change its return value so each result is one line (title, URL and snippet, whitespace collapsed), results separated by a blank line, top result first, and push a record before returning:

```ts
  CALLS.push({ tool: "<your web tool's name>", query, output, ms: Date.now() - t0 });
```

Do this before you measure anything, so the baseline and the new agent both use the same output format and the comparison is fair.

**Step 5. One question, both ways, by hand.** Using however your v2 agent takes a question from the command line:

```bash
KB_ENABLED=0 <run your agent> "<golden question g01>"
KB_ENABLED=1 <run your agent> "<golden question g01>"
```

*You should see* the first answer from web search only, and the second call `search_knowledge_base` and cite a chunk id. Then stop the server (Ctrl-C in its terminal) and run the second line again. *You should see* the tool return `knowledge base unavailable`, and the agent either fall back to web search or say it could not search. Write which one it did into `scratch/a20-web-tool.md`. Restart the server.

```bash
git add rag/serve.py scratch/a20-web-tool.md
git commit -m "A20: retriever behind a local HTTP endpoint"
git push -u origin dev/kb-server
```

```bash
cd ~/version_control/<your v2 agent repo>
git add src/tools/calls.ts src/tools/knowledgeBase.ts <the registration file> <the web tool file>
git commit -m "A20: search_knowledge_base tool next to web search"
git push -u origin dev/grounded-retrieval
```

Open a pull request in each repo, **jd12** as reviewer on both.

**Step 6. Thirty questions, committed before any run.** `evals/grounded/questions.json` in the v2 agent repo:

```json
[
  {"id": "g01", "question": "...", "answer": "...", "category": "paraphrase", "expect_tool": "kb"},
  {"id": "o01", "question": "...", "answer": "...", "category": "outside", "expect_tool": "web"},
  {"id": "b01", "question": "...", "answer": "...", "category": "both", "expect_tool": "both"}
]
```

| How many | `category` | What it is |
|---|---|---|
| 20 | your golden `kind` | your A19 golden questions, copied with their `answer` strings; `expect_tool` is `kb` |
| 5 | `outside` | questions your corpus cannot answer and the web can: recent, or about the world outside it. Look up the answer yourself and write the `answer` string from the source. |
| 5 | `both` | questions that need your corpus and the web together. On the Odyssey: "In the book where Ulysses blinds the Cyclops, what does he tell him his name is, and which 2000 film retold the whole poem in 1930s Mississippi?" |

In `evals/grounded/predictions.md`, write the golden-set `correct` you expect in each mode, and the category you expect `both` to do worst on. Numbers, not words.

```bash
git add evals/grounded/questions.json evals/grounded/predictions.md
git commit -m "A20: thirty questions and predictions, before any run"
git push
```

I will check your commit timestamps. Sign off the log. **Day 1 ends here.**

**Day 2 starts here.** `git switch` to the milestone branch in both repos and `git pull`. Start the server. In the log repo: `bash scripts/start-entry.sh`, then:

```markdown
- [ ] `evals/grounded/run.ts`, six runs (Step 7)
- [ ] `evals/grounded/score.ts`, the table (Step 8)
- [ ] `evals/grounded/REPORT.md` with "Where It Got Worse"
- [ ] Push both branches, sign off
```

**Step 7. `evals/grounded/run.ts`.**

```ts
// evals/grounded/run.ts
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { CALLS } from "../../src/tools/calls";
import { runAgent } from "../../src/agent"; // CHANGE: the function your A12 trajectory eval calls

async function main() {
  const [mode, run] = process.argv.slice(2);
  if ((mode === "web") !== (process.env.KB_ENABLED === "0")) {
    throw new Error("run web mode with KB_ENABLED=0 and both mode with KB_ENABLED=1");
  }
  const questions = JSON.parse(readFileSync("evals/grounded/questions.json", "utf8"));
  mkdirSync("evals/grounded/runs", { recursive: true });
  for (const q of questions) {
    CALLS.length = 0;
    const t0 = Date.now();
    const answer = String(await runAgent(q.question));
    const row = { id: q.id, answer, calls: [...CALLS], ms: Date.now() - t0 };
    appendFileSync(`evals/grounded/runs/${mode}-${run}.jsonl`, JSON.stringify(row) + "\n");
    console.log(q.id, row.calls.map((c) => c.tool).join(" > ") || "(no tool)", `${row.ms} ms`);
  }
}

main();
```

Change the one marked import to the function your A12 eval called; if it returns an object rather than a string, pass `String(...)` the field holding the final answer. Then six runs:

```bash
for n in 1 2 3; do
  KB_ENABLED=0 npx tsx evals/grounded/run.ts web $n
  KB_ENABLED=1 npx tsx evals/grounded/run.ts both $n
done
```

If your repo runs TypeScript some other way than `npx tsx`, use that. *You should see* thirty lines per run, each naming the tools called in order. `web` runs never show `search_knowledge_base`.

*If it broke:* a file that already has rows from a crashed run gets appended to. Delete that one `.jsonl` and re-run it; never edit rows by hand.

**Step 8. `evals/grounded/score.ts`.**

```ts
// evals/grounded/score.ts
import { readdirSync, readFileSync } from "node:fs";

type Q = { id: string; answer: string; category: string; expect_tool: string };
type Row = { id: string; answer: string; calls: { tool: string; output: string }[] };
const TOOL: Record<string, string> = { search_knowledge_base: "kb", "<your web tool's name>": "web" };
const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ");
const qs: Q[] = JSON.parse(readFileSync("evals/grounded/questions.json", "utf8"));

function metrics(q: Q, row: Row): number[] {
  const blocks = row.calls[0] ? row.calls[0].output.split("\n\n").slice(0, 5) : [];
  const rank = blocks.findIndex((b) => norm(b).includes(norm(q.answer))) + 1;
  const used = [...new Set(row.calls.map((c) => TOOL[c.tool] ?? c.tool))].sort().join("+") || "none";
  const expected = q.expect_tool === "both" ? "kb+web" : q.expect_tool;
  return [rank > 0 ? 1 : 0, rank > 0 ? 1 / rank : 0, used === expected ? 1 : 0,
          norm(row.answer).includes(norm(q.answer)) ? 1 : 0];
}

for (const mode of ["web", "both"]) {
  const runs = readdirSync("evals/grounded/runs").filter((f) => f.startsWith(`${mode}-`)).sort()
    .map((f) => readFileSync(`evals/grounded/runs/${f}`, "utf8").trim().split("\n").map((l) => JSON.parse(l) as Row));
  console.log(`\n${mode}: ${runs.length} runs, mean (lowest-highest)`);
  console.log("category       n  recall@5          MRR               routed            correct");
  for (const cat of [...new Set(qs.map((q) => q.category))]) {
    const group = qs.filter((q) => q.category === cat);
    const perRun = runs.map((rows) => [0, 1, 2, 3].map((i) =>
      group.reduce((s, q) => s + metrics(q, rows.find((r) => r.id === q.id)!)[i], 0) / group.length));
    const cell = (i: number) => {
      const v = perRun.map((r) => r[i]);
      const mean = v.reduce((a, b) => a + b, 0) / v.length;
      return `${mean.toFixed(2)} (${Math.min(...v).toFixed(2)}-${Math.max(...v).toFixed(2)})`.padEnd(18);
    };
    console.log(cat.padEnd(13), String(group.length).padStart(2), cell(0), cell(1), cell(2), cell(3));
  }
}
```

```bash
npx tsx evals/grounded/score.ts
```

Retrieval is scored on the first search the agent made, whichever tool it chose, because that is what the agent actually had to work with: `recall@5` is whether the answer string appears in one of its first five results, `MRR` is one over that rank. `routed` is whether the set of tools it used matches `expect_tool`; in `web` mode it can only be right for `outside` questions and means nothing elsewhere. `correct` is whether the final answer contains the answer string.

*You should see* two tables, six categories each, with `n` of 4 to 6 per category, and a lowest-highest range after every mean. If `score.ts` crashes with `Cannot read properties of undefined`, one runs file is short: a run crashed partway, and its file has fewer than 30 lines. For your golden categories, recall@5 and correct higher in `both` than `web`. For `outside`, `both` equal to or lower than `web`. If a category shows `0.00 (0.00-0.00)` for recall@5 in `both`, open one row of that run and read `calls[0].output` before you believe it. If your corpus is public, web search may find some of your golden answers too, and that is a real result, not a bug.

**Extension — the report (ASSIGNED)**

`evals/grounded/REPORT.md` in the v2 agent repo. Five sections, in this order.

**1. Mechanism.** HTTP endpoint or port, why, and the cost in milliseconds: the median `ms` of your `search_knowledge_base` calls from the `both` runs, next to the median of your web search calls.

**2. The table.** Both tables from `score.ts`, pasted whole. Under them, your headline in one sentence with two numbers in it: golden-set `correct` for `web` against `both`, with ranges.

**3. Routing.** For `both` mode, count the questions where the agent picked the wrong tool set on at least two of three runs. For each one, say whether the retriever would have found the answer if it had been asked (run it through `curl`), which tells you whether it was a routing failure or a retrieval failure.

**4. Where It Got Worse.** At least two categories, or kinds of question, where `both` did worse than `web`, or where adding the tool made the agent slower, costlier or wronger. For each: the numbers, one failing question pasted with its `calls` and answer from a `both-*.jsonl` row, and the mechanism in two sentences. If `both` beat `web` everywhere in the table, look at `ms`, at questions where the agent called both tools and used neither, and at answers that cite a chunk and are wrong. The regressions are there.

**5. The number I trust least.** One number from this report, and why.

```bash
git add evals/grounded/run.ts evals/grounded/score.ts evals/grounded/runs evals/grounded/REPORT.md
git commit -m "A20: grounded retrieval eval, 3 runs per configuration, report"
git push
```

The `runs/` folder is committed: I re-run `score.ts` on it. In the rag repo, commit anything you changed there and push. One pull request per repo; push again each day. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Defense · Wed Dec 2, in class**

Five minutes each, at your laptop, from your committed `REPORT.md`, with the server running. I will ask you three things, and you can prepare all three:

1. Pick any cell in your `both` table and show me the rows in `runs/` that produce it.
2. Your worst regression: why it happens, and what you would change first.
3. Kill the server. Ask the agent a golden question. Tell me what it did and whether that is what it should do.

**Deliverable**
Rag repo: `rag/serve.py` + `scratch/a20-web-tool.md`. v2 agent repo: `src/tools/calls.ts` + `src/tools/knowledgeBase.ts` + the registration change + the web tool output change + `evals/grounded/questions.json` and `predictions.md` (committed before any run) + `run.ts` + `score.ts` + `runs/` (six files) + `REPORT.md` with "Where It Got Worse".

**Reflection Questions**

1. Paste your `KB_DESCRIPTION` as it stands now and, from `git log -p` on that file, any earlier version. For one question in Section 3 of your report, paste the tools the agent called on each of the three `both` runs. Is the wording of your description the cause? Say which words you think it read, and what you changed or would change.
2. Paste the `both` row and the `web` row for your worst regression category, then one `both-*.jsonl` row from that category in full. Walk through the row: which tool ran first, what came back, what the answer said, and at which of those points the failure entered.
3. Paste `evals/grounded/predictions.md` and the golden-set `correct` you got in each mode, with its range. Were you right about the worst category? Is the difference between the modes bigger than the range within a mode? Answer with the four numbers.
