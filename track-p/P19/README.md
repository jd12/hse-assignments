# P19 · ★ Milestone A1, Day 1: One Tool, Two Modes

**Meetings:** D42 · **Points:** 10 pts

**Watch — none**

No Boot.dev today. Chapters 1 to 10 are done, and the milestone is the pipeline you built from them, handed to a model as a tool it can choose to call.

**Notes**

**An agent is a loop, and you have written one before.** Unit 0's v2 agent sent a conversation to the model, and if the reply was a tool call it ran the tool, put the result into the conversation, and sent it again. `rag/agent.py` is that loop in forty lines of Python with one tool, `search_corpus`, which is `rag.rerank.rerank` with its five chunks formatted the way `rag/answer.py` formats them. The model decides whether to call it. Your retrieval pipeline is now a function the model can reach for.

**The baseline is the same model with no tool.** Milestone A1 is "grounded agent with a measured delta", and a delta needs two numbers. `none` mode offers no tool and asks the model to answer from what it knows, which for a famous text can be a lot: it has read about the Odyssey. `tool` mode offers the tool. The difference between the two on the same questions, scored the same way, is the delta, and P20 measures it, including where it got worse.

**The model picks the tool using only the strings you wrote.** The tool's description, the parameter's description and the system prompt are everything it knows about when to search. A vague description and it answers from memory; a pushy one and it searches for the capital of Australia. P20 scores routing separately from retrieval so you can tell which one failed.

**Answers are scored by a string, not a judge.** The golden set's `answer` field is the shortest string any correct answer must contain. An answer that contains it is correct; one that says `I don't have enough information` is a refusal; anything else is wrong. For the five out-of-corpus questions the right answer is the refusal. No model grades the model.

**Walkthrough — the agent, one question both ways, twenty questions committed**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P17 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/milestone-a1
```

*If P17 is not approved yet:* `git switch dev/answer && git switch -c dev/milestone-a1`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: rag/agent.py read, the three strings the model decides from written down
- [ ] Step 3: one question, tool mode and none mode, in evidence/P19.md
- [ ] Step 4: rag/eval/grounded_questions.json, twenty questions, committed
- [ ] Step 5: predictions for both modes, committed before P20
- [ ] Fill every slot in evidence/P19.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Read the agent.**

Open `rag/agent.py`. Three things to find before running it:

- `TOOLS`: the one tool, in the shape the OpenAI library expects (a name, a description, and the parameters as a small schema). The model reads the two description strings.
- `SYSTEM_TOOL` and `SYSTEM_NONE`: the two system prompts, one per mode. The first tells the model to use the tool before answering and to cite ids; the second tells it to answer from what it knows. Both allow exactly one refusal line, the same one `rag/answer.py` allows.
- `run`: the loop. Send the conversation; if the reply has `tool_calls`, append the model's request, run `search_corpus` for each, append each result as a `tool` message, send again; otherwise the reply is the answer. `MAX_ROUNDS` stops a model that searches forever.

Set `CORPUS_NOTE` to your corpus line, the same one as `rag/answer.py`. Then write under `## Step 2` in `evidence/P19.md` the three strings the model reads to decide whether to call the tool, quoted. If you change any of them, that is allowed, and the eval in P20 is where you would see what the change did; today, run them as they are.

**Step 3. One question, both ways.**

A golden question whose chunk the reranker finds, in `tool` mode:

```bash
uv run python -m rag.agent tool "<the question>"
```

*You should see* a `tool call:` line with the query the model chose and the five ids the tool returned, a `rounds` line (2 is one tool call and one answer), and the answer with ids. On Homer, for `How did the old nurse recognize Ulysses while washing his feet?`:

```text
{{LLM:P19-tool}}
```

The model's query is not your question: it rewrote it before searching, the way chapter 7 did, and the rewrite is the first thing to look at when the tool returns the wrong chunks. Then the same question with no tool:

```bash
uv run python -m rag.agent none "<the question>"
```

*You should see* no tool call, `rounds 1`, and an answer from the model's own memory. On Homer:

```text
{{LLM:P19-none}}
```

Paste both under `## Step 3` and say which is right, and where the `none` answer came from: your corpus (it cannot), the model's memory of the text, or nowhere.

*If it broke* with `rounds 5` and an empty answer, the model called the tool at every turn and never answered; `MAX_ROUNDS` ended it, and that question is a `wrong` in P20's terms. *If it broke* with the model answering in `tool` mode without calling the tool, that is routing, and the system prompt's second line is the place to look; keep the output for P20. *If it broke* with an error mentioning `tools` or `function calling`, the free router handed the request to a model that cannot call tools; `rag/llm.py` retries, and a second run usually lands on one that can. If it keeps failing, pin a model: in `rag/llm.py`, set `MODEL` to one of the `:free` models on the OpenRouter models page that lists tool calling, and tell me which.

**Step 4. Twenty questions, committed before any run.**

`rag/eval/grounded_questions.json`: fifteen of your golden questions, copied with their `id`, `question`, `answer` and `chunk_ids`, and five out-of-corpus questions with `answer` set to `null` and `chunk_ids` empty. The five should be the kind the model could answer from memory if it wanted to; the eval is partly about whether it refuses when it should:

```json
[
  {"id": "g01", "question": "How did the old nurse recognize Ulysses while washing his feet?",
   "answer": "scar", "chunk_ids": ["c00644"]},
  {"id": "x01", "question": "Who wrote the Divine Comedy, and in what language?",
   "answer": null, "chunk_ids": []}
]
```

Choose the fifteen before looking at `rag/eval/runs.json`: a mix of kinds, not the fifteen the pipeline finds best. Check the file loads and count it:

```bash
uv run python -c "
import json
qs = json.load(open('rag/eval/grounded_questions.json', encoding='utf-8'))
print(len(qs), 'questions,', sum(1 for q in qs if q['answer'] is None), 'out of corpus')
"
```

*You should see* `20 questions, 5 out of corpus`. Paste the twenty ids and questions under `## Step 4`, with the five marked, then commit the file:

```bash
git add rag/agent.py rag/eval/grounded_questions.json evidence/P19.md
git commit -m "P19: the one-tool agent, twenty milestone questions before any run"
git push -u origin dev/milestone-a1
```

Write the commit hash under `## Step 4`. I check the timestamps against P20's results file. Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P20.

**Step 5. Predictions, committed before P20.**

Fill the table under `## Step 5` in `evidence/P19.md`: for each mode, how many of the twenty will be correct, refused, and wrong. Then the two harder ones: the two in-corpus questions you expect the `none` mode to get right anyway, with a reason each, and the out-of-corpus question you expect the `tool` mode to answer when it should refuse. Commit:

```bash
git add evidence/P19.md
git commit -m "P19: milestone predictions before the eval"
git push
```

**Commit, sign off.** Fill `## Reflection` in `evidence/P19.md` (the questions are below) and run `uv run python scripts/check_evidence.py P19` until it says `all slots filled`. Then:

```bash
git add evidence/
git commit -m "P19: reflections"
git push
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`rag/eval/grounded_questions.json` with twenty questions, committed before the predictions and the predictions before P20 + `evidence/P19.md` with every slot filled.

**Reflection Questions** (answer under `## Reflection` in `evidence/P19.md`)

1. Paste your Step 3 outputs. Say what the model did between the two requests in `tool` mode, in terms of the loop in `rag/agent.py`, and what the second request contained that the first did not.

   *How to get it:* the first reply was a tool call, not an answer; the loop ran `search_corpus` with the model's query and appended the five chunks as a `tool` message; the second request was the whole conversation plus that message, and the reply to it was the answer. The second request is longer by five chunks.

2. The baseline is the same model with no tool. Say why that is the right baseline for a milestone called "grounded agent with a measured delta", and name one thing it does not control for.

   *How to get it:* changing one thing, the tool, and holding the model, the prompt's refusal rule and the questions fixed is what makes the difference attributable to grounding. It does not control for the model's memory of your corpus: a famous text makes the `none` baseline strong, and the delta small, through no fault of the retriever.

3. Paste your predictions. Say which number in the table you are least sure of and what would have to be true of the model's training for it to come out the other way.

   *How to get it:* the `none` row's `correct` depends on how much of your corpus the model has read; the `tool` row's `wrong` on the out-of-corpus five depends on whether the model obeys a refusal line when it knows the answer. Either can go both ways, and you have one data point from Step 3.
