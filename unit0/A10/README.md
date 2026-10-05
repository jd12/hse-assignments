# A10 · Failure Catalog: Why Models Lie

**Meetings:** D20–D21 · **Points:** 15 pts

**Watch — none**

**Day 1 — none.** Write the questions, commit them, and run your A09 sampler and the API on them.

**Day 2 — none.** Mark the answers and write the catalog.

**During the video**

No video. You are producing the evidence this time. Have three things open: `data/corpus.txt` in VS Code with search (you will be quoting it), `sampling/lab.py` from A09, and a terminal in the repo root.

**Notes**

**The corpus is the answer key.** Every question you write has its answer in your file, with the sentence that proves it. When a model answers without seeing the file, you can check it against the text rather than against your memory or the internet. That is what turns "it made something up" into a catalog entry.

**Famous corpora are partly memorized.** If your corpus is a well-known public text, the API model has probably read it. It will get the famous facts and invent the obscure ones. That is a finding, not a problem, and it is why a third of your questions should be about details nobody quotes.

**Quotes must match the file exactly.** A quote you retyped from memory will not be found by the checker in Step 3, and a question whose quote cannot be found is not in the answer key. Copy from the file.

**`Answer in one sentence.` is part of the experiment.** `ask.py` appends it to every question. Whatever it does to the model's willingness to say "I don't know" is something you did. Do not change it between runs you compare.

**Rates, not anecdotes.** Every question runs five times on each model. "It said Athens" is a story. "4 of 5 runs said Athens; the file says Sparta" is an entry.

**Commit as you go.** `git log --oneline failures/` shows the order you found things in, and I read it. One commit at 4:47 AM tells me one thing; a dozen across two meetings tells me another.

**Walkthrough — questions your corpus can answer, asked of models that have not seen it**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. A09 is due tomorrow morning and will not have merged, and today uses its sampler, so branch from it:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch dev/sampling && git pull    # or: git switch main && git pull, if A09 has merged
git switch -c dev/failure-catalog
mkdir -p failures/traces
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: empty catalog committed
- [ ] Day 1: 20 questions with quotes, checker says 0 missing, committed
- [ ] Day 1: ask.py run, 5 API + 5 GPT-2 answers per question, traces saved
- [ ] Day 2: every answer marked against the quote
- [ ] Day 2: ten catalog entries with rates, categories, hypotheses
- [ ] Extension: one category, measured against a baseline
- [ ] Push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Commit the empty catalog.**

```bash
printf '# Failure Catalog\n' > failures/CATALOG.md
git add failures/CATALOG.md
git commit -m "A10: open the catalog"
```

**Step 3. Twenty questions, each with its proof.**

Create `failures/questions.jsonl`, one question per line:

```json
{"id": "q01", "question": "What false name does Ulysses give the Cyclops?", "answer": "Noman", "quote": "my name is Noman"}
```

That is an Odyssey example. Yours come from your corpus. Each answer is one checkable fact: a name, a number, an object, who said what to whom. At least seven are about minor details. At least three have an answer that a reasonable guess would get wrong.

Check every quote against the file:

```python
# failures/check_questions.py
import json
from pathlib import Path

text = " ".join(Path("data/corpus.txt").read_text(encoding="utf-8").split())
qs = [json.loads(l) for l in Path("failures/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
missing = [q["id"] for q in qs if " ".join(q["quote"].split()) not in text]
print(len(qs), "questions,", len(missing), "quotes not found:", missing)
```

```bash
uv run python failures/check_questions.py
```

*You should see* `20 questions, 0 quotes not found: []`. Fix every id it lists; the usual cause is a curly quote or an em dash you retyped as a straight one.

Commit before any model sees a question:

```bash
git add failures/questions.jsonl failures/check_questions.py
git commit -m "A10: 20 corpus questions with quotes, before asking any model"
```

I will check your commit timestamps.

**Step 4. Ask both models, five times each.**

```python
# failures/ask.py
import json, math, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sampling.lab import MODEL, generate

N, TEMP = 5, 0.7

def ask_api(question):
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": 60,
                       "logprobs": True, "top_logprobs": 5,
                       "messages": [{"role": "user", "content": question + " Answer in one sentence."}]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        c = json.load(r)["choices"][0]
    tokens = [[t["token"], [[a["token"], round(math.exp(a["logprob"]), 3)] for a in t["top_logprobs"]]]
              for t in c["logprobs"]["content"]]
    return {"text": c["message"]["content"], "tokens": tokens}

for line in Path("failures/questions.jsonl").read_text(encoding="utf-8").splitlines():
    q = json.loads(line)
    q["api"] = [ask_api(q["question"]) for _ in range(N)]
    q["gpt2"] = [generate(f"Q: {q['question']}\nA:", seed=s, temperature=TEMP).split("\n")[0]
                 for s in range(N)]
    Path(f"failures/traces/{q['id']}.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
    print(q["id"], "|", q["api"][0]["text"][:60], "|", q["gpt2"][0][:40])
```

```bash
uv run python failures/ask.py
```

*You should see* twenty lines, one per question, and twenty files in `failures/traces/`. It is 100 API calls of 60 tokens and 100 GPT-2 generations, a few minutes in all. GPT-2 is your A09 sampler, unchanged.

*If it broke* partway through, the traces already written are fine; comment out the ids that finished and run again rather than paying for them twice.

Commit the traces now, before you have read them closely.

**Step 5. Mark every answer (Day 2).**

Open each trace next to its quote. Mark each of the ten answers as one of four things:

| Mark | Means |
|---|---|
| **right** | States the fact in the quote. |
| **fabricated** | States a specific different fact, as fact. |
| **hedged** | Says it does not know, or that the text does not say. |
| **off** | Answers some other question, or produces no answer at all. |

Put the counts in `failures/marks.md`; `off` is whatever is left of five, so it needs no column:

| id | API right/5 | API fabricated/5 | API hedged/5 | GPT-2 right/5 | GPT-2 fabricated/5 | notes |
|---|---|---|---|---|---|---|

*You should see* GPT-2 right on almost nothing and fabricating or wandering off on almost everything: 124 million parameters and no corpus. The API will be spread out, right on famous facts and fabricating on minor ones, with some questions split across the five runs. The split ones are where the catalog gets interesting.

**Step 6. Write ten entries.**

Each entry in `CATALOG.md` has these fields, in this order:

| Field | What goes in it |
|---|---|
| Repro | The question verbatim, the model, every sampling setting, and the command that reruns it. |
| Observed | The answer, pasted. |
| Expected | The quote from your corpus, and where it is in the file. |
| Rate | Failures over runs, with N. `4/5 API runs said X`. |
| Category | One of the eight below, or one you name. |
| Hypothesis | A claim about machinery, tied to something from A03 to A09. |
| Evidence | Logprobs at the failure position, the tokenizer split, a diff across runs, or a token count. |

The eight categories: tokenization artifact · context window or truncation · sampling nondeterminism · retrieval or grounding failure · instruction-following collapse · tool-schema mismatch · confident fabrication · loop or repetition.

At least six entries come from your twenty questions. The other four can come from anything you ran this term: your A06 search asked these questions, the A09 temperature table (GPT-2 at temperature 0 on a long run), or `cl100k_base` splitting the name in one of your answers. At least four entries show the same question on both models. At least three carry evidence; `tokens` in each trace holds the top-5 alternatives at every position of every API answer, so for a fabricated name you can read what the model's probability was on the first wrong token and whether the right one was in the top five.

At least two are failures you caused, where the model did what it was told. Look hard at your own prompt.

One entry you cannot explain goes in bounded, not blank: under Hypothesis write `unknown`, then name the categories you ruled out and the evidence that ruled each one out. An empty Hypothesis field is not bounded; this is:

> Category: unknown. Ruled out: sampling (fails 5/5 at 0.7 and 5/5 at 0), memorization gap (the model quotes the surrounding passage correctly).

A hypothesis with a trace under it is an argument. Without one it is a guess, and I read it as a guess.

**Extension — one category, measured (CHOOSE)**

Pick one of your catalog's categories and go deep on it. Say in `CATALOG.md` which one and why that one. Each option has a number, a baseline, and a comparison. Write your predicted number in the log before you run.

| Option | Category | Do this | Number | Baseline |
|---|---|---|---|---|
| **A** | Confident fabrication | Your five most-fabricated questions again, five runs each, with the full chunk containing the quote pasted into the prompt above the question. | API right/25 open-book | API right/25 closed-book, from `marks.md` |
| **B** | Retrieval or grounding | All twenty questions through your A06 `search()`. Then ask the API each question with the top-1 chunk pasted in. | Right when top-1 held the quote vs right when it did not | Top-3 hit rate of keyword search on the same twenty |
| **C** | Sampling nondeterminism | The question whose five API runs were most split, twenty runs each at `TEMP` 0, 0.7 and 1.2. | Right/20 at each temperature | Temperature 0 |
| **D** | Loop or repetition | GPT-2 on all twenty `Q: ... A:` prompts, 60 new tokens, at temperature 0, at 0.7, and at 0.7 with `top_p=0.9`. Count outputs where any three-word sequence appears three or more times. | Looping outputs/20 at each setting | Temperature 0 |

Then answer, in the deep-dive section: where did it get worse? For A, the questions still wrong with the answer on the page. For B, whether a wrong chunk made the answer worse than no chunk. For C, whether temperature 0 was the most right or only the most consistent. For D, whether the setting that looped least also answered worst.

**Commit, push, PR, sign off.**

```bash
git add failures/
git commit -m "A10: ten catalog entries with rates, deep dive on <category>"
git push -u origin dev/failure-catalog
git log --oneline failures/
```

*You should see* several commits, not one, with the questions commit before the traces.

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the hypothesis you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`failures/CATALOG.md` (ten entries plus the deep dive) + `failures/questions.jsonl` + `failures/check_questions.py` + `failures/ask.py` + `failures/marks.md` + `failures/traces/`.

**Reflection Questions**

1. Your most-fabricated question: paste it, the quote from your corpus, and two of the fabricated answers. From its trace, give the probability on the first wrong token of one of them and say whether the right token was anywhere in the top five at that position. What does that number say about how "confident" the model was?

   *How to get it:* the question is the row of `marks.md` with the largest `API fabricated/5`. Its trace is `failures/traces/<id>.json`; the five answers are the `text` fields under `api`, and `tokens` is a list of `[token, [[alternative, probability] × 5]]` for every position. To print one answer position by position, with your id and run number:

   ```bash
   uv run python -c "
   import json
   q = json.load(open('failures/traces/q07.json')); run = q['api'][0]
   print(run['text'])
   for tok, top5 in run['tokens']:
       print(repr(tok).ljust(14), top5)
   " | head -20
   ```

   Read down until the first token that is not in the quote; the number beside it in `top5` is the probability on the wrong token, and whether the right token appears in that same list of five is the second half of the question.

2. Paste `git log --oneline failures/`. Name the entry you first filed under the wrong category and what moved it. Then name one of the two failures you caused, and the line of `ask.py` or `questions.jsonl` responsible.

   *How to get it:* the log is the command as written, from the repo root. The re-filed entry is one whose category changed between two commits: `git log -p --follow -- failures/CATALOG.md | grep "^[-+].*Category"` prints every category line that was added or removed, in order. The usual self-caused failures are the `Answer in one sentence.` suffix in `ask_api` (it pushes the model away from "the text does not say") and a question that names the answer's category in its wording ("What false name…") so the model has a shape to fill; the line is `grep -n "one sentence" failures/ask.py`, or the question's `id`.

3. Your extension: the number, the baseline, and the comparison, with the prediction you wrote in the log before running. Where did it get worse? State the result that would falsify your catalog hypothesis for that category, and whether anything in your deep dive came close.

   *How to get it:* the prediction is in your Day 1 log entry (`grep -n -i predict logs/*.md` in the log repo); the number and baseline are the two figures your option's row of the extension table asked for, and the deep-dive section of `CATALOG.md` already holds them. The falsifier is the result that would have come out the other way if your hypothesis were wrong: for A, open-book no better than closed-book; for B, right answers independent of whether the top-1 chunk held the quote; for C, temperature 0 no more right than 1.2; for D, the setting that looped least also the most right.
