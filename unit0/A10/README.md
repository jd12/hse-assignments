# A10 · Failure Catalog: Why Models Lie

**Meetings:** D20–D21 · **Points:** 15 pts

**Watch — none**

**Day 1 — none.** Write the questions, commit them, and run your A09 sampler and the API on them.

**Day 2 — none.** Mark the answers and write the catalog.

**During the video**

No video. You are producing the evidence this time. Have three things open: `data/corpus.txt` in VS Code with search (you will be quoting it), `sampling/lab.py` from A09, and a terminal in the repo root.

**Notes**

**The corpus is the answer key.** Every question you write has its answer in your file, with the sentence that proves it. When a model answers without seeing the file, you can check it against the text rather than against your memory or the internet. That is what turns "it made something up" into a catalog entry, and it is why the rate of made-up answers is a number you can measure, not an impression.

**Famous corpora are partly memorized.** If your corpus is a well-known public text, the API model has probably read it. It will get the famous facts and invent the obscure ones. That is a finding, not a problem, and it is why a third of your questions are about details nobody quotes.

**Quotes must match the file exactly.** A quote you retyped from memory will not be found by the checker in Step 3, and a question whose quote cannot be found is not in the answer key. Copy from the file.

**`Answer in one sentence.` is part of the experiment.** `ask.py` appends it to every question. Whatever it does to the model's willingness to say "I don't know" is something you did. Do not change it between runs you compare.

**Rates, not anecdotes.** Every question runs five times on each model. "It said Athens" is a story. "4 of 5 runs said Athens; the file says Sparta" is an entry.

**Logprobs are the evidence.** `ask.py` saves the model's top-5 alternatives at every token of every API answer. At the token where a wrong name begins, those five numbers say how sure the model was and whether the right name was even in the running. `evidence.py` in Step 6 prints them.

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
- [ ] Day 1: catalog template copied to CATALOG.md and committed
- [ ] Day 1: 20 questions with quotes, checker says 0 not found, committed
- [ ] Day 1: ask.py run, 5 API + 5 GPT-2 answers per question, traces committed
- [ ] Day 2: marks.md filled, mark.py counts pasted
- [ ] Day 2: ten catalog entries with rates, categories, hypotheses, evidence.py output in three
- [ ] Extension: option chosen, prediction in the log, deepdive.py run, deep-dive section written
- [ ] Push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Open the catalog from the template.**

Create `failures/CATALOG_TEMPLATE.md` (right-click `failures`, **New File**) and paste this in. Ten entry skeletons with the seven fields, one hint per field in angle brackets, and the deep-dive section at the end. You fill the copy, not the template.

```markdown
# Failure Catalog: <your name>

Corpus: <data/SOURCE.md's one-line description>. API model: <MODEL from sampling/lab.py>. GPT-2: 124M, A09 sampler, T=0.7.
Questions: 20 (failures/questions.jsonl). Runs: 5 per question per model. Marks: failures/marks.md.

<!-- Ten entries. Keep the seven field names and their order. Delete every hint in angle brackets as you fill it.
     At least six entries come from your twenty questions; at least four show the same question on both models;
     at least three carry evidence from evidence.py; at least two are failures you caused; exactly one has Hypothesis: unknown, bounded. -->

## Entry 1: <four-word title>

- **Repro:** <the question verbatim, the model, every sampling setting, and the command that reruns it: `uv run python failures/ask.py` for a trace, or `uv run python failures/evidence.py q07 1` for one run>
- **Observed:** <the answer, pasted from the trace, not retyped; name the run: API run 3 of q07>
- **Expected:** <the quote, pasted, and where it is: `grep -n "..." data/corpus.txt` gives the line>
- **Rate:** <failures over runs, with N: 4/5 API runs said X; 5/5 GPT-2 runs wandered off>
- **Category:** <one of: tokenization artifact · context window or truncation · sampling nondeterminism · retrieval or grounding failure · instruction-following collapse · tool-schema mismatch · confident fabrication · loop or repetition, or one you name>
- **Hypothesis:** <one claim about machinery, tied to something from A03 to A09: the model never saw this text (A05b, A06), the name splits into pieces (A04), temperature 0.7 draws from a spread distribution (A09)>
- **Evidence:** <what evidence.py printed at the failure position, a tokenizer split, a diff across runs, or a token count; or "none yet", honestly>

## Entry 2: <title>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <>
- **Evidence:** <>

## Entry 3: <title>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <>
- **Evidence:** <>

## Entry 4: <title>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <>
- **Evidence:** <>

## Entry 5: <title>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <>
- **Evidence:** <>

## Entry 6: <title>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <>
- **Evidence:** <>

## Entry 7: <title, a failure you caused>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <name the line of ask.py or questions.jsonl that did it>
- **Evidence:** <>

## Entry 8: <title, a failure you caused>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <name the line of ask.py or questions.jsonl that did it>
- **Evidence:** <>

## Entry 9: <title, from something else you ran this term: A06 search, the A09 table, a cl100k_base split>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** <>
- **Hypothesis:** <>
- **Evidence:** <>

## Entry 10: <title, the one you cannot explain>

- **Repro:** <>
- **Observed:** <>
- **Expected:** <>
- **Rate:** <>
- **Category:** unknown
- **Hypothesis:** unknown. Ruled out: <category (the evidence that rules it out)>, <category (evidence)>
- **Evidence:** <the evidence named above, pasted>

## Deep dive: option <A/B/C/D>, <category>

Why this category: <one sentence>
Predicted number, written in the log before running: <n> (`grep -n -i predict logs/*.md` in the log repo finds it)

    <paste the NUMBER, BASELINE and COMPARISON lines deepdive.py printed, indented four spaces>

Where it got worse: <A: the questions still wrong with the answer on the page. B: whether a wrong chunk made the answer worse than no chunk. C: whether temperature 0 was the most right or only the most consistent. D: whether the setting that looped least also answered worst.>
Falsifier: <the result that would have come out the other way if my hypothesis for this category were wrong, and whether anything above came close>
```

```bash
cp failures/CATALOG_TEMPLATE.md failures/CATALOG.md
git add failures/CATALOG_TEMPLATE.md failures/CATALOG.md
git commit -m "A10: open the catalog from the template"
```

**Step 3. Twenty questions, each with its proof.**

Create `failures/questions.jsonl`, one question per line, each line in exactly this shape:

```json
{"id": "q01", "question": "What false name does Ulysses give the Cyclops?", "answer": "Noman", "quote": "my name is Noman"}
```

That is an Odyssey example. Yours come from your corpus. Each answer is one checkable fact: a name, a number, an object, who said what to whom. At least seven are about minor details. At least three have an answer that a reasonable guess would get wrong. **A minor detail is a fact the text states once, in passing, that no summary of the text includes**: on Homer, Euryclea's father is Ops, one half-line in Book I. **A fact a reasonable guess gets wrong has an obvious answer that the text contradicts**: on Homer, how many years Aegisthus ruled Mycene after killing Agamemnon; a reader who knows the story's famous tens guesses ten, and the text says seven, with Orestes back in the eighth year.

Here is how to find twenty facts in the time you have. For each one, four steps:

1. Search the corpus for a proper noun, a place or a term you know is in it, and pick one line number from the output:

   ```bash
   grep -n -i "noman" data/corpus.txt | head
   ```

   On Homer this printed six lines, the first `4082:therefore, the present you promised me; my name is Noman; this is what`. A name that prints one to three lines is a minor detail; a name that prints forty (`grep -c -i "euryclea" data/corpus.txt` says 44 on Homer) is a main character, and the fact you want is on one of those lines, not the name itself.

2. Read the paragraph around that line, with your line number in place of 4082 and about eight lines either side:

   ```bash
   sed -n '4074,4090p' data/corpus.txt
   ```

3. Write the question that paragraph answers, with the answer as a short field. The Homer paragraph says "Cyclops, you ask my name and I will tell it you ... my name is Noman", so the question is "What false name does Ulysses give the Cyclops?" and the answer is `Noman`. Do not put the answer in the question.

4. Copy the quote: four to twelve words from the paragraph, **exactly as the file has them**, select-and-copy from the terminal or VS Code, never retyped. The quote has to contain the answer, or say it in other words (`bound Ino's veil under his arms` for the answer `her veil`).

Then check every line:

```python
# failures/check_questions.py
# Run from the repo root:  uv run python failures/check_questions.py
# Checks failures/questions.jsonl against data/corpus.txt: every quote has to be in the file, word for word.
# Then two counts that say how hard your questions are: how many share no content words with their quote,
# and how many quotes are too short to be a reliable answer key.
import json
from pathlib import Path

STOP = {"the", "a", "an", "of", "and", "to", "in", "is", "was", "what", "who", "how", "why", "does", "did",
        "which", "whose", "where", "when", "do", "he", "she", "it", "they", "his", "her", "its", "their", "them",
        "him", "that", "this", "for", "from", "by", "with", "on", "at", "as", "be", "are", "were", "had", "has", "have"}

def words(s):
    """ the distinct words of s, lowercased, letters digits and apostrophes only, minus STOP """
    cleaned = "".join(ch if ch.isalnum() or ch == "'" else " " for ch in s.lower())
    return {w[:-2] if w.endswith("'s") else w for w in cleaned.split()} - STOP      # Euryclea's -> euryclea

text = " ".join(Path("data/corpus.txt").read_text(encoding="utf-8").split())
qs = [json.loads(l) for l in Path("failures/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
missing = [q["id"] for q in qs if " ".join(q["quote"].split()) not in text]
print(len(qs), "questions,", len(missing), "quotes not found:", missing)

ids = [q["id"] for q in qs]
twice = sorted({i for i in ids if ids.count(i) > 1})
if twice:
    print("ids used more than once:", twice)
no_shared = [q["id"] for q in qs if not (words(q["question"]) & words(q["quote"]))]
short = [q["id"] for q in qs if len(q["quote"]) < 15]
answer_not_in_quote = [q["id"] for q in qs if " ".join(q["answer"].lower().split()) not in " ".join(q["quote"].lower().split())]
print(len(no_shared), "questions share no content words with their quote:", no_shared)
print(len(short), "quotes under 15 characters:", short)
print(len(answer_not_in_quote), "answers not in their quote word for word:", answer_not_in_quote)
```

```bash
uv run python failures/check_questions.py
```

*You should see* four lines. The first is `20 questions, 0 quotes not found: []`; fix every id it lists, and the usual cause is a curly quote or an em dash you retyped as a straight one. The second counts questions that share no content word with their quote: those are the questions a model cannot answer by echoing your wording, and the ones Extension B's search has to understand rather than match; a count of 0 means every question hands the model words to fill in around. The third counts quotes under 15 characters; make those longer, because a short quote can sit in many places in the file and then it proves nothing about where the answer is. The fourth lists answers that are not in their quote word for word; that is fine when the quote says the same thing in other words, and `evidence.py` in Step 6 checks tokens against both. On Homer the four lines were:

```text
20 questions, 0 quotes not found: []
1 questions share no content words with their quote: ['q11']
0 quotes under 15 characters: []
3 answers not in their quote word for word: ['q09', 'q12', 'q18']
```

Commit before any model sees a question:

```bash
git add failures/questions.jsonl failures/check_questions.py
git commit -m "A10: 20 corpus questions with quotes, before asking any model"
```

I will check your commit timestamps.

**Step 4. Ask both models, five times each.**

```python
# failures/ask.py
# Run from the repo root:  uv run python failures/ask.py
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
    if Path(f"failures/traces/{q['id']}.json").exists():   # already asked: skip, so a rerun costs nothing
        continue
    q["api"] = [ask_api(q["question"]) for _ in range(N)]
    q["gpt2"] = [generate(f"Q: {q['question']}\nA:", seed=s, temperature=TEMP).split("\n")[0]
                 for s in range(N)]
    Path(f"failures/traces/{q['id']}.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
    print(q["id"], "|", q["api"][0]["text"][:60], "|", q["gpt2"][0][:40])
```

```bash
uv run python failures/ask.py
```

*You should see* twenty lines, one per question, each the id, the first sixty characters of the first API answer, and the first forty of the first GPT-2 answer, and twenty files in `failures/traces/`. It is 100 API calls of 60 tokens and 100 GPT-2 generations, a few minutes in all. GPT-2 is your A09 sampler, unchanged, at the same `0.7` the API gets; the API is **closed-book**, the question alone, no passage. Each trace holds the question, answer and quote, then `api`, five `{"text", "tokens"}` objects where `tokens` is `[token, [[alternative, probability] × 5]]` for every position, and `gpt2`, five strings.

<!-- JD: fill one Homer line of the twenty after a run. -->

*If it broke* partway through, the traces already written are fine; run the same command again. It skips every question that already has a file in `failures/traces/`, so nothing is paid for twice. `HTTP Error 400` naming `logprobs` means `MODEL` is a reasoning model, as in A09.

Commit the traces now, before you have read them closely:

```bash
git add failures/ask.py failures/traces/
git commit -m "A10: 5 API + 5 GPT-2 answers per question, unread"
```

**Step 5. Mark every answer (Day 2).**

Each of the two hundred answers gets one of four marks:

| Mark | Letter | Means |
|---|---|---|
| **right** | `r` | States the fact in the quote. |
| **fabricated** | `f` | States a specific different fact, as fact. |
| **hedged** | `h` | Says it does not know, or that the text does not say. |
| **off** | `o` | Answers some other question, or produces no answer at all. |

Create `failures/mark.py` and paste this in. Run without an argument it prints every answer beside its quote with a blank mark, and writes `failures/marks.md` with one row per question for you to fill; run with `counts` it reads your letters back and prints the counts table.

```python
# failures/mark.py
# Run from the repo root:
#   uv run python failures/mark.py           prints every answer beside its quote with a blank mark, and writes
#                                            failures/marks.md with one row per question to fill (if it is not there yet)
#   uv run python failures/mark.py counts    after you fill the marks: prints the counts table, the most-fabricated
#                                            question and the most split question
import json, sys
from pathlib import Path

MARKS = {"r": "right", "f": "fabricated", "h": "hedged", "o": "off"}
traces = sorted(Path("failures/traces").glob("q*.json"))
qs = [json.loads(p.read_text(encoding="utf-8")) for p in traces]
assert qs, "no traces in failures/traces; run failures/ask.py first"

def read_marks():
    """ the rows under '## Marks' in failures/marks.md, as {id: (five API letters, five GPT-2 letters, notes)} """
    rows, inside = {}, False
    for line in Path("failures/marks.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Marks"
            continue
        if inside and line.startswith("| q"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = (cells[1], cells[2], cells[3] if len(cells) > 3 else "")
    return rows

if sys.argv[1:] == []:
    for q in qs:
        print(f"\n{'=' * 110}\n{q['id']}  {q['question']}\n      answer: {q['answer']}\n      quote:  {q['quote']}")
        for model in ("api", "gpt2"):
            for k, run in enumerate(q[model], 1):
                text = " ".join((run["text"] if model == "api" else run).split())
                print(f"  {model:<5} {k}  [ ]  {text[:120]}")
    out = Path("failures/marks.md")
    if out.exists():
        print("\nfailures/marks.md already exists; left as it is")
    else:
        lines = ["# Marks: <your name>", "",
                 "One letter per run, in run order, r right, f fabricated, h hedged, o off. Replace every dot.", "",
                 "## Marks", "", "| id | API runs 1-5 | GPT-2 runs 1-5 | notes |", "|---|---|---|---|"]
        lines += [f"| {q['id']} | ..... | ..... |  |" for q in qs]
        lines += ["", "## Counts", "", "<paste what `uv run python failures/mark.py counts` prints>", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nwrote failures/marks.md with {len(qs)} rows to fill")

elif sys.argv[1] == "counts":
    rows = read_marks()
    bad = [i for i, (a, g, _) in rows.items() if len(a) != 5 or len(g) != 5 or set(a + g) - set("rfho")]
    assert not bad, f"these rows are not five letters from r f h o, twice: {bad}"
    missing = [q["id"] for q in qs if q["id"] not in rows]
    assert not missing, f"no row under ## Marks for {missing}"
    byid = {q["id"]: q for q in qs}
    print("| id | API right/5 | API fabricated/5 | API hedged/5 | GPT-2 right/5 | GPT-2 fabricated/5 | notes |")
    print("|---|---|---|---|---|---|---|")
    tot = [0] * 5
    for i, (a, g, notes) in rows.items():
        counts = [a.count("r"), a.count("f"), a.count("h"), g.count("r"), g.count("f")]
        tot = [x + y for x, y in zip(tot, counts)]
        print(f"| {i} | " + " | ".join(str(c) for c in counts) + f" | {notes} |")
    n = 5 * len(rows)
    print(f"| all | " + " | ".join(f"{c}/{n}" for c in tot) + " |  |")
    most_f = max(rows, key=lambda i: (rows[i][0].count("f"), -int(i[1:])))
    most_split = max(rows, key=lambda i: (len(set(rows[i][0])), -max(rows[i][0].count(m) for m in rows[i][0]), -int(i[1:])))
    print(f"\nmost-fabricated question (API): {most_f}, fabricated {rows[most_f][0].count('f')}/5, marks {rows[most_f][0]}")
    print(f"   {byid[most_f]['question']}\n   quote: {byid[most_f]['quote']}")
    print(f"most split question (API): {most_split}, marks {rows[most_split][0]}, {len(set(rows[most_split][0]))} different marks across 5 runs")
    print(f"   {byid[most_split]['question']}")
    both = [i for i, (a, g, _) in rows.items() if a.count("r") == 5 and g.count("r") == 0]
    print(f"API right 5/5 and GPT-2 right 0/5: {len(both)} of {len(rows)} questions {both}")
else:
    print("usage: uv run python failures/mark.py [counts]")
```

```bash
uv run python failures/mark.py > failures/to_mark.txt
wc -l failures/to_mark.txt
```

*You should see* `302 failures/to_mark.txt`: twenty blocks of fifteen lines (a blank line, a rule, the question, the answer field, the quote, five `api` lines, five `gpt2` lines), then a line saying `wrote failures/marks.md with 20 rows to fill`. Open `to_mark.txt` beside `marks.md`. Each answer line has a `[ ]`; read the answer against the quote two lines above it, decide `r`, `f`, `h` or `o`, and type that letter into the matching position of the row in `marks.md`: the first API letter is `api 1`, the fifth GPT-2 letter is `gpt2 5`. `.....` becomes something like `rrfrf`. A GPT-2 answer that is a question, a blank, or wanders into another topic is `o`; a loop of the same words is `o` too, unless the words state a different fact. When the dots are gone:

```bash
uv run python failures/mark.py counts
```

*You should see* a twenty-row table of counts (`off` is whatever is left of five, so it has no column), a totals row, then the most-fabricated question with its quote, the most split question, and how many questions the API got 5/5 while GPT-2 got 0/5. Whatever your corpus, expect GPT-2 right on almost nothing and fabricating or wandering off on almost everything: 124 million parameters and no corpus. The API will be spread out, right on famous facts and fabricating on minor ones, with some questions split across the five runs. The split ones are where the catalog gets interesting, and the most split one is what Extension C runs.

<!-- JD: fill the Homer totals row and the most-fabricated id after a run. -->

*If it broke* with `AssertionError: these rows are not five letters`, a row still has a dot or a letter outside `r f h o`; the message names the ids. Paste the table and the lines under it into `marks.md` under `## Counts`, and commit:

```bash
git add failures/mark.py failures/marks.md failures/to_mark.txt
git commit -m "A10: 200 answers marked"
```

**Step 6. Write ten entries.**

Each entry in `CATALOG.md` has the seven fields the template gives, in this order:

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

At least six entries come from your twenty questions. The other four can come from anything you ran this term: your A06 search asked these questions, the A09 temperature table (GPT-2 at temperature 0 on a long run), or `cl100k_base` splitting the name in one of your answers. At least four entries show the same question on both models. At least two are failures you caused, where the model did what it was told; look hard at your own prompt, the `Answer in one sentence.` suffix and the wording of your questions.

At least three carry evidence from the logprobs. Create `failures/evidence.py` and paste this in. Given a question id and an API run number it prints that answer token by token with the top-5 alternatives at every position, flags the first token that is in neither the quote nor the question, and says whether the right answer was in the top five there.

```python
# failures/evidence.py
# Run from the repo root:  uv run python failures/evidence.py q07 1        (question id, API run number 1 to 5)
# Prints one API answer token by token with the model's top-5 alternatives at every position, flags the first
# token that is in neither the quote nor the question, and says whether the right answer was in the top five there.
import json, sys
from pathlib import Path

STOP = {"the", "a", "an", "of", "and", "to", "in", "is", "was", "what", "who", "how", "why", "does", "did",
        "which", "whose", "where", "when", "do", "he", "she", "it", "they", "his", "her", "its", "their", "them",
        "him", "that", "this", "for", "from", "by", "with", "on", "at", "as", "be", "are", "were", "had", "has", "have"}

assert len(sys.argv) == 3, "usage: uv run python failures/evidence.py <id> <run 1-5>"
qid, run_no = sys.argv[1], int(sys.argv[2])
q = json.loads(Path(f"failures/traces/{qid}.json").read_text(encoding="utf-8"))
run = q["api"][run_no - 1]
quote, question, answer = (" ".join(q[k].lower().split()) for k in ("quote", "question", "answer"))

def core(tok):
    """ the letters, digits and apostrophes of a token, lowercased, minus a possessive: ' Noman' -> 'noman', ' prophet's' -> 'prophet', ',' -> '' """
    c = "".join(ch for ch in tok.lower() if ch.isalnum() or ch == "'")
    return c[:-2] if c.endswith("'s") else c

def where(tok):
    """ where this token's text can be found: quote, question, neither, or too short / a stop word to say """
    c = core(tok)
    if len(c) < 3 or c in STOP:
        return ""
    if c in quote:
        return "in quote"
    if c in question:
        return "in question"
    return "NOT IN QUOTE OR QUESTION"

print(f"{qid} run {run_no}: {q['question']}")
print(f"  answer: {q['answer']}\n  quote:  {q['quote']}\n  text:   {run['text']}")
print("  contains the answer field: " + ("yes, so the flag below points at filler, not a fabrication" if answer in " ".join(run["text"].lower().split()) else "no") + "\n")
print(f"{'pos':>3} {'token':<16} {'top-5 alternatives [token, probability]':<84} where")
first = None
for pos, (tok, top5) in enumerate(run["tokens"]):
    w = where(tok)
    flag = ""
    if w.startswith("NOT") and first is None:
        first, flag = pos, "   <-- first token not in the quote"
    print(f"{pos:>3} {tok!r:<16} {str(top5)[:84]:<84} {w}{flag}")

if first is None:
    print("\nevery token of three or more letters is in the quote or the question: this run is not a fabrication by text")
else:
    tok, top5 = run["tokens"][first]
    own = [p for t, p in top5 if t == tok]
    print(f"\nfirst token not in the quote: position {first}, {tok!r}")
    print("  probability on it: " + (f"{own[0]}" if own else f"below the fifth alternative ({top5[-1][1]} or less)"))
    right = [(t, p) for t, p in top5 if len(core(t)) >= 3 and core(t) not in STOP and (core(t) in answer or core(t) in quote)]
    print("  right answer in the top five there: " + (f"yes, {right}" if right else "no"))
    print("  the five: " + ", ".join(f"{t!r} {p}" for t, p in top5))
```

Run it on a fabricated run, with your id and run number from `marks.md` (an `f` in position 3 of `q07`'s API letters is `q07 3`):

```bash
uv run python failures/evidence.py q07 3
```

*You should see* the question, answer, quote and text, then one line per token of the answer: position, the token, its five alternatives with probabilities, and a `where` column that says `in quote`, `in question`, nothing (a short or common word) or `NOT IN QUOTE OR QUESTION`, with an arrow on the first of those. Under the table: the probability the model put on that first wrong token, whether any of the five alternatives there is the right answer, and the five themselves. **The flag is a pointer, not a verdict**: a right answer also contains words the quote does not (`The text says`), so the `contains the answer field` line tells you which kind you are looking at, and you read down to the token where the answer should have been. A probability near `1.0` on a wrong name with the right one absent from the five is what "confident fabrication" looks like in numbers; a wrong name at `0.4` with the right one at `0.3` beside it is a sampling story, and the same question at temperature 0 (Extension C) would tell you which.

<!-- JD: fill one Homer evidence.py tail (the four lines under the table) after a run. -->

One entry you cannot explain goes in bounded, not blank: under Hypothesis write `unknown`, then name the categories you ruled out and the evidence that ruled each one out. An empty Hypothesis field is not bounded; this is:

> Category: unknown. Ruled out: sampling (fails 5/5 at 0.7 and 5/5 at 0), memorization gap (the model quotes the surrounding passage correctly).

A hypothesis with a trace under it is an argument. Without one it is a guess, and I read it as a guess. Commit after every two or three entries; the log of when you filed what is part of the deliverable.

**Extension — one category, measured (CHOOSE)**

Pick one of your catalog's categories and go deep on it. Say in the deep-dive section of `CATALOG.md` which one and why that one. Each option has a number, a baseline and a comparison, and `deepdive.py` prints all three on lines that start `NUMBER`, `BASELINE` and `COMPARISON`. **Write your predicted number in the log before you run.**

| Option | Category | What the script does | NUMBER | BASELINE |
|---|---|---|---|---|
| **A** | Confident fabrication | Your five most-fabricated questions (from `marks.md`), five runs each, with the chunk of your corpus that holds the quote pasted into the prompt above the question: open-book. | API right/25 open-book | API right/25 closed-book on the same five, from your marks and by the rule |
| **B** | Retrieval or grounding | All twenty questions through your A06 `search()`; then the API asked each question five times with the top-1 chunk pasted in, whether or not it was the right chunk. | API right when the top-1 chunk held the quote vs when it did not | Top-3 hit rate of keyword search on the same twenty, and of semantic search |
| **C** | Sampling nondeterminism | The question whose five API marks were most split, twenty runs each at temperature 0, 0.7 and 1.2. | Right/20 and distinct texts/20 at each temperature | Temperature 0 |
| **D** | Loop or repetition | GPT-2 on all twenty `Q: ... A:` prompts, 60 new tokens, at temperature 0, at 0.7, and at 0.7 with `top_p=0.9`. An output loops when any three-word sequence appears three or more times. | Looping outputs/20 at each setting | Temperature 0 |

"Right" in the script is a rule, not a reading: an answer is right when the `answer` field of `questions.jsonl` appears in it, ignoring case. The script prints its per-question detail so you can overrule the rule by hand in `CATALOG.md` where it is wrong; say that you did. Option A prints the rule's count beside your hand marks on the same five questions, so you can see how far apart the two are.

Create `failures/deepdive.py` and paste this in:

```python
# failures/deepdive.py
# Run from the repo root:  uv run python failures/deepdive.py --option A        (A, B, C or D)
# One option, one category. Each prints NUMBER, BASELINE and COMPARISON lines for CATALOG.md, then the detail
# behind them, and saves every new answer in failures/deepdive_<option>.json so you can quote it.
# "right" is decided by a rule here: an answer is right when the `answer` field of questions.jsonl appears in it,
# ignoring case. Read the detail, overrule the rule by hand in CATALOG.md where it is wrong, and say that you did.
import json, math, os, sys, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sampling.lab import MODEL, generate            # A09's GPT-2 sampler, unchanged; D uses it

OPTION = sys.argv[sys.argv.index("--option") + 1].upper() if "--option" in sys.argv[:-1] else ""
assert OPTION in ("A", "B", "C", "D"), "usage: uv run python failures/deepdive.py --option A|B|C|D"
N = 5
SUFFIX = " Answer in one sentence."             # the same suffix ask.py adds, so the only change is the one the option makes
QS = {q["id"]: q for q in (json.loads(l) for l in Path("failures/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
TRACES = {i: json.loads(Path(f"failures/traces/{i}.json").read_text(encoding="utf-8")) for i in QS}
OUT = Path(f"failures/deepdive_{OPTION}.json")

def ask_api(content, temperature=0.7, max_tokens=60):
    """ ask.py's call, with the whole user message passed in """
    body = json.dumps({"model": MODEL, "temperature": temperature, "max_tokens": max_tokens,
                       "logprobs": True, "top_logprobs": 5,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        c = json.load(r)["choices"][0]
    tokens = [[t["token"], [[a["token"], round(math.exp(a["logprob"]), 3)] for a in t["top_logprobs"]]]
              for t in c["logprobs"]["content"]]
    return {"text": c["message"]["content"], "tokens": tokens}

def right(text, q):
    """ the rule: the answer field appears in the text, ignoring case and extra spaces """
    return " ".join(q["answer"].lower().split()) in " ".join(text.lower().split())

def api_marks():
    """ your API marks from failures/marks.md, {id: 'rfhfo'} """
    rows, inside = {}, False
    for line in Path("failures/marks.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Marks"
            continue
        if inside and line.startswith("| q"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1]
    assert all(len(m) == 5 and not set(m) - set("rfho") for m in rows.values()), "fill failures/marks.md first (Step 5)"
    return rows

def holds(chunk_text, quote):
    return " ".join(quote.split()) in chunk_text

def with_passage(passage, question):
    return "Here is a passage:\n\n" + passage + "\n\n" + question + SUFFIX

# ------------------------------------------------------------ A: confident fabrication, closed-book vs open-book
if OPTION == "A":
    from search.search import chunk, CORPUS
    chunks = chunk(CORPUS.read_text(encoding="utf-8"))
    marks = api_marks()
    ids = sorted(QS, key=lambda i: (-marks[i].count("f"), i))[:5]
    print("five most-fabricated questions, API fabricated/5 by your marks:", [(i, marks[i].count("f")) for i in ids])
    saved = {}
    for i in ids:
        held = [c for c in chunks if holds(c, QS[i]["quote"])]
        assert held, f"{i}: no chunk holds the quote; check_questions.py should have caught this"
        saved[i] = {"chunk": held[0], "runs": [ask_api(with_passage(held[0], QS[i]["question"])) for _ in range(N)]}
        print(" ", i, "asked", N, "times with its chunk pasted in,", len(held[0]), "characters of passage")
    OUT.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    open_right = sum(right(r["text"], QS[i]) for i in ids for r in saved[i]["runs"])
    closed_hand = sum(marks[i].count("r") for i in ids)
    closed_rule = sum(right(r["text"], QS[i]) for i in ids for r in TRACES[i]["api"])
    print(f"\nNUMBER      API right, open-book (the chunk holding the quote pasted above the question): {open_right}/25")
    print(f"BASELINE    API right, closed-book, the same five questions: {closed_hand}/25 by your marks, {closed_rule}/25 by the rule")
    print(f"COMPARISON  open-book minus closed-book (by the rule): {open_right - closed_rule:+d} of 25")
    print("\n| id | closed-book right/5 (marks) | closed-book right/5 (rule) | open-book right/5 (rule) |\n|---|---|---|---|")
    for i in ids:
        print(f"| {i} | {marks[i].count('r')} | {sum(right(r['text'], QS[i]) for r in TRACES[i]['api'])} | {sum(right(r['text'], QS[i]) for r in saved[i]['runs'])} |")
    print("\nstill wrong by the rule with the answer on the page (where it got worse):")
    for i in ids:
        for k, r in enumerate(saved[i]["runs"], 1):
            if not right(r["text"], QS[i]):
                print(f"  {i} run {k}: {' '.join(r['text'].split())[:140]}")

# ------------------------------------------------------------ B: retrieval or grounding, with the top-1 chunk pasted in
elif OPTION == "B":
    from search.search import chunk, CORPUS, VECS, normalize, search, keyword_search
    chunks = chunk(CORPUS.read_text(encoding="utf-8"))
    V = np.load(VECS)
    assert len(V) == len(chunks), "chunking changed since you embedded: delete search/chunks.npy and run search/search.py"
    Vn = normalize(V)
    saved, sem_hits, kw_hits, top1_held = {}, 0, 0, {}
    for i, q in QS.items():
        sem = search(q["question"], Vn, k=3)
        kw = keyword_search(q["question"], chunks, k=3)
        sem_hits += any(holds(chunks[j], q["quote"]) for j, _ in sem)
        kw_hits += any(holds(chunks[j], q["quote"]) for j, _ in kw)
        top1_held[i] = holds(chunks[sem[0][0]], q["quote"])
        saved[i] = {"top1": sem[0][0], "score": sem[0][1], "held": top1_held[i],
                    "runs": [ask_api(with_passage(chunks[sem[0][0]], q["question"])) for _ in range(N)]}
        print(f"  {i} top-1 chunk [{sem[0][0]}] score {sem[0][1]:.3f} {'holds' if top1_held[i] else 'does not hold'} the quote")
    OUT.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    held_ids = [i for i in QS if top1_held[i]]
    miss_ids = [i for i in QS if not top1_held[i]]
    r_held = sum(right(r["text"], QS[i]) for i in held_ids for r in saved[i]["runs"])
    r_miss = sum(right(r["text"], QS[i]) for i in miss_ids for r in saved[i]["runs"])
    c_held = sum(right(r["text"], QS[i]) for i in held_ids for r in TRACES[i]["api"])
    c_miss = sum(right(r["text"], QS[i]) for i in miss_ids for r in TRACES[i]["api"])
    print(f"\nNUMBER      API right with the top-1 chunk pasted in: {r_held}/{N * len(held_ids)} when it held the quote ({len(held_ids)} questions), "
          f"{r_miss}/{N * len(miss_ids)} when it did not ({len(miss_ids)} questions)")
    print(f"BASELINE    top-3 hit rate on the same twenty: keyword {kw_hits}/20, semantic {sem_hits}/20, semantic top-1 {len(held_ids)}/20")
    print(f"COMPARISON  closed-book right (rule) on the same split: {c_held}/{N * len(held_ids)} where top-1 held, {c_miss}/{N * len(miss_ids)} where it did not. "
          f"A wrong chunk made it worse than no chunk if {r_miss} < {c_miss}.")
    print("\n| id | top-1 held the quote | closed-book right/5 (rule) | with top-1 pasted right/5 (rule) |\n|---|---|---|---|")
    for i in QS:
        print(f"| {i} | {'yes' if top1_held[i] else 'no'} | {sum(right(r['text'], QS[i]) for r in TRACES[i]['api'])} | {sum(right(r['text'], QS[i]) for r in saved[i]['runs'])} |")

# ------------------------------------------------------------ C: sampling nondeterminism, the most split question at three temperatures
elif OPTION == "C":
    marks = api_marks()
    i = max(QS, key=lambda i: (len(set(marks[i])), -max(marks[i].count(m) for m in marks[i]), -int(i[1:])))
    print(f"most split question (API): {i}, marks {marks[i]}: {QS[i]['question']}\n  quote: {QS[i]['quote']}")
    saved = {}
    for T in (0, 0.7, 1.2):
        saved[str(T)] = [ask_api(QS[i]["question"] + SUFFIX, temperature=T) for _ in range(20)]
        print(f"  T={T}: 20 runs done")
    OUT.write_text(json.dumps({"id": i, "runs": saved}, indent=1), encoding="utf-8")
    rows = {T: (sum(right(r["text"], QS[i]) for r in saved[str(T)]), len({r["text"] for r in saved[str(T)]})) for T in (0, 0.7, 1.2)}
    print("\nNUMBER      right/20 (rule): " + ", ".join(f"T={T} {rows[T][0]}/20" for T in rows) + "   distinct texts/20: " + ", ".join(f"T={T} {rows[T][1]}" for T in rows))
    print(f"BASELINE    T=0: right {rows[0][0]}/20, distinct {rows[0][1]}/20")
    most_right = max(rows, key=lambda T: rows[T][0])
    most_consistent = min(rows, key=lambda T: rows[T][1])
    print(f"COMPARISON  most right: T={most_right}; most consistent: T={most_consistent}. "
          f"Temperature 0 was {'the most right' if most_right == 0 else 'not the most right'} and {'the most consistent' if most_consistent == 0 else 'not the most consistent'}.")
    for T in (0, 0.7, 1.2):
        print(f"\nT={T}, distinct texts with how often each came up:")
        for text, n in Counter(" ".join(r["text"].split()) for r in saved[str(T)]).most_common():
            print(f"  {n:>2}x {'right' if right(text, QS[i]) else 'wrong'}  {text[:120]}")

# ------------------------------------------------------------ D: loop or repetition, GPT-2 at three settings
elif OPTION == "D":
    SETTINGS = [("T=0", dict(temperature=0)), ("T=0.7", dict(temperature=0.7)), ("T=0.7 top_p=0.9", dict(temperature=0.7, top_p=0.9))]

    def loops(text):
        """ True when any three-word sequence appears three or more times """
        w = text.split()
        if len(w) < 3:
            return False
        return Counter(zip(w, w[1:], w[2:])).most_common(1)[0][1] >= 3

    saved = {}
    for name, knobs in SETTINGS:
        saved[name] = {i: generate(f"Q: {QS[i]['question']}\nA:", n_new=60, seed=0, **knobs) for i in QS}
        print(f"  {name}: 20 generations of 60 tokens done")
    OUT.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    rows = {name: (sum(loops(t) for t in saved[name].values()), sum(right(t, QS[i]) for i, t in saved[name].items())) for name, _ in SETTINGS}
    print("\nNUMBER      looping outputs/20: " + ", ".join(f"{name} {rows[name][0]}/20" for name in rows))
    print(f"BASELINE    T=0: {rows['T=0'][0]}/20 looping, {rows['T=0'][1]}/20 right (rule)")
    least = min(rows, key=lambda n: rows[n][0])
    worst = min(rows, key=lambda n: rows[n][1])
    print(f"COMPARISON  right/20 (rule): " + ", ".join(f"{name} {rows[name][1]}/20" for name in rows)
          + f". Looped least: {least}; answered worst: {worst}; {'the same setting' if least == worst else 'different settings'}.")
    print("\n| id | " + " | ".join(f"{name} loops" for name, _ in SETTINGS) + " |\n|---|---|---|---|")
    for i in QS:
        print(f"| {i} | " + " | ".join("yes" if loops(saved[name][i]) else "no" for name, _ in SETTINGS) + " |")
    for name, _ in SETTINGS:
        ex = [i for i in QS if loops(saved[name][i])]
        if ex:
            print(f"\nfirst looping output at {name}, {ex[0]}: {saved[name][ex[0]][:200]!r}")
```

Write the prediction in your log, then run your option, with its letter in place of `A`:

```bash
uv run python failures/deepdive.py --option A > failures/deepdive.txt
cat failures/deepdive.txt
```

*You should see*, for every option, progress lines, then `NUMBER`, `BASELINE` and `COMPARISON`, then a per-question table and the detail. What each option costs and prints:

- **A**: 25 API calls. `NUMBER` is open-book right out of 25; `BASELINE` is closed-book right on the same five questions, by your marks and by the rule; `COMPARISON` is the difference. Then the five rows, and every open-book run still wrong by the rule, pasted: those are where it got worse, the questions wrong with the answer on the page.
- **B**: 20 embedding calls through `search()` and 100 API calls. `NUMBER` is right-with-chunk split by whether the top-1 chunk held the quote; `BASELINE` is the top-3 hit rates of keyword and semantic search on your twenty, with the semantic top-1 rate; `COMPARISON` is closed-book right on the same split, and the line tells you which inequality means a wrong chunk made it worse than no chunk. Needs `search/chunks.npy` from A06 on this branch.
- **C**: 60 API calls on one question. `NUMBER` is right/20 and distinct texts/20 at each temperature; `BASELINE` is the temperature-0 row; `COMPARISON` names the most right and the most consistent temperature, which may not be the same one. Then every distinct text at each temperature with how often it came up.
- **D**: no API; 60 GPT-2 generations of 60 tokens, several minutes on a laptop CPU. `NUMBER` is looping outputs/20 at each setting; `BASELINE` is temperature 0; `COMPARISON` is right/20 at each setting and whether the setting that looped least also answered worst. Then a yes/no table per question and the first looping output at each setting.

<!-- JD: fill the Homer NUMBER/BASELINE/COMPARISON lines for whichever option you run, after a run. -->

*If it broke* with `fill failures/marks.md first`, Step 5 is not done; A and C read your marks. `FileNotFoundError: search/chunks.npy` on B means the A06 vectors are not on this machine; `uv run python search/search.py` once rebuilds them. `400` naming `temperature` on C means `MODEL` is a reasoning model.

Then fill the deep-dive section at the end of `CATALOG.md`: the three lines pasted, the per-question detail that answers "where did it get worse" for your option (A: the questions still wrong with the answer on the page; B: whether a wrong chunk made the answer worse than no chunk; C: whether temperature 0 was the most right or only the most consistent; D: whether the setting that looped least also answered worst), and the falsifier.

**Commit, push, PR, sign off.**

```bash
git add failures/
git commit -m "A10: ten catalog entries with rates, deep dive on <category>"
git push -u origin dev/failure-catalog
git log --oneline failures/
```

*You should see* several commits, not one, with the questions commit before the traces and the traces before the marks.

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the hypothesis you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`failures/CATALOG.md` (ten entries plus the deep dive) + `failures/CATALOG_TEMPLATE.md` + `failures/questions.jsonl` + `failures/check_questions.py` + `failures/ask.py` + `failures/traces/` + `failures/mark.py` + `failures/to_mark.txt` + `failures/marks.md` (letters and the counts table) + `failures/evidence.py` + `failures/deepdive.py` + `failures/deepdive.txt` + `failures/deepdive_<option>.json`.

**Reflection Questions**

1. Your most-fabricated question: paste it, the quote from your corpus, and two of the fabricated answers. From its trace, give the probability on the first wrong token of one of them and say whether the right token was anywhere in the top five at that position. What does that number say about how "confident" the model was?

   *How to get it:* the question is the `most-fabricated question (API)` line that `uv run python failures/mark.py counts` printed, with its id. The two answers are `api` lines marked `f` for that id in `failures/to_mark.txt`. For the probability, run `evidence.py` on one of those runs, with your id and the position of an `f` in that row of `marks.md`:

   ```bash
   uv run python failures/evidence.py q05 2
   ```

   The four lines under the token table are the answer: `probability on it` is the number, `right answer in the top five there` is the yes or no. Then read the number against the two shapes in Step 6: near `1.0` with the right token absent, or a split between two names.

2. Paste `git log --oneline failures/`. Name the entry you first filed under the wrong category and what moved it. Then name one of the two failures you caused, and the line of `ask.py` or `questions.jsonl` responsible.

   *How to get it:* the log is the command as written, from the repo root. The re-filed entry is one whose category changed between two commits:

   ```bash
   git log -p --follow -- failures/CATALOG.md | grep "^[-+].*Category"
   ```

   prints every `Category` line that was added or removed, in order; a `-` line and a `+` line for the same entry are the move. The usual self-caused failures are the `Answer in one sentence.` suffix in `ask_api` (it pushes the model away from "the text does not say", and your `hedged` column measures how far) and a question that names the answer's category in its wording ("What false name..."), so the model has a shape to fill; the line is `grep -n "one sentence" failures/ask.py`, or the question's `id`. Check the second claim against `check_questions.py`'s second line: a question that shares content words with its quote is one you helped.

3. Your extension: the number, the baseline and the comparison, with the prediction you wrote in the log before running. Where did it get worse? State the result that would falsify your catalog hypothesis for that category, and whether anything in your deep dive came close.

   *How to get it:* the prediction is in your log entry (`grep -n -i predict logs/*.md` in the log repo); the number, baseline and comparison are the three lines in `failures/deepdive.txt` that start with those words:

   ```bash
   grep -n -e "^NUMBER" -e "^BASELINE" -e "^COMPARISON" failures/deepdive.txt
   ```

   The where-it-got-worse detail is the table and the lines under it in the same file. The falsifier is the result that would have come out the other way if your hypothesis were wrong: for A, open-book no better than closed-book; for B, right answers independent of whether the top-1 chunk held the quote; for C, temperature 0 no more right than 1.2; for D, the setting that looped least also the most right.
