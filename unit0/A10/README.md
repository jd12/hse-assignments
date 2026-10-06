# A10 · Failure Catalog: Why Models Lie

**Meetings:** D20–D21 · **Points:** 15 pts

**Watch — none**

**Day 1 — none.** Write the questions, commit them, and run your A09 sampler and the API on them.

**Day 2 — none.** Mark the answers and write the catalog.

**During the video**

No video. You are producing the evidence this time. Have three things open: `data/corpus.txt` in VS Code with search (you will be quoting it), `sampling/lab.py` from A09, and a terminal in the repo root.

**Notes**

**The corpus is the answer key.** Every question you write has its answer in your file, with the sentence that proves it. When a model answers without seeing the file, you check it against the text rather than against your memory or the internet. That is what turns "it made something up" into a catalog entry, and it is why the rate of made-up answers is a number you can measure.

**Famous corpora are partly memorized.** If your corpus is a well-known public text, the API model has probably read it. It will get the famous facts and invent the obscure ones. That is a finding, not a problem, and it is why a third of your questions are about details nobody quotes.

**`Answer in one sentence.` is part of the experiment.** `ask.py` appends it to every question. Whatever it does to the model's willingness to say "I don't know" is something you did. Do not change it between runs you compare.

**Walkthrough — questions your corpus can answer, asked of models that have not seen it**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. A09 is due tomorrow morning and will not have merged, and today uses its sampler, so branch from it:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch dev/sampling && git pull    # or: git switch main && git pull, if A09 has merged
git switch -c dev/failure-catalog
mkdir -p failures/traces
```

Step 2 of A05b's template update, or the pull request I opened on your repo, put `CATALOG_TEMPLATE.md` (ten entry skeletons with the seven fields, one hint per field in angle brackets, and the deep-dive section), `mark.py`, `evidence.py` and `deepdive.py` in `failures/`; `ls failures` should show them, and if it does not, `git merge main` brings them onto this branch.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist. Two meetings, one branch: push again each day, one PR, not two, one log entry per meeting.

```markdown
- [ ] Day 1: catalog opened from the template; 20 questions with quotes, checker says 0 not found; both committed
- [ ] Day 1: ask.py run, 5 API + 5 GPT-2 answers per question, traces committed
- [ ] Day 2: marks.md filled, mark.py counts pasted; ten catalog entries with rates, categories, hypotheses, evidence.py output in three
- [ ] Extension: option chosen, prediction in the log, deepdive.py run, deep-dive section written
- [ ] Push and open the PR
```

**Step 2. Open the catalog from the template.** You fill the copy, not the template, and delete each hint as you fill it.

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
cp failures/CATALOG_TEMPLATE.md failures/CATALOG.md
git add failures/CATALOG.md
git commit -m "A10: open the catalog from the template"
```

**Step 3. Twenty questions, each with its proof.**

Create `failures/questions.jsonl`, one question per line, each line in exactly this shape:

```json
{"id": "q01", "question": "What false name does Ulysses give the Cyclops?", "answer": "Noman", "quote": "my name is Noman"}
```

That is an Odyssey example. Yours come from your corpus. Each answer is one checkable fact: a name, a number, an object, who said what to whom. **At least seven are about minor details**: a fact the text states once, in passing, that no summary includes; on Homer, Euryclea's father is Ops, one half-line in Book I. **At least three have an answer a reasonable guess would get wrong**: the obvious answer is one the text contradicts; on Homer, how many years Aegisthus ruled Mycene after killing Agamemnon; the story's famous tens make a reader guess ten, and the text says seven.

Four steps per fact:

1. Search for a proper noun, a place or a term you know is in the corpus and pick one line number from the output: `grep -n -i "noman" data/corpus.txt | head`. On Homer this printed six lines, the first `4082:therefore, the present you promised me; my name is Noman; this is what`. A name that prints one to three lines is a minor detail; a name that prints forty (`grep -c -i "euryclea" data/corpus.txt` says 44) is a main character, and the fact you want is on one of those lines, not the name itself.
2. Read the paragraph around that line, about eight lines either side: `sed -n '4074,4090p' data/corpus.txt`.
3. Write the question that paragraph answers, with the answer as a short field. The Homer paragraph says "my name is Noman", so the question is "What false name does Ulysses give the Cyclops?" and the answer is `Noman`. Do not put the answer in the question.
4. Copy the quote: four to twelve words from the paragraph, **exactly as the file has them**, select-and-copy, never retyped. The quote has to contain the answer, or say it in other words (`bound Ino's veil under his arms` for the answer `her veil`).

Then check every line. Create `failures/check_questions.py` and paste this in:

```python
# failures/check_questions.py
# Run from the repo root:  uv run python failures/check_questions.py
# Checks that every quote in failures/questions.jsonl is in data/corpus.txt word for word, then prints three
# counts that say how hard your questions are. There is nothing to edit in this file.
import json
from pathlib import Path

# words too common to count as shared between a question and its quote
STOP = {"the", "a", "an", "of", "and", "to", "in", "is", "was", "what", "who", "how", "why", "does", "did", "which", "whose",
        "where", "when", "do", "he", "she", "it", "they", "his", "her", "its", "their", "them", "him", "that", "this", "for",
        "from", "by", "with", "on", "at", "as", "be", "are", "were", "had", "has", "have"}

# In: a string.  Out: the set of its distinct words, lowercased, with punctuation and STOP words removed.
def words(s):
    cleaned = ""
    for ch in s.lower():
        if ch.isalnum() or ch == "'":      # keep letters, digits and apostrophes
            cleaned += ch
        else:
            cleaned += " "                 # everything else (commas, periods, quotes) becomes a space
    found = set()
    for w in cleaned.split():
        if w.endswith("'s"):
            w = w[:-2]                     # Euryclea's -> euryclea
        found.add(w)
    return found - STOP                    # - removes every STOP word from the set

text = " ".join(Path("data/corpus.txt").read_text(encoding="utf-8").split())   # the whole corpus as one line, single spaces
qs = []                                    # one dict per question, straight from the JSON lines
for line in Path("failures/questions.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():                       # skip blank lines
        qs.append(json.loads(line))

missing, ids, no_shared, short, answer_not_in_quote = [], [], [], [], []   # lists of question ids, one per check
for q in qs:
    ids.append(q["id"])
    if " ".join(q["quote"].split()) not in text:          # the quote, with its spacing collapsed the same way as the corpus
        missing.append(q["id"])
    if not (words(q["question"]) & words(q["quote"])):     # & keeps the words both sets share; empty means none
        no_shared.append(q["id"])
    if len(q["quote"]) < 15:
        short.append(q["id"])
    if " ".join(q["answer"].lower().split()) not in " ".join(q["quote"].lower().split()):
        answer_not_in_quote.append(q["id"])
print(len(qs), "questions,", len(missing), "quotes not found:", missing)
if len(set(ids)) < len(ids):               # set(ids) drops duplicates, so a smaller set means an id was used twice
    print("some ids are used more than once; make every id different")
print(len(no_shared), "questions share no content words with their quote:", no_shared)
print(len(short), "quotes under 15 characters:", short)
print(len(answer_not_in_quote), "answers not in their quote word for word:", answer_not_in_quote)
```

```bash
uv run python failures/check_questions.py
```

*You should see* four lines. The first must be `20 questions, 0 quotes not found: []`; fix every id it lists, and the usual cause is a curly quote or an em dash you retyped as a straight one. The second counts questions that share no content word with their quote: the ones a model cannot answer by echoing your wording. The third counts quotes under 15 characters; make those longer, because a short quote can sit in many places in the file. The fourth lists answers that are not in their quote word for word; that is fine when the quote says the same thing in other words. Then commit, before any model sees a question. On Homer:

```text
20 questions, 0 quotes not found: []
1 questions share no content words with their quote: ['q11']
0 quotes under 15 characters: []
3 answers not in their quote word for word: ['q09', 'q12', 'q18']
```

```bash
git add failures/questions.jsonl failures/check_questions.py
git commit -m "A10: 20 corpus questions with quotes, before asking any model"
```

I will check your commit timestamps.

**Step 4. Ask both models, five times each.**

Create `failures/ask.py` and paste this in. Read `ask_api()` once: it is what you are sending, and the suffix is yours.

```python
# failures/ask.py
# Run from the repo root:  uv run python failures/ask.py
# Asks every question in failures/questions.jsonl five times of the API (closed-book: the question alone, with
# "Answer in one sentence." added) and five times of GPT-2 through your A09 sampler; saves the answers to
# failures/traces/<id>.json, skipping any question that already has a trace. There is nothing to edit in this file.
import json, math, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the repo root, so "from sampling.lab" works
from sampling.lab import MODEL, generate                           # A09's model name and your NumPy sampler, unchanged

N, TEMP = 5, 0.7            # five runs per question per model, both at temperature 0.7

# In: a question.  Out: a dict with the API's answer text and, at every token, its five most likely alternatives.
def ask_api(question):
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": 60,
                       "logprobs": True, "top_logprobs": 5,                           # the five most likely tokens at every position
                       "messages": [{"role": "user", "content": question + " Answer in one sentence."}]}).encode()   # the suffix is part of the experiment
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:                                            # send it and wait for the reply
        c = json.load(r)["choices"][0]
    tokens = []                                   # one entry per token of the answer: [the token, [[alternative, probability] x 5]]
    for t in c["logprobs"]["content"]:
        alternatives = []
        for a in t["top_logprobs"]:
            alternatives.append([a["token"], round(math.exp(a["logprob"]), 3)])      # the API gives log probabilities; exp turns them back into probabilities
        tokens.append([t["token"], alternatives])
    return {"text": c["message"]["content"], "tokens": tokens}

for line in Path("failures/questions.jsonl").read_text(encoding="utf-8").splitlines():
    q = json.loads(line)
    if Path(f"failures/traces/{q['id']}.json").exists():   # already asked: skip, so a rerun costs nothing
        continue
    q["api"] = []
    for _ in range(N):
        q["api"].append(ask_api(q["question"]))
    q["gpt2"] = []
    for s in range(N):                                     # seeds 0 to 4, so the five GPT-2 runs differ
        answer = generate(f"Q: {q['question']}\nA:", seed=s, temperature=TEMP)
        q["gpt2"].append(answer.split("\n")[0])            # keep only the first line: GPT-2 often goes on to invent a next question
    Path(f"failures/traces/{q['id']}.json").write_text(json.dumps(q, indent=1), encoding="utf-8")
    print(q["id"], "|", q["api"][0]["text"][:60], "|", q["gpt2"][0][:40])   # a one-line preview: the first 60 and 40 characters
```

```bash
uv run python failures/ask.py
```

*You should see* twenty lines, one per question: the id, the first sixty characters of the first API answer, the first forty of the first GPT-2 answer; and twenty files in `failures/traces/`. It is 100 API calls of 60 tokens and 100 GPT-2 generations, a few minutes in all. **The API is closed-book**: the question alone, no passage. Each trace holds the question, answer and quote, then `api`, five `{"text", "tokens"}` objects where `tokens` is `[token, [[alternative, probability] × 5]]` for every position, and `gpt2`, five strings. *If it broke* partway through, run the same command again: it skips every question that already has a trace, so nothing is paid for twice. `HTTP Error 400` naming `logprobs` means `MODEL` is a reasoning model, as in A09. Commit the traces now, before you have read them closely. <!-- JD: fill one Homer line of the twenty after a run. -->

```bash
git add failures/ask.py failures/traces/
git commit -m "A10: 5 API + 5 GPT-2 answers per question, unread"
```

**Step 5. Mark every answer (Day 2).**

Every question ran five times on each model, because "it said Athens" is a story and "4 of 5 runs said Athens; the file says Sparta" is an entry. Each of the two hundred answers gets one of four letters: **`r` right**, states the fact in the quote; **`f` fabricated**, states a specific different fact, as fact; **`h` hedged**, says it does not know, or that the text does not say; **`o` off**, answers some other question, or produces no answer at all.

```bash
uv run python failures/mark.py > failures/to_mark.txt
wc -l failures/to_mark.txt
```

The script prints every answer beside its quote with a blank `[ ]`, and writes `failures/marks.md` with one row per question for you to fill. *You should see* `302 failures/to_mark.txt`: twenty blocks of fifteen lines (a rule, the question, the answer field, the quote, five `api` lines, five `gpt2` lines), then `wrote failures/marks.md with 20 rows to fill`. Open `to_mark.txt` beside `marks.md`. For each answer line, read the answer against the quote above it, decide `r`, `f`, `h` or `o`, and type that letter into the matching position of the row in `marks.md`: the first API letter is `api 1`, the fifth GPT-2 letter is `gpt2 5`, so `.....` becomes something like `rrfrf`. A GPT-2 answer that is a question, a blank, or another topic is `o`; a loop of the same words is `o` too, unless the words state a different fact. When the dots are gone:

```bash
uv run python failures/mark.py counts
```

*You should see* a twenty-row table of counts (`off` is whatever is left of five, so it has no column), a totals row, the most-fabricated question with its quote, the most split question, and how many questions the API got 5/5 while GPT-2 got 0/5. Whatever your corpus, **expect GPT-2 right on almost nothing**: 124 million parameters and no corpus. The API will be right on famous facts and fabricating on minor ones, with some questions split across the five runs; the most split one is what Extension C runs. *If it broke* with `AssertionError: these rows are not five letters`, a row still has a dot or a letter outside `r f h o`; the message names the ids. Paste the table and the lines under it into `marks.md` under `## Counts`, and commit: <!-- JD: fill the Homer totals row and the most-fabricated id after a run. -->

```bash
git add failures/marks.md failures/to_mark.txt
git commit -m "A10: 200 answers marked"
```

**Step 6. Write ten entries.**

Each entry in `CATALOG.md` has the seven fields the template gives, in this order: **Repro** (the question verbatim, the model, every sampling setting, and the command that reruns it), **Observed** (the answer, pasted from the trace, with the run named), **Expected** (the quote from your corpus, and where it is in the file), **Rate** (failures over runs, with N: `4/5 API runs said X`), **Category** (one of the eight below, or one you name), **Hypothesis** (a claim about machinery, tied to something from A03 to A09), **Evidence** (logprobs at the failure position, the tokenizer split, a diff across runs, or a token count). The eight categories: tokenization artifact · context window or truncation · sampling nondeterminism · retrieval or grounding failure · instruction-following collapse · tool-schema mismatch · confident fabrication · loop or repetition.

At least six entries come from your twenty questions; the other four can come from anything you ran this term (your A06 search on these questions, the A09 table, `cl100k_base` splitting a name in one of your answers). At least four show the same question on both models. At least two are failures you caused, where the model did what it was told: look at the `Answer in one sentence.` suffix and the wording of your questions. At least three carry evidence from the logprobs, which `evidence.py` prints. Run it on a fabricated run, with your id and run number from `marks.md` (an `f` in position 3 of `q07`'s API letters is `q07 3`):

```bash
uv run python failures/evidence.py q07 3
```

The script prints that answer token by token with the top-5 alternatives at every position, flags the first token that is in neither the quote nor the question, and says whether the right answer was in the top five there. *You should see* the question, answer, quote and text, then one line per token: position, token, its five alternatives with probabilities, and a `where` column that reads `in quote`, `in question`, nothing (a short or common word) or `NOT IN QUOTE OR QUESTION`, with an arrow on the first of those. Under the table: the probability the model put on that first wrong token, whether any of the five alternatives there is the right answer, and the five themselves. **The flag is a pointer, not a verdict**: a right answer also contains words the quote does not, so the `contains the answer field` line tells you which kind you are looking at. **A probability near `1.0` on a wrong name with the right one absent from the five is what confident fabrication looks like in numbers**; a wrong name at `0.4` with the right one at `0.3` beside it is a sampling story, and the same question at temperature 0 (Extension C) would tell you which. <!-- JD: fill one Homer evidence.py tail (the four lines under the table) after a run. -->

One entry you cannot explain goes in bounded, not blank: under Hypothesis write `unknown`, then name the categories you ruled out and the evidence that ruled each one out. An empty Hypothesis field is not bounded; this is: `Category: unknown. Ruled out: sampling (fails 5/5 at 0.7 and 5/5 at 0), memorization gap (the model quotes the surrounding passage correctly).` A hypothesis with a trace under it is an argument. Without one it is a guess, and I read it as a guess. Commit after every two or three entries: `git log --oneline failures/` shows the order you found things in, and I read it.

**Extension — one category, measured (CHOOSE)**

Pick one of your catalog's categories and go deep on it; say in the deep-dive section of `CATALOG.md` which one and why. `failures/deepdive.py` has one function per option, each headed by a comment that says its number, baseline and comparison in words, and each prints them on lines that start `NUMBER`, `BASELINE` and `COMPARISON`. "Right" in the script is a rule, not a reading: the `answer` field appears in the text, ignoring case; the script prints its per-question detail so you can overrule the rule by hand in `CATALOG.md`, and say that you did.

| Option | Category | What the script does | NUMBER | BASELINE | COMPARISON |
|---|---|---|---|---|---|
| **A** | Confident fabrication | Your five most-fabricated questions (from `marks.md`), five runs each, with the chunk of your corpus that holds the quote pasted above the question: open-book. 25 API calls. | API right/25 open-book | API right/25 closed-book on the same five, by your marks and by the rule | Open-book minus closed-book, then every run still wrong with the answer on the page |
| **B** | Retrieval or grounding | All twenty questions through your A06 `search()`; then the API asked each five times with the top-1 chunk pasted in, whether or not it was the right chunk. 100 API calls; needs `search/chunks.npy`. | API right when the top-1 chunk held the quote vs when it did not | Top-3 hit rate of keyword and of semantic search on the same twenty | Closed-book right on the same split: whether a wrong chunk made it worse than no chunk |
| **C** | Sampling nondeterminism | The question whose five API marks were most split, twenty runs each at temperature 0, 0.7 and 1.2. 60 API calls. | Right/20 and distinct texts/20 at each temperature | The temperature 0 row | The most right temperature vs the most consistent one: was temperature 0 both, or only consistent |
| **D** | Loop or repetition | GPT-2 on all twenty `Q: ... A:` prompts, 60 new tokens, at temperature 0, at 0.7, and at 0.7 with `top_p=0.9`. An output loops when any three-word sequence appears three or more times. No API; several minutes on a CPU. | Looping outputs/20 at each setting | The temperature 0 row | Right/20 at each setting: whether the setting that looped least also answered worst |

**Write your predicted number in the log before you run.** Then, with your letter in place of `A`:

```bash
uv run python failures/deepdive.py --option A | tee failures/deepdive.txt
```

*You should see* progress lines, then `NUMBER`, `BASELINE` and `COMPARISON`, then a per-question table and the detail behind the comparison. *If it broke* with `fill failures/marks.md first`, Step 5 is not done; A and C read your marks. `FileNotFoundError: search/chunks.npy` on B means the A06 vectors are not on this machine; `uv run python search/search.py` once rebuilds them. `400` naming `temperature` on C means `MODEL` is a reasoning model. Then fill the deep-dive section at the end of `CATALOG.md`: the three lines pasted, where it got worse (the detail under the comparison), and the falsifier. <!-- JD: fill the Homer NUMBER/BASELINE/COMPARISON lines for whichever option you run, after a run. -->

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
`failures/CATALOG.md` (ten entries plus the deep dive) + `failures/questions.jsonl` + `failures/check_questions.py` + `failures/ask.py` + `failures/traces/` + `failures/to_mark.txt` + `failures/marks.md` (letters and the counts table) + `failures/deepdive.txt` + `failures/deepdive_<option>.json`.

**Reflection Questions**

1. Your most-fabricated question: paste it, the quote from your corpus, and two of the fabricated answers. From its trace, give the probability on the first wrong token of one of them and say whether the right token was anywhere in the top five at that position. What does that number say about how "confident" the model was?

   *How to get it:* the question is the `most-fabricated question (API)` line that `mark.py counts` printed, with its id; the two answers are `api` lines marked `f` for that id in `failures/to_mark.txt`. For the probability, run `uv run python failures/evidence.py q05 2` with your id and the position of an `f` in that row of `marks.md`: the four lines under the token table are the answer, `probability on it` is the number and `right answer in the top five there` is the yes or no. Then read the number against the two shapes in Step 6: near `1.0` with the right token absent, or a split between two names.

2. Paste `git log --oneline failures/`. Name the entry you first filed under the wrong category and what moved it. Then name one of the two failures you caused, and the line of `ask.py` or `questions.jsonl` responsible.

   *How to get it:* the log is the command as written, from the repo root. `git log -p --follow -- failures/CATALOG.md | grep "^[-+].*Category"` prints every `Category` line that was added or removed, in order; a `-` line and a `+` line for the same entry are the move. The usual self-caused failures are the `Answer in one sentence.` suffix in `ask_api` (it pushes the model away from "the text does not say", and your `hedged` column measures how far; `grep -n "one sentence" failures/ask.py` prints the comment and the code line; the second is the line) and a question that names the answer's category in its wording ("What false name...") so the model has a shape to fill; `check_questions.py`'s second line counts the questions you did not help that way.

3. Your extension: the number, the baseline and the comparison, with the prediction you wrote in the log before running. Where did it get worse? State the result that would falsify your catalog hypothesis for that category, and whether anything in your deep dive came close.

   *How to get it:* the prediction is in your log entry (`grep -n -i predict logs/*.md` in the log repo); the three lines are `grep -n -e "^NUMBER" -e "^BASELINE" -e "^COMPARISON" failures/deepdive.txt`, and the where-it-got-worse detail is the table and the lines under it in the same file. The falsifier is the result that would have come out the other way if your hypothesis were wrong: for A, open-book no better than closed-book; for B, right answers independent of whether the top-1 chunk held the quote; for C, temperature 0 no more right than 1.2; for D, the setting that looped least also the most right.
