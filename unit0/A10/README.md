# A10 · Failure Catalog: Why Models Lie

**Meetings:** D20 · **Points:** 15 pts

**Watch — none**

Have three things open: `data/corpus.txt` in VS Code with search (you will be quoting it), `sampling/lab.py` from A09, and a terminal in the repo root.

**Notes**

**The corpus is the answer key.** Every question you write has its answer in your file, with the sentence that proves it. When a model answers without seeing the file, you check it against the text, not against your memory or the internet. That is what turns "it made something up" into a number.

**Famous corpora are partly memorized.** If your corpus is a well-known public text, the API model has probably read it. It will get the famous facts and invent the obscure ones. That is a finding, not a problem, and it is why two of your six questions are about details nobody quotes.

**`Answer in one sentence.` is part of the experiment.** `ask.py` appends it to every question. Whatever it does to the model's willingness to say "I don't know" is something you did. Do not change it between runs you compare.

**Walkthrough — six questions your corpus can answer, asked of two models that have not seen it**

**Step 1. Merge, branch, log.**
If A08's pull request is still open and I have approved it, merge it on GitHub and click **Delete branch**. Do not merge A09's: today uses its sampler, so branch from it (if A09 has already merged, use the `main` form of the first line):

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch dev/sampling && git pull    # or: git switch main && git pull, if A09 has merged
git switch -c dev/failure-catalog
```

The template-update pull request on your repo put `mark.py`, `evidence.py` and `deepdive.py` in `failures/`, and `evidence/A10.md`; `ls failures evidence` should show them, and if it does not, `git merge main` brings them onto this branch. The `README.md` files already in `failures/` and `failures/traces/` are from an earlier version of this assignment; this text is the one to follow.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] 6 questions with quotes, checker says 0 not found, its four lines in evidence/A10.md; committed
- [ ] ask.py run, 5 API + 5 GPT-2 answers per question, traces committed
- [ ] 60 marks filled, mark.py counts pasted; three entries, evidence.py output in the first
- [ ] Extension: prediction in the log, deepdive.py --option A run, the three lines pasted
- [ ] Reflection: three answers; check_evidence.py says all slots filled; push and open the PR
```

**Step 2. Six questions, each with its proof.**

In the foundations repo, create `failures/questions.jsonl`, one question per line, ids `q01` to `q06`, each line in exactly this shape:

```json
{"id": "q01", "question": "What false name does Ulysses give the Cyclops?", "answer": "Noman", "quote": "my name is Noman"}
```

That is an Odyssey example. Yours come from your corpus. Each answer is one checkable fact: a name, a number, an object, who said what to whom. **Two are about minor details**: a fact the text states once or twice, in passing, that no summary includes; on Homer, Euryclea's father is Ops, two half-lines in Books I and II. **One has an answer a reasonable guess would get wrong**: on Homer, how many years Aegisthus ruled Mycene after killing Agamemnon; the story's famous tens make a reader guess ten, and the text says seven. The other three can be anything the text settles.

Four steps per fact:

1. Search for a name, a place or a term you know is in the corpus and pick one line number from the output: `grep -n -i "noman" data/corpus.txt | head`. On Homer this printed six lines, the first `4082:therefore, the present you promised me; my name is Noman; this is what`. A name that prints one to three lines is a minor detail; a name that prints forty (`grep -c -i "euryclea" data/corpus.txt` says 44) is a main character, and the fact you want is on one of those lines, not the name itself.
2. Read the paragraph around that line, about eight lines either side: `sed -n '4074,4090p' data/corpus.txt`.
3. Write the question that paragraph answers, with the answer as a short field. Do not put the answer in the question.
4. Copy the quote: four to twelve words from one paragraph, **exactly as the file has them**, select-and-copy, never retyped. The quote has to contain the answer, or say it in other words (`bound Ino's veil under his arms` for the answer `her veil`).

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

*You should see* four lines. The first must be `6 questions, 0 quotes not found: []`; fix every id it lists, and the usual cause is a curly quote or an em dash you retyped as a straight one. The second counts questions that share no content word with their quote: the ones a model cannot answer by echoing your wording. The third counts quotes under 15 characters; make those longer, because a short quote can sit in many places in the file. The fourth lists answers that are not in their quote word for word; that is fine when the quote says the same thing in other words. Open `evidence/A10.md`, change `Questions: 20` on its first line to `Questions: 6`, and paste the four lines under `## Step 2`. Then commit, before any model sees a question:

```bash
git add failures/questions.jsonl failures/check_questions.py evidence/A10.md
git commit -m "A10: 6 corpus questions with quotes, before asking any model"
```

I will check your commit timestamps.

**Step 3. Ask both models, five times each.**

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

*You should see* a `Loading weights` bar, then six lines, one per question: the id, the first sixty characters of the first API answer, the first forty of the first GPT-2 answer; and six `q*.json` files in `failures/traces/`. On Homer the first line ends `| He's a very dark name, because he has t`, which is GPT-2 answering the Noman question. It is 30 API calls of 60 tokens and 30 GPT-2 generations, about three minutes on a laptop CPU; the GPT-2 half of each question prints nothing until its five runs are done. **The API is closed-book**: the question alone, no passage. *If it broke* partway through, run the same command again: it skips every question that already has a trace, so nothing is paid for twice. `HTTP Error 400` naming `logprobs` means `MODEL` is a reasoning model, as in A09. Commit the traces now, before you have read them closely:

```bash
git add failures/ask.py failures/traces/
git commit -m "A10: 5 API + 5 GPT-2 answers per question, unread"
```

**Step 4. Mark all sixty answers.**

Every question ran five times on each model, because "it said Athens" is a story and "4 of 5 runs said Athens; the file says Sparta" is an entry. Each answer gets one of four letters: **`r` right**, states the fact in the quote; **`f` fabricated**, states a specific different fact, as fact; **`h` hedged**, says it does not know, or that the text does not say; **`o` off**, answers some other question, or produces no answer at all.

```bash
uv run python failures/mark.py > failures/to_mark.txt
wc -l failures/to_mark.txt
```

The script prints every answer beside its quote, and fills the table under `## Marks` in `evidence/A10.md` with one row per question for you to fill; it touches nothing else in the file. *You should see* `92 failures/to_mark.txt`: six blocks of fifteen lines (a blank line, a rule, the question, the answer field, the quote, five `api` lines, five `gpt2` lines), then a blank line and the last line, `wrote 6 rows to fill under ## Marks in evidence/A10.md`. Open `to_mark.txt` beside `evidence/A10.md`. For each answer line, read the answer against the quote above it, decide `r`, `f`, `h` or `o`, and type that letter into the matching position of the row under `## Marks`: the first API letter is `api 1`, the fifth GPT-2 letter is `gpt2 5`, so `.....` becomes something like `rrfrf`. A GPT-2 answer that is a question, a blank, or another topic is `o`; a loop of the same words is `o` too, unless the words state a different fact. When the dots are gone:

```bash
uv run python failures/mark.py counts
```

*You should see* a six-row table of counts (`off` is whatever is left of five, so it has no column), a totals row, the most-fabricated question with its quote, the most split question, and how many questions the API got 5/5 while GPT-2 got 0/5. Whatever your corpus, **expect GPT-2 right on almost nothing**: 124 million parameters and no corpus. The API will be right on the famous facts and fabricating on the minor ones. *If it broke* with `AssertionError: these rows are not five letters`, a row still has a dot or a letter outside `r f h o`; the message names the ids. Paste the table and the lines under it under `## Counts`, and commit:

```bash
git add evidence/A10.md failures/to_mark.txt
git commit -m "A10: 60 answers marked"
```

**Step 5. Write three entries.**

In `evidence/A10.md`, delete `### Entry 4` through `### Entry 10` and everything under them, down to the `## Extension` heading, and under `### 2.` in `## Reflection` delete the `<answer: ...>` line that begins `one or two sentences: the entry number you first filed under the wrong category`; `check_evidence.py` then counts only what is left. Each entry has seven fields, one slot each, in this order: **Repro** (the question verbatim, the model, every sampling setting, and the command that reruns it), **Observed** (the answer, pasted from the trace, with the run named), **Expected** (the quote from your corpus, and its line in the file), **Rate** (failures over runs, with N: `4/5 API runs said X`), **Category** (one of the eight below, or one you name), **Hypothesis** (a claim about machinery, tied to something from A03 to A09), **Evidence** (logprobs at the failure position, the tokenizer split, a diff across runs, or a token count). The eight categories: tokenization artifact · context window or truncation · sampling nondeterminism · retrieval or grounding failure · instruction-following collapse · tool-schema mismatch · confident fabrication · loop or repetition.

- **Entry 1: the most-fabricated question, on the API.** Its Evidence field is what `evidence.py` prints. Run it with the id, the position of an `f` in that row under `## Marks` (an `f` in position 3 of `q02`'s API letters is `q02 3`), and the wrong word that run gave:

  ```bash
  uv run python failures/evidence.py q02 3 Pisenor
  ```

  *You should see* the question, answer, quote and text, then one line per token: position, token, its five alternatives with probabilities, and a `where` column that reads `in quote`, `in question`, nothing, or `NOT IN QUOTE OR QUESTION`, with an arrow on the word you named. Under the table: the word pointed at and its first token, the probability on it, whether any of the five alternatives there begins the right answer, and the five themselves. The first token is the one that matters: the API splits a rare name into pieces (`Pisenor` is ` P`, `isen`, `or`), and once the first piece is chosen the rest follow at probabilities near 1.0 whatever the model believed about the name. **A probability near `1.0` on a wrong name's first token with the right one absent from the five is what confident fabrication looks like in numbers**; a wrong name at `0.4` with the right one at `0.3` beside it is a sampling story.
- **Entry 2: the same question, on GPT-2.** Five answers, usually none of them an answer. Say what the five did instead, with the rate.
- **Entry 3: a failure you caused.** The model did what it was told. Look at the `Answer in one sentence.` suffix in `ask_api` (it pushes the model away from "the text does not say"; your `hedged` column measures how far) and at the wording of your questions ("What false name..." hands the model a shape to fill). The Evidence field quotes the line responsible.

A hypothesis with a trace under it is an argument. Without one it is a guess, and I read it as a guess. Commit after the first entry and again after the third: `git log --oneline failures/ evidence/A10.md` shows the order you found things in, and I read it.

**Extension — the answer on the page (ASSIGNED)**

`failures/deepdive.py --option A` takes your five most-fabricated questions, by your marks, and asks the API each one five more times with the paragraph of your corpus that holds the quote pasted above the question: open-book. "Right" in the script is a rule, not a reading: the `answer` field appears in the reply as whole words, ignoring case (`Ops` counts, `Cyclops` does not); it prints its per-question table so you can overrule the rule by hand in `evidence/A10.md`, and say that you did.

**Write your predicted number of right answers out of 25 in the log before you run**, and in the `Predicted number` slot under `## Extension`. Then:

```bash
uv run python failures/deepdive.py --option A | tee failures/deepdive.txt
```

*You should see* the five ids with their fabricated counts, one progress line per question, then three lines that start `NUMBER` (right out of 25, open-book), `BASELINE` (right out of 25 closed-book on the same five, by your marks and by the rule) and `COMPARISON` (open-book minus closed-book), then a five-row table and, under `where it got worse`, every run still wrong with the answer on the page. 25 API calls, under a minute. *If it broke* with `fill the marks in evidence/A10.md first`, Step 4 is not done. `no chunk holds the quote` means that question's quote runs across a paragraph break; shorten it to one paragraph and rerun `check_questions.py`. Fill `## Extension`: `Option: A`, the category it measures, the three lines pasted, where it got worse, and the falsifier: the result that would have come out the other way if fabrication were not about what the model has seen, which for this option is open-book no better than closed-book.

**Commit, push, PR, sign off.** Fill `## Reflection` (the questions are below) and run `uv run python scripts/check_evidence.py A10` until it says `all slots filled`. Then:

```bash
git add failures/ evidence/A10.md
git commit -m "A10: three catalog entries with rates, open-book deep dive"
git push -u origin dev/failure-catalog
git log --oneline failures/ evidence/A10.md
```

*You should see* several commits, not one, with the questions commit before the traces and the traces before the marks.

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the hypothesis you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/A10.md` with every slot filled (the checker's four lines, the marks table with its letters and the counts under it, three entries with rates, categories, hypotheses and `evidence.py` output in the first, the open-book deep dive, the three reflection answers) + `failures/questions.jsonl` + `failures/check_questions.py` + `failures/ask.py` + `failures/traces/` + `failures/to_mark.txt` + `failures/deepdive.txt` + `failures/deepdive_A.json`.

**Reflection Questions** (answer under `## Reflection` in `evidence/A10.md`)

1. Your most-fabricated question: paste it, the quote from your corpus, and two of the fabricated answers. From its trace, give the probability on the first wrong token of one of them and say whether the right token was anywhere in the top five at that position. What does that number say about how "confident" the model was?

   *How to get it:* the question is the `most-fabricated question (API)` line that `mark.py counts` printed, with its id; the two answers are `api` lines marked `f` for that id in `failures/to_mark.txt`. The probability is the `probability on it` line that `evidence.py` printed for Entry 1, and `right answer in the top five there` is the yes or no. Then read the number against the two shapes in Step 5: near `1.0` with the right token absent, or a split between two names.

2. Paste `git log --oneline failures/ evidence/A10.md`. Name the failure you caused and the line of `ask.py` or `questions.jsonl` responsible.

   *How to get it:* the log is the command as written, from the repo root. The line responsible is the one you quoted in Entry 3's Evidence field; `grep -n "one sentence" failures/ask.py` prints the comment and the code line, and the second is the line.

3. Your extension: the number, the baseline and the comparison, with the prediction you wrote in the log before running. Which questions were still wrong with the answer on the page, and what would it have taken for the open-book number to come out no better than the closed-book one?

   *How to get it:* the prediction is in your log entry (`grep -n -i predict logs/*.md` in the log repo); the three lines are `grep -n -e "^NUMBER" -e "^BASELINE" -e "^COMPARISON" failures/deepdive.txt`, and the still-wrong questions are the table and the lines under `where it got worse` in the same file. The last part is the falsifier from `## Extension`, in your own words: open-book no better than closed-book would mean the model was not missing the text, it was ignoring it.
