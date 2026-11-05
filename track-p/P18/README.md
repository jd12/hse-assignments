# P18 · Citations That Resolve, and a Gate With a Number in It

**Meetings:** D41 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 10 Augmented Generation, lessons 4 and 5, about 20 minutes.

**During the chapter**

**Copy the two refusal lines.** Lesson 4's prompt and lesson 5's example prompt each tell the model what to say when the documents do not contain the answer. Write both lines in `scratch/p17-rag.md`, and under them the one thing neither prompt can do: stop the model from being asked at all. `rag/answer.py`'s gate is that.

**Notes**

**A citation is only real if it resolves.** The chapter numbers the documents `[1]` to `[5]` and asks the model to cite the numbers. On your chunks the citation is the chunk id, which points at a character range in your corpus, and `rag/answer.py` checks every id the model wrote against the five it was given. An id that was never retrieved is logged as `invalid`. An answer with an invalid citation is worse than one with none, because it looks checked.

**The refusal policy is a written rule with a number in it.** "Try not to make things up" is not a policy. "If the reranker's best score is below `-6`, the pipeline answers `I don't have enough information` without calling the model; otherwise the model answers only from the passages and says the same line if they do not contain it" is a policy. The number comes from your own scores, in Step 4, and the two halves of the rule fail in different ways.

**The near-miss is the hard case.** A question whose answer is almost in your corpus: the right subject, a different detail. The retriever finds relevant chunks, the score is high, the gate passes, and the model has to notice the specific thing is not there. On Homer, `What was the name of Ulysses' ship?` retrieves five chunks about Ulysses' ship, and the text never names it. Build one of these on purpose.

**Walkthrough — the chapter, then the pipeline with its gate**

**Step 1. Branch, log.**
Same branch as P17:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/answer && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 10, lessons 4 and 5, every lesson green
- [ ] Step 3: one cited answer, the citation read by hand
- [ ] Step 4: the gate from my own scores, REFUSE_BELOW set
- [ ] Step 5: three refusals, the near-miss that got through
- [ ] Extension: the bare-string trap, the gate at -100
- [ ] Fill every slot in evidence/P18.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Chapter 10, lessons 4 and 5.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p18-bootdev-ch10.png` in the rag repo.

**Lesson 4: citations.** A third command and a third prompt in `cli/augmented_generation_cli.py`. The prompt:

```python
CITATIONS = """Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
```

The parser, the command list, and the arm:

```python
    citations_parser = subparsers.add_parser("citations", help="Answer with citations to the results")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="How many results to cite from")
```

```python
    if args.command not in ("rag", "summarize", "citations"):
```

```python
        case "citations":
            print(f"\nLLM Answer:\n{ask(CITATIONS.format(query=query, docs=docs))}")
```

`format_docs` already numbers the documents `[1]` to `[5]`, which is why it was written that way in lesson 1. Run, then submit:

```bash
uv run cli/augmented_generation_cli.py citations 'action movie with lasers'
bootdev run -s 09fd0c23-752e-4c00-8f83-456d874366d1
```

*You should see* the titles with `Eliminators` among them, then an answer with `[1]`-style numbers in it; the title is the check. Write the numbers the answer used in `evidence/P18.md` under Step 2. Look at one: does the sentence that cites `[3]` say what document `[3]` says? That question is Step 3 on your corpus.

**Lesson 5: a question, answered.** The fourth command takes a `question` instead of a `query`, which is the one place this file's shape changes:

```python
QUESTION = """Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {query}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
```

```python
    question_parser = subparsers.add_parser("question", help="Answer a question from the results")
    question_parser.add_argument("question", type=str, help="The question")
    question_parser.add_argument("--limit", type=int, default=5, help="How many results to answer from")
```

```python
    if args.command not in ("rag", "summarize", "citations", "question"):
        parser.print_help()
        return

    query = args.question if args.command == "question" else args.query
```

```python
        case "question":
            print(f"\nAnswer:\n{ask(QUESTION.format(query=query, docs=docs))}")
```

Run, then submit:

```bash
uv run cli/augmented_generation_cli.py question 'Who are the main characters in Jurassic Park?'
bootdev run -s 697bd109-e54d-46bc-90e7-290c58efa365
```

*You should see* `Jurassic Park` in the titles and an answer naming Alan Grant, Ellie Sattler and Ian Malcolm; the six names are the check. That is chapter 10 done, and the course's scheduled chapters with it; 11 and 12 are extra credit, in January if you want them.

**Step 3. A cited answer.**

`rag/answer.py`'s `cite` mode is lesson 4 with the chapter's prompt tightened: every chunk goes in with its id, every sentence must end with one, and the only refusal line allowed is `I don't have enough information`. Pick a golden question whose chunk the reranker finds (`rag/eval/runs.json` says which), then:

```bash
uv run python -m rag.answer cite "<the question>"
```

*You should see* the five ids with scores, an answer whose sentences end in ids, and a line `cited: [...]   invalid: []   gate: passed`. On Homer, for `How did the old nurse recognize Ulysses while washing his feet?`:

```text
{{LLM:P18-cite}}
```

Paste it under `## Step 3`. Then read the cited chunk and decide whether the sentence citing it says what the chunk says:

```bash
uv run python -c "
from rag.chunking import load_chunks
c = {c['id']: c for c in load_chunks()}['<a cited id>']
print(c['char_start'], c['char_end']); print(' '.join(c['text'].split()))
"
```

`invalid: []` proves every id the model wrote was one it was given. It does not prove the sentence in front of the id is true of that chunk; your eyes do that, once, today. *If it broke* with `cited: []` under an answer that plainly came from the chunks, the model used a different bracket style; paste the answer in `scratch/p17-rag.md` and say so. The Reflection's first question is about exactly this.

**Step 4. The gate, from your own scores.**

Three groups of questions through the reranker, top score only: your twenty-five golden questions, five questions your corpus cannot answer because it never mentions the subject, and three near-misses, questions about your corpus's subjects whose specific answer is not in the text. Write the eight new questions first, then:

```bash
uv run python -c "
from rag import rerank
from rag.eval.check_golden import load_golden
for g in load_golden():
    h = rerank.rerank(g['question'], 1)[0]
    print('in  ', g['id'], round(h['score'], 2), 'labeled' if h['id'] in g['chunk_ids'] else 'other')
for q in ['<out-of-corpus 1>', '<out-of-corpus 2>', '<out-of-corpus 3>', '<out-of-corpus 4>', '<out-of-corpus 5>']:
    print('out ', round(rerank.rerank(q, 1)[0]['score'], 2), q)
for q in ['<near-miss 1>', '<near-miss 2>', '<near-miss 3>']:
    print('near', round(rerank.rerank(q, 1)[0]['score'], 2), q)
"
```

*You should see* thirty-three lines. On Homer the in-corpus top scores run from `-5.78` (g10, the many-headed monster, where the top chunk is the wrong one) to `7.04` (g02, Antinous and the arrow), most of them between `0` and `6`; the five out-of-corpus questions (the Divine Comedy, the capital of Australia, the Colosseum, the Moon landing, a bicycle's gears) score `-7.22` to `-11.01`; and the three near-misses (`How many suitors did Penelope have in all?`, `What was the name of Ulysses' ship?`, `How many years did the Trojan war last?`) score `-0.55`, `2.17` and `0.68`, right among the in-corpus questions. The cross-encoder's scores are not probabilities and the line is yours to draw: on Homer, `REFUSE_BELOW = -6.0` sits between the lowest in-corpus score and the highest out-of-corpus one, refuses none of the twenty-five, and lets all three near-misses through. Set `REFUSE_BELOW` in `rag/answer.py`, paste the thirty-three lines under `## Step 4`, and write the number, how many in-corpus questions it refuses, and how many out-of-corpus questions it lets through. If your two groups overlap, the two counts are the cost of the gate, and you choose which cost to pay.

**Step 5. Three refusals.**

With the gate set, three questions through `cite`:

```bash
uv run python -m rag.answer cite "<an out-of-corpus question>"
uv run python -m rag.answer cite "<near-miss 1>"
uv run python -m rag.answer cite "<near-miss 2>"
```

*You should see* the first refused with `gate: refused before generation` and no request made (the counter in `rag/llm.py` is how you know); and the near-misses each getting to the model, which either refuses with the one allowed line or answers anyway. On Homer:

```text
{{LLM:P18-refusals}}
```

Paste all three under `## Step 5`, and write "where it got worse": the near-miss the model answered anyway, the sentence it made up, and what it cited for it. If the model refused all three near-misses, run the third; if it refused that too, say so, and the Reflection's third question becomes the reason you still do not trust it.

**Extension — The bare-string trap, then the gate too low** *(assigned)*

Two things done wrong on purpose, once each.

First, the chunks without their ids. `answer` takes a `with_ids` argument; call it with the ids off for the Step 3 question:

```bash
uv run python -c "
from rag.answer import answer
r = answer('cite', '<the Step 3 question>', with_ids=False)
print(r['answer']); print('cited:', r['cited'], ' invalid:', r['invalid'])
"
```

*You should see* citations anyway: ids in the right shape that were never in the prompt, or numbers like `[1]`, which `cited_ids` does not count. On Homer:

```text
{{LLM:P18-bare}}
```

Second, the gate off. Set `REFUSE_BELOW = -100.0`, run `cite` on the out-of-corpus question from Step 5, and put the gate back:

```text
{{LLM:P18-nogate}}
```

Paste both under `## Extension` with one line each: what the model cited when it had no ids, and what it answered when the gate was off.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P18.md` (the questions are below) and run `uv run python scripts/check_evidence.py P18` until it says `all slots filled`. Then:

```bash
git add rag/answer.py scratch/ evidence/
git commit -m "P18: cited answers checked against the retrieved ids, the gate from my own scores, three refusals"
git push
```

The P17 pull request is still open and now has both days on it; leave it for me.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P18.md` with every slot filled + `rag/answer.py` with your `REFUSE_BELOW` + `evidence/p18-bootdev-ch10.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P18.md`)

1. Paste your Step 3 output. Say what `invalid: []` proves and what it does not prove about the answer, and what the hand check in Step 3 added.

   *How to get it:* it proves the ids resolve to chunks the model was shown; it does not prove the sentence is in the chunk. The hand check is the only step that compared a sentence with its source, and it does not scale, which is why the milestone scores answers with `answer` strings instead.

2. Paste your calibration numbers and your threshold. Say what the gate costs when it is set too high and when it is set too low, each with one question of yours it would get wrong.

   *How to get it:* too high refuses in-corpus questions whose best chunk scored low; on Homer, anything above `-5.78` refuses g10, which the pipeline would have answered badly anyway, and anything above `-1.34` refuses g07, which it answers well. Too low lets out-of-corpus questions reach the model, which then answers from memory and cites whatever it was given.

3. Paste the near-miss the model answered. Say why the gate cannot catch a near-miss, what the prompt asks the model to do instead, and which of the two defenses you trust less and why.

   *How to get it:* the gate reads a score, and a near-miss retrieves chunks that really are about the subject, so the score is as high as a true hit. The prompt asks the model to notice the detail is absent and say the line. A rule with a number in it does the same thing every time; a model on the free router does not, and you have the output to prove which way it went today.
