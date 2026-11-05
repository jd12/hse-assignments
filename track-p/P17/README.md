# P17 · Generation, Augmented

**Meetings:** D40 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 10 Augmented Generation, lessons 1 to 3, about 25 minutes.

**During the chapter**

**Write the three verbs.** Lesson 1 says what RAG does in three steps, each one verb. Write them in `scratch/p17-rag.md` in the rag repo, and beside each the P-assignment where you first built it. Two of the three were done by P14.

**Lesson 3 is a reading with a question; answer it from your corpus, not the movies.** The question asks which kind of document would least need a conflict step. Write your answer in `scratch/p17-rag.md`, then one line on whether your corpus is that kind.

**Notes**

**The model finally sees your text, and only five chunks of it.** Every chapter so far put the model before the search or beside it. Lesson 1 puts the five retrieved chunks into the prompt and asks for an answer from them. What the model writes is only as good as what the reranker handed it: a wrong chunk in, a confident wrong answer out, and you will see one today.

**A summary is for a reader, and the reader is named in the prompt.** Lesson 2's prompt says who the users are. On your corpus the reader is someone who knows the text and wants the five chunks condensed, not a streaming customer choosing a movie. `CORPUS_NOTE` in `rag/answer.py` is that one line; set it before anything runs.

**Nothing today is checkable from the outside.** The answers have no citations yet, so a sentence the model added from memory looks the same as one it took from a chunk. The walkthrough has you find one by hand. P18 makes it mechanical.

**Walkthrough — the chapter, then answers from your chunks**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P15 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/answer
```

*If P15 is not approved yet:* `git switch dev/eval && git switch -c dev/answer`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 10, lessons 1 to 3, every lesson green
- [ ] scratch/p17-rag.md: the three verbs, lesson 3's answer for my corpus
- [ ] Step 3: predictions in evidence/P17.md, committed
- [ ] Step 4: two rag answers, the sentence not in the chunks found by hand
- [ ] Step 5: two summaries, the better one named
- [ ] Fill every slot in evidence/P17.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 10, lessons 1 to 3.**

In `~/version_control/rag-search-engine/`. Stop after lesson 3; lessons 4 and 5 are P18. When the chapter page shows lessons 1 to 3 complete, save it as `evidence/p17-bootdev-ch10a.png` in the rag repo.

**Lesson 1: retrieve, augment, generate.** A fourth CLI. Create `cli/augmented_generation_cli.py` with the chapter's `rag` command. The `format_docs` helper numbers the documents, which lesson 4 needs for citations, so it is here from the start:

```python
import argparse

from lib.hybrid_search import HybridSearch
from lib.llm import ask
from lib.search_utils import load_movies

RAG = """You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""


# In: the RRF entries.  Out: the Documents block of a prompt, numbered so the model can cite [1], [2].
def format_docs(results):
    return "\n\n".join(f"[{i}] {e['doc']['title']}: {e['doc']['description']}" for i, e in enumerate(results, start=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()
    if args.command not in ("rag",):
        parser.print_help()
        return

    query = args.query
    limit = 5
    searcher = HybridSearch(load_movies())
    results = searcher.rrf_search(query, 60, limit)[:limit]
    print("Search Results:")
    for entry in results:
        print(f"- {entry['doc']['title']}")
    docs = format_docs(results)

    match args.command:
        case "rag":
            print(f"\nRAG Response:\n{ask(RAG.format(query=query, docs=docs))}")


if __name__ == "__main__":
    main()
```

The search and the printing of titles happen before the `match`, because every command in this file does them the same way; only the prompt differs. Run, then submit:

```bash
uv run cli/augmented_generation_cli.py rag 'movies about action and dinosaurs'
bootdev run -s 9b47662d-1ee7-4139-ac41-cba1d7c4e553
```

*You should see* five titles with `Jurassic Park` among them, then `RAG Response:` and a paragraph about them; the title is the check. Write the response's first sentence in `evidence/P17.md` under Step 2. *If it broke* with a response about movies not in the five, the model is answering from memory; the prompt says "based on documents retrieved", and free models do not always listen. Keep the output; it is the Reflection's second question.

**Lesson 2: a summary of several documents.** A second command with a `--limit`, and a second prompt. Add the prompt above `format_docs`:

```python
SUMMARIZE = """Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
```

The parser, and the `limit` line and the `match` arm:

```python
    summarize_parser = subparsers.add_parser("summarize", help="Summarize the search results")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="How many results to summarize")
```

```python
    if args.command not in ("rag", "summarize"):
```

```python
    limit = 5 if args.command == "rag" else args.limit
```

```python
        case "summarize":
            print(f"\nLLM Summary:\n{ask(SUMMARIZE.format(query=query, docs=docs))}")
```

Run, then submit:

```bash
uv run cli/augmented_generation_cli.py summarize 'movies about action and dinosaurs'
bootdev run -s d49e3e9d-7ead-475d-8b83-d34aed09d19e
```

*You should see* the titles and `LLM Summary:` with three or four sentences that mention several of them; `Jurassic Park` in the output is the check.

**Lesson 3: conflicts.** A reading. The question is about which documents least need a conflict step; the answer is the curated kind, and your `scratch/p17-rag.md` line says whether your corpus is that kind.

**Step 3. Predictions, committed before the run.**

Pick two questions your corpus answers, not from the golden set: one whose answer is in one sentence, one whose answer is spread over a passage. Before you run them, write under `## Step 3` in `evidence/P17.md`, for each, whether the five retrieved chunks will contain the answer (yes, partly, no) and whether the model's answer will stay inside them. Commit the file. I check your commit timestamps.

**Step 4. Two answers from your chunks.**

Open `rag/answer.py` and set `CORPUS_NOTE`. Read `answer`: retrieve with the reranker, build the Documents block, fill the chapter's prompt, call the model, log everything to `data/answers.jsonl`. Then:

```bash
uv run python -m rag.answer rag "<your one-sentence question>"
uv run python -m rag.answer rag "<your passage question>"
```

*You should see* the five chunk ids with their reranker scores, then the answer. On Homer, for `How did the old nurse recognize Ulysses?`:

```text
{{LLM:P17-rag1}}
```

and for `Why did Achilles withdraw from the fighting?`:

```text
{{LLM:P17-rag2}}
```

Paste both under `## Step 4`. Then the hand check: for each answer, open the five chunks (`uv run python -m rag.search` prints ids; the chunk text is `load_chunks()` by id, the P18 command for that is below) and find one sentence of the answer that is in none of them, if there is one. The model is allowed to say things the chunks do not; nothing in today's prompt forbids it, and nothing in the output marks it.

```bash
uv run python -c "
from rag.chunking import load_chunks
c = {c['id']: c for c in load_chunks()}['<a chunk id>']
print(' '.join(c['text'].split()))
"
```

*If it broke* with `OPENROUTER_API_KEY is not set`, the `.env` is missing from the rag repo on this branch; it is ignored by git, so it never travels with a branch, but it is in the folder, and a fresh clone would need it again.

**Step 5. Two summaries.**

Two topics rather than questions, the kind of thing a reader looks up:

```bash
uv run python -m rag.answer summarize "<a character or place>"
uv run python -m rag.answer summarize "<another>"
```

*You should see* the five chunks and three or four sentences combining them. On Homer, for `the Cyclops`:

```text
{{LLM:P17-sum1}}
```

Paste both under `## Step 5` and say which is the better summary for a reader of your corpus, and one fact it took from a chunk you would not have picked for the topic.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P17.md` (the questions are below) and run `uv run python scripts/check_evidence.py P17` until it says `all slots filled`. Then:

```bash
git add rag/answer.py scratch/ evidence/
git commit -m "P17: rag and summarize on my chunks, one sentence the chunks did not say"
git push -u origin dev/answer
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P18.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P17.md` with every slot filled, the predictions committed before the runs + `scratch/p17-rag.md` + `evidence/p17-bootdev-ch10a.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P17.md`)

1. Paste one rag answer beside its five chunks' ids. Say what "augmented" means in the chapter's three verbs, and which of the three the pipeline from P06 to P14 already did.

   *How to get it:* retrieve is P06 to P14 entire; augment is putting the retrieved text into the prompt, which `rag/answer.py` does in one line; generate is the model. Two of three were built before the model ever saw a chunk.

2. Paste the answer with the sentence not in the chunks, or the answer you trust least. Say how you would know, as a reader with no corpus in front of you, whether that sentence came from the chunks or from the model, and what P18 adds to make that checkable.

   *How to get it:* you would not know; the sentence reads the same either way. P18 puts an id on every chunk, requires one at the end of every sentence, and checks that every cited id was actually retrieved.

3. Lesson 3 says conflict resolution would not help the chapter's movie data. Say whether your corpus has the kind of conflicts lesson 3 lists, with one example from your chunks or a reason there is none, and what a conflict step would have to compare.

   *How to get it:* Homer has emphasis conflicts and a few factual ones (Butler's footnotes argue with the text about geography); a single-author novel has almost none; a corpus of reviews or forum posts is all conflicts. The step compares what two retrieved chunks say about the same thing, before the summary blends them.
