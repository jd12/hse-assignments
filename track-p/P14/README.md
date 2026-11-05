# P14 · A Second Reader for the Top Twenty

**Meetings:** D37 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 8 Reranking, lessons 1 to 4, about 35 minutes.

**During the chapter**

**Write down the two stages.** Lesson 1 names them with numbers: how many candidates the fast stage pulls and how many the slow stage keeps. Write both in `scratch/p14-rerank.md` in the rag repo, and beside them the number the chapter's code actually uses (five times the limit). `rag/rerank.py` uses 20; the Reflection asks what that costs.

**Count the requests before you run lesson 2.** Lesson 2 sends one request per document with a three-second sleep between them. Work out, before running it, how many requests and how many seconds the check command costs for `--limit 3`, and write both in `scratch/p14-rerank.md`. Then run it and see.

**Notes**

**The cross-encoder reads the query and the chunk together.** Chapter 4's model made one vector per chunk, once, and one per query, and the score was the angle between them. The cross-encoder takes the pair as a single input and outputs the score directly, so it can see which words of the query sit next to which words of the chunk. That is why it is better, and why it cannot be precomputed: the query is half the input. Twenty pairs per query is what makes it affordable.

**Its scores are not probabilities and can be negative.** The chapter's example has `3.314` first and `-1.276` third. Only the order means anything, and scores from two different cross-encoders cannot be compared. P18 uses the top score as a gate, and you will set the line from your own numbers, not from a scale.

**The chapter's model is the smallest one that works on movies.** `ms-marco-TinyBERT-L2-v2` is two layers, trained on web search questions. On Butler's prose it reads differently than on movie blurbs, and the walkthrough measures what it does to your five queries. P16 puts a six-layer model beside it on twenty-five questions and chooses.

**Walkthrough — the chapter, then the cross-encoder on your chunks**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P12 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/rerank
```

*If P12 is not approved yet:* `git switch dev/llm && git switch -c dev/rerank`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 8, every lesson green; the request count worked out before lesson 2 ran
- [ ] Step 3: predictions in evidence/P14.md, committed
- [ ] Step 4: rag.rerank on one query, positions counted
- [ ] Step 5: the scoreboard with four columns, predictions checked, where it got worse
- [ ] Extension: the batch LLM reranker on the same query, one request
- [ ] Fill every slot in evidence/P14.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 8, lessons 1 to 4.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p14-bootdev-ch8.png` in the rag repo.

**Lesson 1: two stages.** A reading and one question; the answer is the one about the query and the full documents together.

**Lesson 2: one request per movie.** Create `cli/lib/reranking.py`. The individual method asks the model for a number per document and sleeps between requests; `parse_json_list` is for lesson 3 and is here because both reranking prompts return JSON or a bare number:

```python
import json
import time

from lib.llm import ask

INDIVIDUAL = """Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


# In: a reply that should be a JSON list.  Out: the list, with anything around it (fences, words) dropped.
def parse_json_list(reply):
    start = reply.find("[")
    end = reply.rfind("]")
    return json.loads(reply[start:end + 1])


# In: a query and RRF entries.  Out: the same entries with "rerank_score", one LLM request each, best score first.
def rerank_individual(query, entries):
    for entry in entries:
        doc = entry["doc"]
        reply = ask(INDIVIDUAL.format(query=query, title=doc.get("title", ""), document=doc.get("description", "")))
        try:
            entry["rerank_score"] = float(reply.strip().split()[0])
        except (ValueError, IndexError):
            entry["rerank_score"] = 0.0
        time.sleep(3)                     # the chapter's advice for the free tier
    return sorted(entries, key=lambda e: e["rerank_score"], reverse=True)
```

`reply.strip().split()[0]` takes the first word of the reply, so a model that answers `8.` or `8 out of 10` still gives `8`; the `try` turns a reply with no number at all into `0.0` instead of a crash. In `cli/hybrid_search_cli.py`, import it and add the option to `rrf-search`:

```python
from lib.reranking import rerank_individual
```

```python
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="Re-ranking method",
    )
```

In the case, the search pulls five times the limit when a reranker is set, reranks, and cuts to the limit; the printing gains one line when a rerank score is present:

```python
            searcher = HybridSearch(load_movies())
            pool = args.limit * 5 if args.rerank_method else args.limit      # the reranker reads five times the limit
            results = searcher.rrf_search(query, args.k, pool)[:pool]
            if args.rerank_method:
                print(f"Re-ranking top {pool} results using {args.rerank_method} method...")
                if args.rerank_method == "individual":
                    results = rerank_individual(query, results)
                results = results[:args.limit]
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, entry in enumerate(results, start=1):
                print(f"{i}. {entry['doc']['title']}")
                if "rerank_score" in entry:
                    print(f"   Re-rank Score: {entry['rerank_score']:.3f}/10")
                print(f"   RRF Score: {entry['rrf']:.3f}")
                print(f"   BM25 Rank: {entry['bm25_rank'] or '-'}, Semantic Rank: {entry['semantic_rank'] or '-'}")
                print(f"   {entry['doc']['description'][:100]}...")
                print()
```

Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method individual --limit 3
bootdev run -s 33a9d312-7560-4398-a831-e972f5a29b73
```

*You should see* `Re-ranking top 15 results using individual method...`, about forty-five seconds of sleeping, then three movies with `Re-rank Score` lines and `The Berenstain Bears' Christmas Tree` among them; that title is the check. *If it broke* with the scores all `0.000`, the model replied with words; print one reply to see, and retry.

**Lesson 3: all the movies in one request.** The batch prompt lists every candidate with its id and asks for the ids back as a JSON list. Add to `cli/lib/reranking.py`:

```python
BATCH = """Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""


# In: a query and RRF entries.  Out: the entries the model ranked, with "rerank_rank", in one request.
def rerank_batch(query, entries):
    doc_list_str = "\n".join(f"{e['doc']['id']}: {e['doc']['title']} - {e['doc']['description']}" for e in entries)
    order = parse_json_list(ask(BATCH.format(query=query, doc_list_str=doc_list_str)))
    by_id = {e["doc"]["id"]: e for e in entries}
    ranked = []
    for rank, movie_id in enumerate(order, start=1):
        if movie_id in by_id and by_id[movie_id] not in ranked:
            by_id[movie_id]["rerank_rank"] = rank
            ranked.append(by_id[movie_id])
    return ranked
```

The `if` skips ids the model invented or repeated, which free models do. In the CLI: the import, the choice `"batch"`, an `elif` in the case, and a `Re-rank Rank` line in the printing:

```python
from lib.reranking import rerank_batch, rerank_individual
```

```python
        choices=["individual", "batch"],
```

```python
                elif args.rerank_method == "batch":
                    results = rerank_batch(query, results)
```

```python
                if "rerank_rank" in entry:
                    print(f"   Re-rank Rank: {entry['rerank_rank']}")
```

Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method batch --limit 3
bootdev run -s fbdd5c9c-a24f-49f2-89ca-c9dd0b539684
```

*You should see* the same three movies in about three seconds instead of forty-five, with `Re-rank Rank: 1`, `2`, `3`; the Berenstain title is the check. *If it broke* with `json.decoder.JSONDecodeError`, the model wrapped its list in text with another bracket in it; print the reply, and retry.

**Lesson 4: the cross-encoder.** No model request at all. Add to `cli/lib/reranking.py`:

```python
from sentence_transformers import CrossEncoder
```

```python
# In: a query and RRF entries.  Out: the same entries with "cross_encoder_score", best first; no LLM involved.
def rerank_cross_encoder(query, entries):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
    pairs = []
    for entry in entries:
        doc = entry["doc"]
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
    scores = cross_encoder.predict(pairs)
    for entry, score in zip(entries, scores):
        entry["cross_encoder_score"] = float(score)
    return sorted(entries, key=lambda e: e["cross_encoder_score"], reverse=True)
```

The model is loaded once per query, not once per document; `predict` takes all the pairs at once. In the CLI: the import, the choice `"cross_encoder"`, an `else` in the case, and the score line:

```python
from lib.reranking import rerank_batch, rerank_cross_encoder, rerank_individual
```

```python
        choices=["individual", "batch", "cross_encoder"],
```

```python
                else:
                    results = rerank_cross_encoder(query, results)
```

```python
                if "cross_encoder_score" in entry:
                    print(f"   Cross Encoder Score: {entry['cross_encoder_score']:.3f}")
```

Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method cross_encoder --limit 25
bootdev run -s 73025033-2a95-4c2a-96f7-bb3c74de1c62
```

*You should see* a model download the first time (a few megabytes, far smaller than chapter 4's), then `Re-ranking top 125 results using cross_encoder method...` and 25 movies with `Cross Encoder Score` lines, `Care Bears Movie II: A New Generation` first with a positive score and `A Bear for Punishment` and `The Country Bears` with negative ones; the three titles are the check. Write the three scores in `evidence/P14.md` under Step 2. *If it broke* with a long error about `MPS` or `CUDA`, `device="cpu"` is missing.

**Step 3. Predictions, committed before the run.**

`rag/rerank.py` reads the hybrid top 20 for a query, scores every (query, chunk) pair with the chapter's cross-encoder, and reorders. Before you run it, write under `## Step 3` in `evidence/P14.md`, for each of your five scoreboard queries, whether the answer chunk will move up, down, or stay, and how many of the twenty will change position. Commit the file:

```bash
git add evidence/P14.md
git commit -m "P14: predictions before the reranker runs"
```

I check your commit timestamps.

**Step 4. The reranker on one query.**

```bash
uv run python -m rag.rerank "<your rare-name query from rag/queries.json>"
```

*You should see* the model download once (the same model as lesson 4), then five chunks with their cross-encoder score and the hybrid rank each one came from. On Homer, for `how was the Cyclops blinded`:

```text
cross-encoder rerank of the hybrid top 20 for: how was the Cyclops blinded
1. c00341  score 4.073  was hybrid rank 9  Remember how the Cyclops treated us when our comrades went inside his ...
2. c00305  score 2.132  was hybrid rank 4  and talk, you would tell me where the wretch is hiding, and I would da...
3. c00306  score 1.788  was hybrid rank 6  board at once and put out to sea; so they went aboard, took their plac...
4. c00302  score 0.888  was hybrid rank 16  they went away, and I laughed inwardly at the success of my clever str...
5. c00763  score 0.849  was hybrid rank 11  he thence reached the fertile land of the Lotus-eaters. He told her al...
```

Nineteen of the twenty changed position, and only one chunk of the hybrid top five survived in the reranked top five. The chunk that was hybrid rank 1, the one with `bore the red hot beam into his eye`, is now rank 9: the cross-encoder preferred a chunk where Ulysses' men *remember* the Cyclops, which has the word `Cyclops` beside words about the blinding, over the chunk that narrates the blinding without naming him. Paste your output under `## Step 4` and count your two numbers: positions changed, and hybrid top-five survivors. The count is quick by hand from the `was hybrid rank` column, or:

```bash
uv run python -c "
from rag import search, rerank
q = '<your query>'
pool = search.hybrid(q, rerank.POOL)
scores = rerank.score(q, pool)
order = sorted(range(len(pool)), key=lambda i: -scores[i])
print('positions changed:', sum(1 for new, old in enumerate(order) if new != old), 'of', len(pool))
print('hybrid top five still in the top five:', len({pool[i]['id'] for i in order[:5]} & {c['id'] for c in pool[:5]}))
"
```

*If it broke* with `HF_TOKEN` in a warning, that is a warning about download speed and not an error; the model still loaded.

**Step 5. The scoreboard with four columns.**

`rag/scoreboard.py` has a fourth method since P14, `rerank`, the cross-encoder over the hybrid top 20, cut to five:

```bash
uv run python -m rag.scoreboard
```

*You should see* four columns. On Homer:

```text
query      bm25 semantic   hybrid   rerank
q01           -        -        -        -
q02           -        5        -        -
q03           -        1        1        -
q04           1        2        1        1
q05           -        -        3        -
hit@3       1/5      2/5      3/5      1/5
```

The reranker made the five queries worse, from 3 of 5 to 1 of 5. That is a real result, not a bug, and it is why the `rerank` column exists: the chapter's two-layer model, trained on web questions about web pages, moved the right chunk out of the top five on two queries where hybrid search had it. Paste the table under `## Step 5`, check your predictions, and write "where it got worse": the query, its hybrid and rerank ranks, and the chunk the cross-encoder put first instead, with one line on why that chunk looks right to a model that reads the query and the chunk side by side.

**Extension — The batch LLM reranker on the same query** *(assigned)*

The chapter's lesson 3, on your chunks, one request:

```bash
uv run python -m rag.rerank --llm "<the same query as Step 4>"
```

*You should see* five chunks with `llm rank` and the hybrid rank each came from. On Homer:

```text
{{LLM:P14-llm}}
```

Paste it under `## Extension` and write where the two rerankers agree and where they do not: the chunk each put first, and how many of their top fives are shared. The model read the same twenty chunks the cross-encoder did, so any difference is a difference of readers, not of candidates.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P14.md` (the questions are below) and run `uv run python scripts/check_evidence.py P14` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/ rag/scoreboard.py
git commit -m "P14: the cross-encoder over the hybrid top 20, the scoreboard's fourth column, the LLM reranker beside it"
git push -u origin dev/rerank
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is your "where it got worse" line.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P14.md` with every slot filled, the predictions committed before the run + `scratch/p14-rerank.md` + `evidence/p14-bootdev-ch8.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P14.md`)

1. Paste your Step 4 output. Say what the cross-encoder reads that the bi-encoder of chapter 4 never did, and why that is also the reason it cannot run over all 1,800 chunks per query.

   *How to get it:* the bi-encoder saw each chunk alone and the query alone; the cross-encoder sees the pair, so it can match `blinded` in the query to `eye` in the chunk. Because the input includes the query, there is nothing to precompute: every query means 1,800 fresh model runs if there is no first stage.

2. Paste your scoreboard and the "where it got worse" line. Say what the reranker can and cannot do about a chunk that hybrid search left outside the top twenty, and name the stage that decides that.

   *How to get it:* the reranker only reorders what it is given. A chunk at hybrid rank 24 is invisible to it, however relevant. The pool size in `rag/rerank.py` and the fusion in `rag/hybrid.py` decide what it sees; P21 traces misses to exactly that stage.

3. Paste your extension output. The chapter offers three rerankers. Say which one you would run in a pipeline that answers a hundred questions an hour, and what the cost of the other two is in requests and in seconds, using the chapter's 3-second sleep.

   *How to get it:* individual is 20 requests and 60 seconds of sleep per question, 2,000 requests an hour; batch is one request per question, 100 an hour, still twice the free tier's daily limit; the cross-encoder is zero requests and under a second on your laptop.
