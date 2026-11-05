# P13 · Rewrite, Expand, Measure

**Meetings:** D36 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 7 LLMs, lessons 4 and 5, about 20 minutes.

**Notes**

**Rewrite replaces the query; expand adds to it.** Lesson 4's prompt asks for a new query under ten words, written the way a search engine likes: names, nouns, no filler. Lesson 5's prompt asks only for extra terms, and the code appends them to what the user typed. The difference matters to BM25: a rewrite can drop the one rare word that was finding the chunk, and an expansion can add five common words that match everything.

**The model knows things your corpus does not say.** Ask it to expand `the Cyclops` and it may add `Polyphemus`, which is right about the story and absent from Butler, who writes `Cyclops` and `Polyphemus` both, but never near each other. A term can be true and still pull the search toward the wrong chunks. The walkthrough measures this instead of eyeballing it: for ten queries, the hybrid rank of the answer chunk before and after each enhancement.

**Twenty requests.** The checker sends two per query, and it caches both. Running it a second time costs nothing. Edit `CORPUS_NOTE` or a prompt and the changed prompts cost again.

**Walkthrough — the chapter, then the two enhancements measured**

**Step 1. Branch, log.**
Same branch as P12:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/llm && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 7, lessons 4 and 5, every lesson green
- [ ] Step 3: five more queries with needles in rag/questions.json, scoreboard still passes
- [ ] Step 4: rag.enhance_check, the table in evidence/P13.md, predictions checked
- [ ] Step 5: one that helped and one that hurt, read closely
- [ ] Fill every slot in evidence/P13.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Chapter 7, lessons 4 and 5.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p13-bootdev-ch7.png` in the rag repo.

**Lesson 4: rewriting.** A second prompt and a second choice. In `cli/lib/query_enhancement.py`, add the prompt above `PROMPTS` and the entry inside it:

```python
REWRITE = """Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
```

```python
PROMPTS = {"spell": SPELL, "rewrite": REWRITE}
```

In the CLI, the choice:

```python
        choices=["spell", "rewrite"],
```

Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "bear movie that gives me the lulz" --enhance rewrite
bootdev run -s 5fd4c9aa-f463-429f-8e3a-3a980118dce4
```

*You should see* an `Enhanced query (rewrite)` line with something like `funny bear comedy movie` and results to match; the word `rewrite` is the check. Write the rewritten query in `evidence/P13.md` under Step 2. *If it broke* with `KeyError: 'rewrite'`, the `PROMPTS` entry is missing.

**Lesson 5: expansion.** The third prompt returns only new terms, so `enhance_query` has to append them:

```python
EXPAND = """Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
```

```python
PROMPTS = {"spell": SPELL, "rewrite": REWRITE, "expand": EXPAND}


# In: a method name and the query.  Out: the enhanced query; for expand, the original with the new terms appended.
def enhance_query(method, query):
    reply = ask(PROMPTS[method].format(query=query)).strip('"')
    if method == "expand":
        return f"{query} {reply}"
    return reply
```

```python
        choices=["spell", "rewrite", "expand"],
```

Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search --enhance expand "math movie" --limit 25
bootdev run -s c7be50a9-501b-474a-9970-592f2556c2b2
```

*You should see* the query grow into something like `math movie mathematics numbers equations genius professor`, and `Donald in Mathmagic Land`, `Good Will Hunting` and `I.Q.` somewhere in the 25; those three titles are the check. *If it broke* with none of the three in the list, run it again: the router may have given a model that added unhelpful terms, and the chapter marks this lesson safe to retry.

**Step 3. Five more queries.**

Ten queries make a better table than five. Create `rag/questions.json` with five more, in the same shape as `rag/queries.json`: an `id` from `q06` to `q10`, a `query`, a `needle` copied exactly from the chunk that answers it, and a `kind`. Pick passages you have not used: open the corpus, choose the passage first, write the query a reader would type, then copy a few words of the answer as the needle. On Homer:

```json
[
  {"id": "q06", "query": "the old dog that knew his master after twenty years",
   "needle": "Argos died as soon as he had recognised his master", "kind": "paraphrase"},
  {"id": "q07", "query": "what Circe's drug did to the sailors",
   "needle": "turned them into pigs by a stroke of her wand", "kind": "paraphrase"},
  {"id": "q08", "query": "Scamandrius Astyanax",
   "needle": "the people called him Astyanax", "kind": "rare-name"},
  {"id": "q09", "query": "Glaucus trades armour with Diomed",
   "needle": "exchanged golden armour for bronze", "kind": "rare-name"},
  {"id": "q10", "query": "how many layers the god hammered into the great shield",
   "needle": "He made the shield in five thicknesses", "kind": "detail"}
]
```

Then check every needle is really in a chunk:

```bash
uv run python -c "
import json
from rag.chunking import load_chunks
from rag.scoreboard import norm
units = load_chunks()
for q in json.load(open('rag/questions.json', encoding='utf-8')):
    hits = [u['id'] for u in units if norm(q['needle']) in norm(u['text'])]
    print(q['id'], hits)
"
```

*You should see* one or two chunk ids per query; an empty list means the needle is not an exact copy. Paste the five lines under `## Step 3`.

**Step 4. The table.**

Open `rag/enhance_check.py`: for each of the ten queries, the hybrid rank of the answer chunk in the top ten for the query as typed, then rewritten, then expanded, two requests per query. Then:

```bash
uv run python -m rag.enhance_check
```

*You should see* twenty `rewrite` and `expand` lines showing what the model did to each query, a table with ten rows and three rank columns, and two count lines. On Homer:

```text
{{LLM:P13-table}}
```

Paste the whole output under `## Step 4` and check your P12 predictions against the `expand` column.

*If it broke* with `no JSON list` or a model reply that is a paragraph instead of a query, a free model ignored "output only"; `rag/llm.py` retries empty replies but not wordy ones. Run again: the cache keeps the good replies and only the bad query goes out again.

**Step 5. One that helped and one that hurt.**

Take one row where the rank improved and one where it got worse. For each, put the enhanced query next to the answer chunk (`uv run python -m rag.eval.find <needle>` prints it, or open the chunk id from Step 3) and find the words the model added: for the helped one, which added words are in the chunk; for the hurt one, which added words are true about the subject and absent from the chunk. Two lines each under `## Step 5`.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P13.md` (the questions are below) and run `uv run python scripts/check_evidence.py P13` until it says `all slots filled`. Then:

```bash
git add rag/questions.json evidence/
git commit -m "P13: rewrite and expand measured on ten queries, five new needles"
git push
```

The P12 pull request is still open and now has both days on it; leave it for me.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P13.md` with every slot filled + `rag/questions.json` + `evidence/p13-bootdev-ch7.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P13.md`)

1. Paste your table. Say which of the two enhancements helped more of your ten and why that fits what each one does to the query (rewrite replaces words, expand adds them), using BM25's behavior on added terms from chapter 3.

   *How to get it:* BM25 sums a score per query term, so every added term that is in the answer chunk adds to its score and every added term that is in two hundred other chunks adds to theirs. A rewrite can lose the rare term; an expansion keeps it and dilutes it.

2. Paste your "hurt" case. Say what the model knew that your corpus does not say, and why a term that is true about the subject can still lower the rank of the right chunk.

   *How to get it:* the model's added words come from everything it read, not from your text. If the added word is in your corpus but in other chunks, the search is pulled toward those; if it is in no chunk, BM25 ignores it and the semantic vector drifts anyway.

3. The chapter runs the enhancement on every query. Say what you would check about a query before deciding to expand it, and what the cost of expanding every query is on the free tier, in requests per day for your five scoreboard queries.

   *How to get it:* a query with a rare proper noun in it already has what BM25 needs; a query of common words is the one expansion can help. Five scoreboard queries, one expansion each, is five requests, a tenth of the day; the scoreboard caches them, and a prompt edit spends them again.
