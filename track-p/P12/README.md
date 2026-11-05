# P12 · A Model in Front of the Search

**Meetings:** D35 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 7 LLMs, lessons 1 to 3, about 25 minutes.

**During the chapter**

**Write down what the model is allowed to do.** Lesson 1 lists four things a model can do to a query before the search runs. Write the four in `scratch/p12-llm.md` in the rag repo. Lesson 3's prompt forbids three of them; mark which three. P13 adds the other two back on purpose.

**Count the requests.** Lesson 2 gives two numbers for the free tier: requests per day and requests per minute. Write both in `scratch/p12-llm.md`. Every script in this repo from today on prints how many requests it made, and the Reflection asks you to add them up.

**Notes**

**The key is the one I hand you, and it goes in exactly one place.** Lesson 2 has you make an OpenRouter account; you do not. I made the key, with a spending cap on it, and I hand it to you on paper today. It goes into a file called `.env` in the course project and in the rag repo, one line each, and nowhere else. Both repos already ignore `.env`, and the rag repo's secret guard refuses a commit that contains a key. If you ever see the key in a terminal's scrollback, close the terminal.

**The model is not a search engine.** Lesson 1's picture puts the model before the search: it fixes, expands or rewrites the query, and the search still runs on your chunks. The model never sees your corpus in this chapter. Chapter 10 is where it does.

**`openrouter/free` is a router, and it is rate-limited.** The model id picks whichever free model is available, so two runs of the same prompt can land on different models and give different words. About 50 requests a day, 20 a minute. `rag/llm.py` caches every reply under `data/llm_cache.json`, keyed by the exact prompt, so rerunning a script is free and gives the same answer; the counter it prints is how many real requests went out. Plan a day's work in requests, the way you planned P08's embedding in minutes.

**Walkthrough — the chapter, then the model on your queries**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P11 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/llm
```

*If P11 is not approved yet:* `git switch dev/hybrid && git switch -c dev/llm`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 7, lessons 1 to 3, every lesson green; the key in both .env files and nowhere else
- [ ] scratch/p12-llm.md: the four things, the three forbidden, the two limits
- [ ] Step 3: the first call from the rag repo, twice, in evidence/P12.md
- [ ] Step 4: three typo'd queries corrected and searched, the table
- [ ] Predictions for P13 in evidence/P12.md, committed
- [ ] Fill every slot in evidence/P12.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 7, lessons 1 to 3.**

In `~/version_control/rag-search-engine/`. Stop after lesson 3; lessons 4 and 5 are P13. When the chapter page shows lessons 1 to 3 complete, save it as `evidence/p12-bootdev-ch7a.png` in the rag repo.

**Lesson 1: what a model can do before a search.** A reading and one multiple-choice question. The answer is the one that says the model processes the query before the search.

**Lesson 2: the client.** Skip the account step; the key is mine. Put it in `.env` at the root of the course project:

```text
OPENROUTER_API_KEY="the-key-I-handed-you"
```

Check that `.env` is in the course project's `.gitignore` (add the line if it is not), then add the two libraries:

```bash
uv add python-dotenv==1.1.0
uv add openai==2.44.0
```

Create `cli/test_llm.py`. The `load_dotenv()` line reads `.env` into the environment; the client is the OpenAI library pointed at OpenRouter's address; the model id is the router:

```python
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]

response = client.chat.completions.create(model="openrouter/free", messages=messages)
print(response.choices[0].message.content)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Response tokens: {response.usage.completion_tokens}")
```

`messages` is a list of turns; this one has a single user turn. `response.usage` is where the two token counts live.

Run, then submit:

```bash
uv run cli/test_llm.py
bootdev run -s e42152ba-a4ad-4578-9688-13d3f37e5bee
```

*You should see* a paragraph about Boot.dev, then `Prompt tokens:` and `Response tokens:` with numbers; the two labels are the check. This lesson's submission is marked safe: a failed check costs nothing, and the chapter says to retry if the provider was the problem. *If it broke* with `RuntimeError: OPENROUTER_API_KEY environment variable not set`, `.env` is in the wrong folder or the line has a typo in the name. *If it broke* with `429` or `rate limit` in the message, wait a minute; the free tier counts per minute as well as per day.

**Lesson 3: spelling.** Two things: a helper that every later lesson calls, and the first `--enhance` option on `rrf-search`. Create `cli/lib/llm.py` so the client is built once and the prompt-to-reply call is one function:

```python
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


# In: nothing.  Out: the OpenAI client pointed at OpenRouter, built once, with the key from .env.
def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


# In: a prompt.  Out: the model's reply as a string, with surrounding whitespace removed.
def ask(prompt, model="openrouter/free"):
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()
```

`global _client` lets the function change the variable outside it, so the second call finds the client already built. Then create `cli/lib/query_enhancement.py` with the chapter's prompt. `PROMPTS` is a dictionary from a method name to its prompt; lessons 4 and 5 add two more entries:

```python
from lib.llm import ask

SPELL = """Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

PROMPTS = {"spell": SPELL}


# In: a method name and the query.  Out: the enhanced query.
def enhance_query(method, query):
    reply = ask(PROMPTS[method].format(query=query)).strip('"')
    return reply
```

`.format(query=query)` fills the `{query}` hole in the prompt; `.strip('"')` removes the quotes a model sometimes puts around its answer. In `cli/hybrid_search_cli.py`, import it and add the option to the `rrf-search` parser:

```python
from lib.query_enhancement import enhance_query
```

```python
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        help="Query enhancement method",
    )
```

Then, at the top of the `rrf-search` case, before the searcher is built, enhance the query and print the chapter's line:

```python
        case "rrf-search":
            query = args.query
            if args.enhance:
                enhanced = enhance_query(args.enhance, query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            searcher = HybridSearch(load_movies())
```

and use `query` instead of `args.query` in the search call and the results heading below it. Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "briish bear" --enhance spell
bootdev run -s 92edfe47-1d5a-4063-9676-b6900c851f45
```

*You should see* `Enhanced query (spell): 'briish bear' -> 'British bear'` or close to it, then the RRF results for the corrected query with `Paddington` near the top; the word `spell` in the output is the check. *If it broke* with the results heading still showing `briish bear`, the heading uses `args.query`; change it to `query`.

**Step 3. The first call from the rag repo.**

In the rag repo, the key goes in a second `.env`, at the repo root, same one line. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
uv add python-dotenv openai
uv run python -m rag.llm
uv run python -m rag.llm
```

Open `rag/llm.py` first and read `chat` and `chat_messages`: the same client as lesson 2, plus a cache. The first run sends one request; the second run finds the same prompt in `data/llm_cache.json` and sends none.

*You should see* a paragraph, `Prompt tokens:`, `Response tokens:`, and `requests made this run: 1` the first time and `0` the second. The paragraph is identical both times, because the second one is the cached first one. Paste the first run under `## Step 3` in `evidence/P12.md` and the second run's last line beside it. *If it broke* with `OPENROUTER_API_KEY is not set`, the `.env` is in the course project and not the rag repo; each repo reads its own. *If it broke* with `the model did not answer after several tries`, the router had nothing free to give for a minute; `rag/llm.py` already waited and retried three times, so wait a minute more and run again.

**Step 4. Spelling, on three of your queries.**

Take three of your five `rag/queries.json` queries and misspell one word in each, the way a reader would: a dropped letter, a doubled one, a swapped pair. Then:

```bash
uv run python -m rag.enhance spell "<typo'd query 1>"
uv run python -m rag.enhance spell "<typo'd query 2>"
uv run python -m rag.enhance spell "<typo'd query 3>"
```

Open `rag/enhance.py` first: the chapter's prompt with one line added, `CORPUS_NOTE`, that tells the model what the query is going to be searched against. Set it to one line about your corpus; the model fixes `Ciclops` to `Cyclops` more readily when it knows the text is Homer.

*You should see* an `Enhanced query (spell)` line and the hybrid top five for the corrected query. On Homer, with `CORPUS_NOTE` set to the Iliad and Odyssey:

```text
{{LLM:P12-spell}}
```

Then the ranks. Run `uv run python -m rag.search "<typo'd query>"` and `uv run python -m rag.search "<corrected query>"` for each and find your needle chunk's rank in the `hybrid` list (the scoreboard does this for the clean query already). On Homer, the typo'd `Ciclops` query had its answer chunk at hybrid rank 6 and the clean one at rank 1; `captian` for `captain` put a chunk at rank 6 that the clean query did not find in the top ten at all, because the misspelled word matched nothing and the other words did the work; `Nausica` for `Nausicaa` did not move the rank 1 result, because the semantic engine does not spell. Fill the Step 4 table in `evidence/P12.md` and write what, if anything, the model changed that was not a typo.

*If it broke* with the model returning your query with a word "corrected" to something else, that is the Reflection's third question; keep the output.

**Predictions for P13.** P13 runs the chapter's `expand` and `rewrite` on ten queries: your five from `rag/queries.json` and five more you write tomorrow. Before any of that, write under `## Predictions for P13` in `evidence/P12.md`, for each of your five, whether expansion will help, hurt, or not change the hybrid rank of the answer chunk, and how many of the ten you expect to be helped. Commit the file. I check your commit timestamps.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P12.md` (the questions are below) and run `uv run python scripts/check_evidence.py P12` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/ rag/enhance.py pyproject.toml uv.lock
git commit -m "P12: the model client with a cache, spelling on three queries, P13 predictions"
git push -u origin dev/llm
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P13.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P12.md` with every slot filled + `scratch/p12-llm.md` + `evidence/p12-bootdev-ch7a.png` + the key in `.env` and in no commit, ever.

**Reflection Questions** (answer under `## Reflection` in `evidence/P12.md`)

1. Paste your Step 3 output. Say what the cache in `rag/llm.py` does on the second run, why the free tier makes it worth having, and what would go wrong with the cache if you edited a prompt and kept the old reply.

   *How to get it:* the cache key is a fingerprint of the model id and the exact prompt, so an edited prompt is a new key and a new request; the old reply stays filed under the old prompt and is never used for the new one. The danger is the other way: a prompt you did not edit, whose reply you want fresh, comes back cached.

2. Paste your Step 4 table. For the query whose rank changed most, say which engine the typo had broken (keyword, semantic, or both) and why, using what lesson 1 of chapter 6 said about each engine's blind spots.

   *How to get it:* BM25 matches tokens, so a misspelled name is a token that is in no chunk, and the query loses its rarest, most valuable word. The semantic model turns `Nausica` into nearly the same vector as `Nausicaa`, so it barely notices.

3. Lesson 3 told the model to change nothing but clear typos. Say what the search would do with a model that "helpfully" rewrote a name in your corpus to its common spelling, and name one query of yours where that would hurt.

   *How to get it:* Butler writes `Ulysses` and `Minerva`; a model that corrects them to `Odysseus` and `Athena` hands BM25 words that appear in no chunk. Your corpus has its own spellings; the prompt's "do not rewrite" line is what protects them.
