# A22 · Agentic Retrieval and a Retention Policy

**Meetings:** D43 · **Points:** 15 pts

**Watch**
Mon Dec 7 — no new video. A21's lessons 4 to 6 are the reference; reopen them rather than guessing.

**During the video**

No video. Before any code, open `memory/scripts/policy.json` and `memory/retention_policy.md` from A21 side by side. For each of your twelve statements, write in `scratch/a22-enforce.md` which line of code will enforce its intent after today: a regular expression, the kind a fact is saved under, or nothing. Any statement whose answer is "the model will know not to" is a statement your policy does not yet enforce. Step 3 is where you fix that.

**Notes**

**Agentic retrieval means the model controls the search.** It decides whether to search, rewrites the query, reads what came back, and searches again. That is more powerful than A18's fixed pipeline and harder to debug, because a bad answer is now a bad retriever *or* a bad decision to stop searching. Every search is logged with its query so you can tell which.

**Cap the loop.** A model that keeps rewriting a query it will never satisfy spends your key until the cap stops you, and the cap does not warn you. Three searches per question. The fourth call gets a message telling it to answer with what it has.

**Memory has to survive the process.** A21's store lived in RAM. Today facts go to a file, `data/memory.json`, and the test is the one that matters: ask, exit, start a new process, ask what it remembers. A memory system that only works inside one run is a variable.

**A policy the model is asked to follow is a request. A policy the code enforces is a policy.** The prompt can say "never store passwords" and the model can store one anyway, rephrased. The walkthrough checks two things in code before anything is written: the text being stored, and the reader's own message, because "don't remember this" is in the message and usually gone from the fact the model writes.

**Forgetting is the part that shows you thought about it.** "We store everything forever" is not a retention policy. Yours names what it refuses, what it expires and after how long, and how a reader deletes anything.

**Walkthrough — search that decides, memory that persists**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. A21 is due tomorrow morning, so branch from it:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/memory && git pull           # skip this line if A21 has merged
git switch -c dev/agentic-memory
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] `scratch/a22-enforce.md`: one enforcing line per policy statement
- [ ] `memory/filestore.py`: never-list, expiry, delete (Step 2)
- [ ] Agent: remember/forget tools, search budget, memories in the prompt (Step 3)
- [ ] `memory/chat.py`, cross-process test (Steps 4–5)
- [ ] Extension: policy script vs A21 baseline, agentic vs fixed on the golden set
- [ ] `memory/retention_policy.md` final
- [ ] Push, PR, sign off
```

**Step 2. `memory/filestore.py`.**

```python
# memory/filestore.py
import json
import re
import time
import uuid
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "memory.json"
DAY = 86400

NEVER = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b(password|passcode|api key|secret)\b", re.I),
    re.compile(r"don'?t remember this", re.I),
]
TTL_DAYS = {"durable": None, "expiring": 30}

def _load():
    return json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else []

def _save(items):
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")

def blocked(text):
    return next((rule.pattern for rule in NEVER if rule.search(text)), None)

def remember(user, text, kind):
    if rule := blocked(text):
        return f"REFUSED by policy ({rule}); nothing stored"
    if kind not in TTL_DAYS:
        return f"REFUSED: kind must be one of {sorted(TTL_DAYS)}"
    now = time.time()
    ttl = TTL_DAYS[kind]
    item = {"key": uuid.uuid4().hex[:8], "user": user, "text": text, "kind": kind,
            "created": now, "expires": now + ttl * DAY if ttl else None}
    _save(_load() + [item])
    return f"stored {item['key']} ({kind})"

def recall(user, now=None):
    now = now or time.time()
    return [m for m in _load() if m["user"] == user and (m["expires"] is None or m["expires"] > now)]

def forget(user, key):
    items = _load()
    kept = [m for m in items if not (m["user"] == user and m["key"] == key)]
    _save(kept)
    return f"deleted {len(items) - len(kept)}"
```

Edit `NEVER` and `TTL_DAYS` to match your A21 draft: every `never` statement in your script should match a rule, and every duration in your policy should be a number here. `data/` is git-ignored, so the reader's memories never go into a commit.

```bash
uv run python -c "
from memory import filestore as f
print(f.remember('test', 'my password is orchid-lantern-42', 'durable'))
print(f.remember('test', 'prefers short answers', 'expiring'))
import time; print(len(f.recall('test')), len(f.recall('test', now=time.time() + 31 * 86400)))
"
```

*You should see* the first refused with the rule that caught it, the second stored with a key, then `1 0`: present today, gone after its expiry. Delete the test rows with `forget('test', '<key>')` or by removing `data/memory.json`.

**Step 3. The agent: persistent facts, enforced rules, a search budget.** Four changes to `memory/agent.py`.

First, delete the two `langmem` tools from the `tools=[...]` list and the `langmem` import. The model no longer writes to a store you do not control.

Second, replace `search_corpus` and `USER`, and add the two memory tools and a per-question `turn` record. The imports go at the top of the file with the others:

```python
import json
import os
import time

from memory import filestore
from rag.corpus import ROOT

USER = os.environ.get("CORPUS_READER", "me")
MAX_SEARCHES = 3
SEARCH_LOG = ROOT / "data" / "search_log.jsonl"
turn = {"blocked": False, "searches": 0}

@tool
def search_corpus(query: str) -> str:
    """Search <your corpus title> and return five passages, each starting with its id.
    If the passages do not answer the question, rewrite the query and search again."""
    turn["searches"] += 1
    with SEARCH_LOG.open("a") as f:
        f.write(json.dumps({"time": time.time(), "n": turn["searches"], "query": query}) + "\n")
    if turn["searches"] > MAX_SEARCHES:
        return "Search budget used up. Answer from the passages you already have, or say you don't know."
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in retrieve(query, k=5))

@tool
def remember(fact: str, kind: str) -> str:
    """Save one fact about the reader for future sessions. kind is "durable" for stable facts
    they state about themselves, "expiring" for preferences or plans that may change."""
    if turn["blocked"]:
        return "REFUSED by policy: the reader's message matched the never-store list"
    return filestore.remember(USER, fact, kind)

@tool
def forget_memory(key: str) -> str:
    """Delete one saved memory by its key, when the reader asks you to forget something."""
    return filestore.forget(USER, key)
```

Put `remember` and `forget_memory` in the `tools=[...]` list next to `search_corpus`.

Third, in `prompt`, after `system = FIXED + "\n\n" + style()`, add what the agent knows about this reader:

```python
    known = "\n".join(f"- ({m['key']}, {m['kind']}) {m['text']}" for m in filestore.recall(USER))
    if known:
        system += "\n\nWhat you know about this reader (key and kind in parentheses):\n" + known
```

Fourth, at the top of `ask`, reset the turn and check the reader's message:

```python
    turn["searches"] = 0
    turn["blocked"] = filestore.blocked(text) is not None
```

Change `FIXED`'s search sentence to: `"Search before answering any question about it; if the passages do not answer it, rewrite the query and search again, at most three times."`

**Step 4. `memory/chat.py`: one process per command.**

```python
# memory/chat.py
import sys

from memory import filestore

cmd, *rest = sys.argv[1:]
if cmd == "ask":
    from memory.agent import USER, ask, turn
    print(ask(" ".join(rest), thread=f"cli-{USER}"))
    print(f"(searches: {turn['searches']}, blocked: {turn['blocked']})")
elif cmd == "list":
    from memory.agent import USER
    for m in filestore.recall(USER):
        print(m["key"], m["kind"], m["text"])
elif cmd == "forget":
    from memory.agent import USER
    print(filestore.forget(USER, rest[0]))
```

**Step 5. Kill it and ask again.**

```bash
uv run python -m memory.chat ask "I'm reading this for a class and I prefer answers with direct quotes."
uv run python -m memory.chat list
uv run python -m memory.chat ask "What do you know about me?"
uv run python -m memory.chat ask "My password for the class site is orchid-lantern-42, don't remember this."
uv run python -m memory.chat list
```

*You should see*, after the first command, `list` print one or two rows with kinds you would agree with. The third command is a new process that knows it. The fourth prints `blocked: True`, and the second `list` has nothing new. Then run `forget` on one key and `list` again: it is gone. That sequence, with its output, is what a grader should be able to repeat, and it goes in the retention policy.

*If it broke:* `list` empty after the first command means the model never called `remember`; read the `remember` docstring as if you were the model and make the trigger clearer. A row written with the password means `turn["blocked"]` was never set, which means the Step 3 change to `ask` is not in.

```bash
git add memory/filestore.py memory/agent.py memory/chat.py scratch/a22-enforce.md
git commit -m "A22: persistent facts with code-enforced policy, search budget of three"
git push -u origin dev/agentic-memory
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**.

**Extension — enforced policy and agentic search, measured (ASSIGNED)**

**Part 1. The policy script, against the A21 baseline.** Clear your test memories, then run the same twelve statements. `run_script` runs them in one process; the check happens in a second one.

```bash
rm -f data/memory.json
uv run python -m memory.run_script memory/scripts/policy.json > memory/enforced.txt
uv run python -m memory.chat list >> memory/enforced.txt
```

In `memory/retention_policy.md`, add a column to your A21 table for today, and the same two counts out of twelve. Put them next to the baseline counts. Then check expiry for real: for one `expiring` row, print `filestore.recall(USER, now=time.time() + 31 * DAY)` and show it is absent.

**Part 2. Agentic against fixed, on the golden set.**

```python
# memory/eval_agentic.py
import json

from memory.agent import MAX_SEARCHES, ask, turn
from rag.answer import answer
from rag.corpus import ROOT
from rag.scoreboard import norm

GOLD = json.loads((ROOT / "rag" / "eval" / "golden.json").read_text(encoding="utf-8"))
rows = []
for g in GOLD:
    agentic = ask(g["question"], f"eval-{g['id']}")
    searches = turn["searches"]
    fixed = answer(g["question"])["answer"]
    rows.append((g["id"], g["kind"], searches,
                 norm(g["answer"]) in norm(agentic), norm(g["answer"]) in norm(fixed)))
    print(*rows[-1])
n = len(rows)
print(f"agentic correct {sum(r[3] for r in rows)}/{n}   fixed correct {sum(r[4] for r in rows)}/{n}")
print(f"mean searches {sum(r[2] for r in rows) / n:.1f}   hit the cap {sum(r[2] > MAX_SEARCHES for r in rows)}")
```

Before you run it, write in `scratch/a22-enforce.md` the two correct counts you expect, agentic and fixed, and commit that file. I will check your commit timestamps.

```bash
git add scratch/a22-enforce.md && git commit -m "A22: agentic vs fixed prediction"
CORPUS_READER=eval uv run python -m memory.eval_agentic
```

`CORPUS_READER=eval` keeps these twenty questions out of your own memories. *You should see* twenty rows, then two correct counts and a mean number of searches between 1 and 4. A mean of exactly 1.0 means the model never rewrote a query; read `data/search_log.jsonl` to see whether it had reason to.

**Part 3. Where it got worse.** Pick the golden question where agentic got it wrong and fixed got it right, or, if there is none, where agentic used the most searches. Paste its queries from `data/search_log.jsonl` in order. Say whether the failure was the retriever (a good query that got bad passages) or the decision (a bad rewrite, or stopping early).

**Part 4. `memory/retention_policy.md`, final.** The four categories with your enforcement written next to each one: the regex, the TTL, or the command. The cross-process demonstration from Step 5 as a numbered sequence a grader can repeat. The two A21 counts and the two A22 counts. One sentence on what happens if `data/memory.json` is deleted.

```bash
git add memory/eval_agentic.py memory/enforced.txt memory/retention_policy.md
git commit -m "A22: enforced policy vs baseline, agentic vs fixed on the golden set"
git push
```

Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`memory/filestore.py` + `memory/agent.py` (remember/forget tools, search budget) + `memory/chat.py` + `memory/eval_agentic.py` + `memory/enforced.txt` + `memory/retention_policy.md` (final, with A21 and A22 counts) + `scratch/a22-enforce.md`.

**Reflection Questions**

1. Paste your A21 baseline counts and today's counts. For one statement whose outcome changed, paste the line of `memory/filestore.py` or `memory/agent.py` responsible. For any `never` statement still stored today, paste the `facts` text the model wrote and say which rule it slipped past and how.
2. Paste the `search_log.jsonl` lines for the golden question with the most searches, and the `eval_agentic` row for it. Was the second query better than the first? Name the word it added or dropped and what that did to the passages that came back.
3. Paste the prediction you committed before Part 2, then the two correct counts and the mean searches. If agentic lost or tied, name the golden `kind` it lost on; if it won, name the cost in searches and say whether a reader waiting for the answer would notice.
