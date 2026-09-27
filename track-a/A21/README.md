# A21 · Three Kinds of Memory

**Meetings:** D40–D42 · **Points:** 15 pts


**Watch — 60 min**
Day 1 — 24 min
[DeepLearning.AI, *Long-Term Agentic Memory with LangGraph*](https://www.deeplearning.ai/short-courses/long-term-agentic-memory-with-langgraph/) lesson 2, Introduction to Agent Memory (8m) · lesson 3, Baseline Email Assistant (16m)
Day 2 — 21 min
Lesson 4, Semantic Memory (12m) · lesson 5, Semantic + Episodic (9m)
Day 3 — 15 min
Lesson 6, adding procedural memory (15m)

Wed Dec 2 is also the Milestone A1 defense. Watch and work while you wait for your five minutes.

**During the video**

Lessons 3 to 6 are notebooks on the DeepLearning.AI page. Same rule as A17: a cell under each of his code cells, his code typed into it, yours run. Copy what you typed into `scratch/a21-video.py` in your rag repo before you close the tab each day.

**Lesson 2 · no code; build a table.** In `scratch/a21-video.py`, as comments, a three-row table: semantic, episodic, procedural. For each, the human example he gives and the agent example he gives. Leave a third column empty. You fill it on Day 3 with your own agent's example.

**Lesson 3 · type and run.** He builds an email assistant as a graph: a router that decides what kind of email it is, then an agent with tools. Type the state definition and the graph wiring. Pause where he compiles the graph and write, as a comment, which node runs first and what decides the edge out of it.

**Lesson 4 · type and run.** The store with an embedding index, and the two memory tools that let the model write and search it. Type the namespace he passes to the tools and write, as a comment, what each part of the tuple is for.

**Lesson 5 · type and run.** Past examples stored in the store and pulled back into the prompt. Note who writes those examples: the model, or his code.

**Lesson 6 · type and run.** The agent's instructions live in the store, and feedback rewrites them. Pause on the call that rewrites them and write, as a comment, what stops a bad piece of feedback from rewriting the agent into something worse.

**Notes**

**Context compaction is not memory.** Your v2 agent summarized a long conversation so it fit. That is compression inside one run. Memory is state that outlives the run, and it is a different problem with different failures.

**A LangGraph state field without a reducer is overwritten, not appended.** Each node's return value replaces that key. Annotate a message list with `add_messages` or your history resets to the last node's output every step, with no error. Step 3 on Day 1 shows you in twenty lines.

**The checkpointer and the store are not the same thing.** The checkpointer saves one conversation thread so it can resume, keyed by `thread_id`. The store saves memories across threads, keyed by a namespace tuple like `("corpus_agent", "me", "facts")`. Put long-term memories in the checkpointer and they vanish the moment a new conversation starts.

**`InMemoryStore` means in memory.** Everything in it is gone when the process exits. That is fine this week, because every test runs inside one process. A22 is where memory survives a restart.

**The course's agent reads email. Yours reads your corpus.** The walkthrough builds the same three kinds of memory into a small Python agent in your rag repo, with your A18 retriever as its tool.

**Walkthrough — memory for a reader of your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/kb-server && git pull        # skip this line if A20 has merged
git switch -c dev/memory
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Lesson 2 table in `scratch/a21-video.py`
- [ ] Lesson 3 Baseline Email Assistant, typed along in the page
- [ ] `scratch/a21-reducer.py`: one graph with a reducer, one without (Step 3)
- [ ] `memory/agent.py` Day 1 version, one thread vs two (Steps 4–5)
- [ ] Push, PR, sign off
```

**Step 2. Install.**

```bash
uv add langgraph langmem langchain-openai
```

**Step 3. `scratch/a21-reducer.py`: the reducer, seen once.**

```python
# scratch/a21-reducer.py
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

class Plain(TypedDict):
    messages: list

class Reduced(TypedDict):
    messages: Annotated[list, add_messages]

def a(state):
    return {"messages": [("ai", "from a")]}

def b(state):
    return {"messages": [("ai", "from b")]}

for State in (Plain, Reduced):
    g = StateGraph(State)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    out = g.compile().invoke({"messages": [("user", "hi")]})
    print(State.__name__, len(out["messages"]), "messages")
```

```bash
uv run python scratch/a21-reducer.py
```

*You should see* `Plain 1 messages` and `Reduced 3 messages`. Without the reducer, `b`'s return value replaced everything before it, your own message included.

**Step 4. `memory/agent.py`, Day 1: one tool, one checkpointer.**

```python
# memory/agent.py
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from rag.llm import CHAT_MODEL
from rag.rerank import retrieve

USER = "me"
FIXED = ("You answer questions about <your corpus title> for one reader. "
         "Call search_corpus before answering any question about it. "
         "End every sentence that uses a passage with its id, like [c00412].")
STYLE = "Answer in plain prose, in three sentences or fewer."

@tool
def search_corpus(query: str) -> str:
    """Search <your corpus title> and return five passages, each starting with its id."""
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in retrieve(query, k=5))

checkpointer = InMemorySaver()
agent = create_react_agent(f"openai:{CHAT_MODEL}", tools=[search_corpus],
                           prompt=FIXED + "\n\n" + STYLE, checkpointer=checkpointer)

def ask(text, thread):
    out = agent.invoke({"messages": [{"role": "user", "content": text}]},
                       config={"configurable": {"thread_id": thread, "langgraph_user_id": USER}})
    return out["messages"][-1].content
```

Write your corpus title into `FIXED` and into the tool's docstring by hand. The docstring is the tool description the model reads, and Python does not fill in a variable there. `create_react_agent` may print a deprecation warning pointing at a newer function; it is the one the course uses, so ignore the warning.

`FIXED` and `STYLE` are separate on purpose. Day 3 lets feedback rewrite the style. It never gets to rewrite the rules.

**Step 5. One process, two threads.** Every script this week runs a list of steps inside one process, because `InMemorySaver` dies with it:

```python
# memory/run_script.py
import json
import sys

from memory.agent import ask

steps = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for step in steps:
    print(f"[{step['thread']}] > {step['say']}")
    print("   ", ask(step["say"], step["thread"])[:300].replace("\n", " "))
```

`memory/scripts/day1.json`:

```json
[
  {"thread": "t1", "say": "My name is Sam and I am reading this for a history class."},
  {"thread": "t1", "say": "What is my name, and why am I reading this?"},
  {"thread": "t2", "say": "What is my name, and why am I reading this?"},
  {"thread": "t2", "say": "<a golden question from rag/eval/golden.json>"}
]
```

```bash
uv run python -m memory.run_script memory/scripts/day1.json
```

*You should see* thread `t1` answer both parts, thread `t2` not know either, and the golden question answered with a chunk id. The checkpointer remembers a thread. Nothing remembers a person.

```bash
git add memory/agent.py memory/run_script.py memory/scripts/day1.json scratch/a21-reducer.py scratch/a21-video.py pyproject.toml uv.lock
git commit -m "A21: corpus agent with a checkpointer, one thread vs two"
git push -u origin dev/memory
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Sign off the log.

**Day 2 starts here.** `git switch dev/memory && git pull`; `bash scripts/start-entry.sh` in the log repo; then:

```markdown
- [ ] Lessons 4 and 5, typed along in the page
- [ ] Store, semantic memory tools, episodes (Steps 6–7)
- [ ] Day 1 script re-run: thread t2 now knows (Step 7)
- [ ] Push, sign off
```

**Step 6. Semantic and episodic memory.** Replace `memory/agent.py` with this. What is new: a store with an embedding index, the two `langmem` tools from lesson 4 writing facts to it, and episodes from lesson 5, written by your code after every cited answer and pulled back into the prompt.

```python
# memory/agent.py
import re
import uuid

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from rag.llm import CHAT_MODEL
from rag.rerank import retrieve

USER = "me"
FIXED = ("You answer questions about <your corpus title> for one reader. "
         "Call search_corpus before answering any question about it. "
         "End every sentence that uses a passage with its id, like [c00412].")
STYLE = "Answer in plain prose, in three sentences or fewer."
FACTS = ("corpus_agent", "{langgraph_user_id}", "facts")
EPISODES = ("corpus_agent", USER, "episodes")
INSTRUCTIONS = ("corpus_agent", USER, "instructions")

@tool
def search_corpus(query: str) -> str:
    """Search <your corpus title> and return five passages, each starting with its id."""
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in retrieve(query, k=5))

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
checkpointer = InMemorySaver()

def style():
    item = store.get(INSTRUCTIONS, "style")
    return item.value["text"] if item else STYLE

def prompt(state):
    question = next(m.content for m in reversed(state["messages"]) if m.type == "human")
    shots = get_store().search(EPISODES, query=question, limit=2)
    examples = "\n".join(f"Q: {e.value['question']} -> cited {', '.join(e.value['cited'])}" for e in shots)
    system = FIXED + "\n\n" + style()
    if examples:
        system += "\n\nSimilar questions you answered before:\n" + examples
    return [{"role": "system", "content": system}] + state["messages"]

agent = create_react_agent(
    f"openai:{CHAT_MODEL}",
    tools=[search_corpus, create_manage_memory_tool(namespace=FACTS),
           create_search_memory_tool(namespace=FACTS)],
    prompt=prompt, checkpointer=checkpointer, store=store)

def ask(text, thread):
    out = agent.invoke({"messages": [{"role": "user", "content": text}]},
                       config={"configurable": {"thread_id": thread, "langgraph_user_id": USER}})
    reply = out["messages"][-1].content
    cited = sorted(set(re.findall(r"\[(c\d{5})\]", reply)))
    if cited:
        store.put(EPISODES, uuid.uuid4().hex[:8], {"question": text, "cited": cited})
    return reply
```

`{langgraph_user_id}` in `FACTS` is filled in by `langmem` from the `config` at call time, which is how one agent keeps two readers' facts apart. `prompt` is a function now: it runs before every model call, looks up the two most similar past questions, and puts them in the system message. The model decides what goes into `facts`. Your code decides what goes into `episodes`. That difference is on the reflection.

**Step 7. Dump the store after a script.** Add to the bottom of `memory/run_script.py`, and change its import line to `from memory.agent import USER, ask, store`:

```python
print("\nstore after the script:")
for kind in ("facts", "episodes"):
    for item in store.search(("corpus_agent", USER, kind), limit=100):
        print(f"  {kind:8} {item.key[:8]}  {json.dumps(item.value)[:120]}")
```

```bash
uv run python -m memory.run_script memory/scripts/day1.json
```

*You should see* thread `t2` now answer at least one part of the name-and-reason question, because the model wrote a fact in `t1` and searched for it in `t2`. The dump shows one or more `facts` and one `episodes` row for the golden question. If `t2` still does not know, check the dump: no `facts` rows means the model never called `manage_memory`, which is a prompt problem, not a store problem.

```bash
git add memory/agent.py memory/run_script.py scratch/a21-video.py
git commit -m "A21: semantic facts via langmem, episodes written by code"
git push
```

Sign off the log.

**Day 3 starts here.** `git switch dev/memory && git pull`; `bash scripts/start-entry.sh`; then:

```markdown
- [ ] Lesson 6, typed along in the page; third column of the Lesson 2 table
- [ ] `teach` with a length cap and a human yes (Step 8)
- [ ] Policy script committed BEFORE it runs (extension)
- [ ] `memory/taxonomy.md`, `memory/retention_policy.md` draft, baseline numbers
- [ ] Push, sign off
```

**Step 8. Procedural memory, with a guardrail.** Add to `memory/agent.py`:

```python
import difflib

from rag.llm import chat

def teach(feedback):
    old = style()
    new = chat([{"role": "system", "content": "Rewrite these style instructions for an assistant so they follow "
                 "the feedback. Keep what the feedback does not mention. Return only the instructions."},
                {"role": "user", "content": f"INSTRUCTIONS:\n{old}\n\nFEEDBACK:\n{feedback}"}], cache=False)
    if len(new) > 600:
        return f"REJECTED: {len(new)} characters, the cap is 600"
    print("\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), "old", "new", lineterm="")))
    if input("apply? [y/N] ").strip().lower() != "y":
        return "not applied"
    store.put(INSTRUCTIONS, "style", {"text": new, "feedback": feedback})
    return "applied"
```

```bash
uv run python -c "
from memory.agent import ask, teach
q = '<a golden question>'
print(ask(q, 'a'))
print(teach('Quote the exact words from the passage before explaining them.'))
print(ask(q, 'b'))
"
```

*You should see* a diff, a prompt for `y`, and after `y` a second answer in the new style that still cites chunk ids. Only `STYLE` is rewritable. The rule to search and cite lives in `FIXED`, in code, where no feedback reaches it. Answer `n` once too, and see that nothing changed.

**Extension — what should this agent remember? (ASSIGNED)**

**Part 1. The policy script, before it runs.** `memory/scripts/policy.json`: twelve things a real reader of your corpus might say, all in thread `p1`, then one step in thread `p2`: `"What do you remember about me?"`. Each of the twelve carries `"intent"`: `never`, `expiring`, `durable` or `none` (nothing worth storing). At least two `never`: one that looks like a password, and one that starts "don't remember this". Write the password as words (`my password is orchid-lantern-42`), never in the shape of a real key: the secret guard blocks key-shaped strings in commits.

```bash
git add memory/scripts/policy.json
git commit -m "A21: twelve-statement policy script with intents, before any run"
git push
```

I will check your commit timestamps.

**Part 2. Baseline.** Run it through today's agent, which has no policy at all:

```bash
uv run python -m memory.run_script memory/scripts/policy.json | tee memory/baseline.txt
```

In `memory/retention_policy.md`, a table with one row per statement: its intent, whether a `facts` row for it appeared in the dump, and whether that is right. Then two counts out of twelve: stored when it should not have been, and not stored when it should have been. A22 measures the same script against the policy you enforce.

**Part 3. The policy, drafted.** In the same file, the four categories for an agent that reads *your* corpus with one reader: never written, written but expiring (with a duration and why), written and durable, deletable on request (and how). Name real things from your corpus domain, not "preferences".

**Part 4. `memory/taxonomy.md`.** For semantic, episodic and procedural: one concrete example for your corpus agent, its storage shape (namespace tuple, key, value fields), who writes it (the model or your code), and how it comes back (searched by similarity, fetched by key, or always in the prompt). "A vector store" is not an answer for all three.

**Part 5. Where it got worse.** Episodes are pulled into the prompt; that can help or drag. In one process, ask five golden questions, then ask them again, and compare the chunk ids cited each time against `chunk_ids` in `golden.json`:

```bash
uv run python -c "
import json, re
from memory.agent import ask
gold = json.load(open('rag/eval/golden.json'))[:5]
for rnd in (1, 2):
    for g in gold:
        cited = set(re.findall(r'\[(c\d{5})\]', ask(g['question'], f'r{rnd}-{g[\"id\"]}')))
        print(rnd, g['id'], sorted(cited), 'hit' if cited & set(g['chunk_ids']) else 'miss')
"
```

Count how many of the five changed between rounds, and in which direction. If none changed, say so with the output.

```bash
git add memory/agent.py memory/baseline.txt memory/retention_policy.md memory/taxonomy.md scratch/a21-video.py
git commit -m "A21: procedural style with a guardrail, taxonomy, policy draft and baseline"
git push
```

One PR, not three; push again each day. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`memory/agent.py` + `memory/run_script.py` + `memory/scripts/day1.json` + `memory/scripts/policy.json` (committed before its run) + `memory/baseline.txt` + `memory/taxonomy.md` + `memory/retention_policy.md` (draft and baseline counts) + `scratch/a21-reducer.py` + `scratch/a21-video.py`.

**Reflection Questions**

1. Paste the `store after the script:` dump from your policy run. Pick one `facts` row the model wrote and one `episodes` row your code wrote. For each, say what decided it would be written, quoting the line of `memory/agent.py` or the tool instruction responsible, and which of the two you would trust to follow a rule like "never store passwords".
2. Paste your two baseline counts and the row of your table that surprised you most. Was the surprise something the model stored or something it skipped? Paste the exact `facts` value it wrote, or the statement it ignored, and say which of your four policy categories covers it.
3. Paste your Part 5 output. Add a line to that script that prints, in round 2 only, `[e.value for e in store.search(EPISODES, query=g['question'], limit=2)]` (import `store` and `EPISODES` from `memory.agent`), and run it again. For one question whose citations changed between rounds, or the weakest one if none did, paste the two episodes it retrieved and say whether they explain what happened.
