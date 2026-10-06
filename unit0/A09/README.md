# A09 · Transformer LLMs 10 + Sampling by Hand

**Meetings:** D19 · **Points:** 15 pts

**Watch — 21 min**

**One meeting — 21 min**
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 10, Model Example (9m, code)
[Let's reproduce GPT-2 (124M), Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU) · segment 00:33:31 → 00:45:50 (12m): sampling init, prefix tokens, tokenization, the sampling loop, auto-detecting the device
 · Do Steps 2 and 3 before you press play: the GPT-2 download is about half a gigabyte and finishes while you watch, and `scratch/a09-video.py` below needs the three prompts from Step 3.

Optional, not in the total: lesson 11, Recent Improvements (10m), and lesson 12, Mixture of Experts (9m). Lesson 12 is where the best explanation for Step 7 comes from.

**During the video**

He samples from a model he wrote, with a prompt he made up. You run the same loop on the Hugging Face GPT-2, with a prompt cut from your corpus, from one file you make before the video starts. Create `scratch/a09-video.py` (right-click `scratch`, **New File**) and paste all of this in. It needs `sampling/lab.py` from Step 3 and nothing from the API.

```python
# scratch/a09-video.py
# Run from the repo root:  uv run python scratch/a09-video.py
# Karpathy's sampling loop, 00:33:31 -> 00:45:50, with two things swapped: the Hugging Face GPT-2 in place
# of the GPT class he wrote earlier in the video, and the first prompt from your corpus in place of
# "Hello, I'm a language model,". Needs sampling/lab.py from Step 3.
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # so "from sampling.lab" works
from sampling.lab import model, enc, PROMPTS, pick_prompts           # the same GPT-2 and the same tokenizer lab.py uses

num_return_sequences = 5      # his first two lines. His max_length counts the prompt; his prompt is 8 tokens
max_length = 50               # and yours is 20, so 50 here gives thirty new tokens, the same as lab.py's n_new=30

device = "cpu"                # his device detection, from the end of the segment
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)
model.to(device)

prompt = PROMPTS[0] if PROMPTS else pick_prompts()[0]        # the first prompt "pick" printed, from your corpus
print("prompt:", repr(prompt))
tokens = enc.encode(prompt)                                  # his "prefix tokens": the prompt as GPT-2 token ids
tokens = torch.tensor(tokens, dtype=torch.long)              # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, T): the same prompt five times, one row each
x = tokens.to(device)
print("x", tuple(x.shape))

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x).logits                   # his line is  logits = model(x). Hugging Face returns an object; .logits is the tensor
        logits = logits[:, -1, :]                  # (5, 50257): only the last position predicts the next token
        probs = F.softmax(logits, dim=-1)          # scores -> probabilities, one row per sequence, each row sums to one
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)   # (5, 50) each: the 50 most likely, and which tokens they are
        ix = torch.multinomial(topk_probs, 1)      # (5, 1): one draw among the 50, weighted by probability. This is top-k.
        xcol = torch.gather(topk_indices, -1, ix)  # (5, 1): the token id each draw stands for
        x = torch.cat((x, xcol), dim=1)            # append it; T grows by one

print(f"x after the loop {tuple(x.shape)}: {max_length - tokens.shape[1]} new tokens per row\n")
for i in range(num_return_sequences):
    out = x[i, :max_length].tolist()
    print(">", repr(enc.decode(out)))
```

Run it once before you press play:

```bash
uv run python scratch/a09-video.py
```

*You should see* the device line, your prompt in quotes, `x (5, T)` with your `T` near 20, a line saying `x after the loop (5, 50): 30 new tokens per row` (the second number is `50 - T`), and five continuations starting with `>`, each beginning with your prompt and going on for thirty tokens, all five different. On a laptop CPU the loop takes under a minute. Paste the five into your log; the Karpathy pause points below each point at one line of this file.

<!-- JD: fill the Homer T and one of the five continuations after a run. -->

*If it broke* with `ModuleNotFoundError: No module named 'sampling'`, Step 3 is not done: `sampling/lab.py` has to exist. `FileNotFoundError: data/corpus.txt` means the terminal is not in the repo root; `pwd` should end in `foundations-<your-username>`.

**DLAI lesson 10 · write down the shapes.** No typing. Run the notebook in the DLAI page as he goes; the model he loads is too big to be pleasant on a laptop. Every time he prints a shape, write it in your log with what it is. The two that matter are what comes out of the body of the model and what comes out of the head on top of it. Step 4 prints the same two out of GPT-2, and they should look alike with different numbers.

**Karpathy 00:33:31 → 00:45:50 · read along; the file has already run.** He is using the GPT class he wrote earlier in the video, loaded with GPT-2's weights; your file loads the same weights from Hugging Face. Five pause points, one line in your log each:

1. When he types `num_return_sequences = 5` and `max_length = 30`: your file has the same two names, with `max_length = 50`. Write down why the number differs (his prompt is 8 tokens, yours is 20, and both want about thirty new ones).
2. When he encodes `"Hello, I'm a language model,"` with `tiktoken.get_encoding("gpt2")` and does `unsqueeze(0).repeat(...)`: your `x (5, T)` line is the same shape with your prompt's length in place of his 8. Write down both shapes.
3. When he types `logits = model(x)`: the one line that differs. His model returns the logits; Hugging Face returns an object, so your line reads `model(x).logits`. Write down what `logits[:, -1, :]` keeps and why only that row matters.
4. When he types `torch.topk(probs, 50, dim=-1)` and `torch.multinomial`: that is top-k, and it is the only knob in his loop. **Temperature and top-p do not appear.** Write that down; Step 5 adds both by hand.
5. When his five continuations print: they are about language models, because that is what his prompt said. Read yours again. Write one line on whether GPT-2 stayed in your corpus's voice or drifted toward something it read more of. When he adds device detection, your file already has it at the top; on a Mac it printed `mps`.

**Notes**

**Two tokenizers, two vocabularies.** GPT-2 reads ids from `tiktoken.get_encoding("gpt2")`, 50,257 of them. Hand it ids from `cl100k_base` (A08's, 100,277) and you get `IndexError: index out of range in self`, or, for ids that happen to be small enough, fluent garbage.

**Temperature 0 is not a division.** Logits divided by zero are infinities. The sampler treats `temperature == 0` as "take the highest-probability token" before it divides anything.

**The graded sampler is NumPy.** PyTorch appears in one function, `next_logits`, which runs GPT-2 and hands back a NumPy array of 50,257 numbers. Everything that turns those numbers into a choice is in `sample()`, in NumPy, and you can read every line of it.

**The API is not your laptop.** Temperature 0 on GPT-2 on your machine is one computation, done the same way every time. Temperature 0 on the API runs on shared hardware alongside other people's requests. Step 7 is about the difference.

**Some models refuse the knobs.** If the API answers `400` naming `temperature` or `logprobs` as unsupported, `MODEL` is a reasoning model. Use a non-reasoning model for this lab and write down which.

**Walkthrough — temperature, top-k and top-p on your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/sampling
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2, GPT-2 downloaded
- [ ] Step 3, lab.py started, three prompts from my corpus pasted into PROMPTS
- [ ] scratch/a09-video.py run once, five continuations in the log
- [ ] DLAI lesson 10, shapes written down
- [ ] Karpathy 00:33:31 → 00:45:50 read along, five pause-point lines in the log
- [ ] Step 4, two shapes printed from GPT-2
- [ ] Steps 5–6, sampler pasted and checked, snapshots saved
- [ ] Step 7, twenty calls at temperature 0 on the API
- [ ] Extension: predictions committed, table run, usable column filled, RESULTS.md finished
- [ ] Push and open the PR
```

**Step 2. Install and download.**

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv add --dev transformers
mkdir -p sampling
uv run python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"
```

*You should see* a download progress bar and then nothing. The weights are cached in your home folder; the next load takes seconds.

**Step 3. Make `sampling/lab.py` and pick three prompts from your corpus.**

Create `sampling/lab.py` (right-click `sampling`, **New File**). Two pastes. First, the top of the file: imports, the model, the two constants, and the two functions that touch GPT-2.

```python
# sampling/lab.py
# Run from the repo root:  uv run python sampling/lab.py <command>
# Commands, one per step of A09: pick, shapes, check, snapshot, api0, table.
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
import tiktoken, torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # so "from search.search" works from any folder
from search.search import chunk                  # A06's chunker, so the prompts are cut the way the corpus was chunked

enc = tiktoken.get_encoding("gpt2")              # GPT-2's tokenizer. Not cl100k_base: different ids, different vocabulary.
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
MODEL = "gpt-4o-mini"  # the model your v2 agent calls, if it takes temperature and logprobs
PROMPTS = []           # filled in once, in Step 3, from what "pick" printed, and never changed after

def pick_prompts(n=3, seed=0, n_tokens=20):
    """ the first n_tokens GPT-2 tokens of n chunks of your corpus, chosen with a fixed seed """
    chunks = chunk((ROOT / "data" / "corpus.txt").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    return [enc.decode(enc.encode(chunks[i])[:n_tokens]) for i in rng.choice(len(chunks), n, replace=False)]

def next_logits(ids):
    """ run GPT-2 on a list of token ids and hand back the logits for the next token, as a NumPy array of 50,257 numbers """
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1].numpy().astype(np.float64)
```

Second, the main block, at the very bottom of the file. It is complete now: one branch per command, for every step of today, and you never add to it. Steps 5 to 7 paste functions **above** it; the branches that call them will not run until those functions exist, and `pick` does not need them.

```python
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd != "pick":
        assert len(PROMPTS) == 3, "paste the three strings that 'pick' printed into PROMPTS first (Step 3)"
    if cmd == "pick":                                     # Step 3
        for p in pick_prompts():
            print(repr(p))
    elif cmd == "shapes":                                 # Step 4
        x = torch.tensor([enc.encode(PROMPTS[0])])
        with torch.no_grad():
            hidden = model.transformer(x).last_hidden_state   # what comes out of the body of the model
            logits = model(x).logits                          # what comes out of the head on top of it
        print("hidden state", tuple(hidden.shape), "  logits", tuple(logits.shape))
        print(f"T = {x.shape[1]} tokens in PROMPTS[0]; next_logits keeps the last of the {x.shape[1]} rows of logits, {logits.shape[-1]} numbers")
    elif cmd == "check":                                  # Step 5
        print(repr(generate(PROMPTS[0], seed=1, top_k=50)))
        print(repr(generate(PROMPTS[0], seed=1, top_k=50)))
        print(repr(generate(PROMPTS[0], seed=2, top_k=50)))
        print(repr(generate(PROMPTS[0], seed=9, temperature=0)))
    elif cmd == "snapshot":                               # Step 6
        for k, p in enumerate(PROMPTS, 1):
            print(f"## prompt {k}  ends {p[-40:]!r}")
            snapshot(p)
    elif cmd == "api0":                                   # Step 7
        runs = [api(PROMPTS[0], 0) for _ in range(20)]
        (ROOT / "sampling" / "api_t0.json").write_text(
            json.dumps([{"text": t, "first_token_top5": lp} for t, lp in runs], indent=1), encoding="utf-8")
        print(len({t for t, _ in runs}), "distinct texts in 20")
    elif cmd == "table":                                  # Extension X2
        for k, p in enumerate(PROMPTS, 1):
            for t in (0, 0.7, 1.2):
                outs = [generate(p, seed=s, temperature=t) for s in range(5)]
                print(f"\n## prompt {k}  T={t}  distinct {len(set(outs))} of 5  ends {p[-40:]!r}")
                for o in outs:
                    print("  ", repr(o))
    else:
        print("commands: pick, shapes, check, snapshot, api0, table")
```

```bash
uv run python sampling/lab.py pick
```

*You should see* three quoted strings, each the opening twenty GPT-2 tokens of a chunk of your corpus, chosen by a fixed seed, so everyone with your corpus gets the same three. On Homer the seed picks chunks 1363, 1094 and 1820 of 2142, which open `Hector hurried from the house when she had done speaking, ...`, `These, then, went on board and sailed their ways over the sea. ...` and `The Trojans with Hector at their head charged in a body. ...`; the printed strings are those openings cut at twenty tokens, often mid-sentence.

<!-- JD: fill the three exact Homer strings after a run; the chunk indices and openings above are exact (computed offline), the 20-token cut needs the gpt2 tokenizer. -->

Paste the three strings into `PROMPTS`, between the square brackets, with commas between them, so the line reads `PROMPTS = ['...', '...', '...']`. From here on every experiment uses those three, and they cannot drift, because they are constants. Run `pick` again any time; it prints the same three.

*If it broke:* `ModuleNotFoundError: No module named 'search'` means `search/search.py` is not on this branch, which means A06 has not merged. Merge it, then `git merge main` into this branch. `OSError: ... gpt2 ... not found` means Step 2's download did not finish.

**Step 4. The two shapes from the lesson.**

```bash
uv run python sampling/lab.py shapes
```

*You should see* two lines: `hidden state (1, T, 768)   logits (1, T, 50257)` with your `T` near 20, and a line saying `T = ... tokens in PROMPTS[0]`. The first shape is one residual vector per position, 768 wide: the body of the model. The second is one score for every token in GPT-2's vocabulary, at every position: the head. `next_logits` takes the last row of the second, the only row that predicts a token you have not seen yet. Compare both with the two shapes you wrote down from the lesson: same three dimensions, his width and vocabulary in place of 768 and 50,257.

<!-- JD: fill the Homer T after a run; it is 20 unless decode-then-encode of the cut changes the token count. -->

**Step 5. The sampler.**

Paste these two functions directly **above** the `if __name__ == "__main__":` line. `sample()` turns one row of logits into one token id and holds all three knobs; `generate()` calls it once per new token.

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """ turn one row of logits into one token id. Everything here is NumPy. """
    if temperature == 0:
        return int(np.argmax(logits))            # temperature 0 is "take the top", decided before any division
    z = logits / temperature                     # temperature stretches (below 1) or squashes (above 1) the gaps
    order = np.argsort(z)[::-1]                  # token ids, most likely first
    z = z[order]
    p = np.exp(z - z[0])                         # softmax, with the largest logit subtracted so exp cannot overflow
    p /= p.sum()
    keep = len(p)
    if top_k:
        keep = min(keep, top_k)                  # top-k: the k most likely survive
    if top_p is not None:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), top_p)) + 1)   # top-p: the fewest whose probabilities reach p
    p = p[:keep] / p[:keep].sum()                # renormalize what survived
    return int(order[rng.choice(keep, p=p)])

def generate(prompt, n_new=30, seed=0, **knobs):
    """ n_new tokens after the prompt, one sample() per token, with the knobs passed through to sample() """
    rng = np.random.default_rng(seed)
    start = enc.encode(prompt)
    ids = list(start)
    for _ in range(n_new):
        ids.append(sample(next_logits(ids), rng=rng, **knobs))
    return enc.decode(ids[len(start):])
```

Read `sample()` top to bottom once. **Temperature divides every logit before softmax.** Below 1 it stretches the gaps, so the favorite gets more of the probability; above 1 it squashes them, so the probability spreads out. It changes no order: the most likely token at `T=0.7` is the most likely token at `T=1.2`. **Top-k keeps the `k` most likely tokens.** **Top-p keeps the fewest tokens whose probabilities add up to at least `p`.** Both then renormalize what survived, so the survivors share the whole budget. `z - z[0]` subtracts the largest logit, which is what keeps `np.exp` from overflowing; it changes nothing else, because softmax only looks at differences. `topk` and `multinomial` in your video file are `top_k` and `rng.choice` here.

Check it reproduces his loop, then check temperature 0:

```bash
uv run python sampling/lab.py check
```

*You should see* four quoted continuations of your first prompt, thirty tokens each: the first two identical (same seed, same top-k of 50), the third different (seed 2), and the fourth a plain, often repetitive continuation that would come out the same with any seed, because temperature 0 never asks the random number generator anything. The first three are the same experiment as `scratch/a09-video.py`, in NumPy; the texts differ because NumPy's and PyTorch's random draws differ. Each line takes GPT-2 thirty forward passes, so the four take about a minute on a laptop CPU.

<!-- JD: fill the four Homer check lines after a run; shape only here. -->

*If it broke* with `NameError: name 'generate' is not defined`, the two functions landed below the `if __name__` block instead of above it. `ValueError: probabilities do not sum to 1` means the renormalize line is missing.

**Step 6. Snapshot the distribution.**

Paste this directly above the `if __name__ == "__main__":` line. It runs GPT-2 once on a prompt and shows what the sampler is choosing from at three temperatures.

```python
def snapshot(prompt, temps=(0.7, 1.0, 1.2)):
    """ the top-10 next tokens after the prompt at each temperature, and how many tokens hold 90% of the probability """
    z = next_logits(enc.encode(prompt))
    for t in temps:
        p = np.exp((z - z.max()) / t)
        p /= p.sum()
        order = np.argsort(p)[::-1]
        top = ", ".join(f"{enc.decode([int(i)])!r} {p[i]:.3f}" for i in order[:10])
        n90 = int(np.searchsorted(np.cumsum(p[order]), 0.9)) + 1
        print(f"T={t}  top-10: {top}\n       tokens for 90% of the mass: {n90}")
```

Run it into a file:

```bash
uv run python sampling/lab.py snapshot > sampling/snapshots.txt
cat sampling/snapshots.txt
```

*You should see* three blocks, one per prompt, each headed `## prompt k  ends '...'` and holding three `T=` lines with a `tokens for 90% of the mass` line under each. Whatever your corpus, these hold in every block: **the same ten tokens in the same order at all three temperatures**; the first token's probability highest at 0.7 and lowest at 1.2; the tokens-for-90% count rising with temperature, never falling. Temperature reorders nothing. It only changes how much of the budget the leaders get. **The tokens-for-90% count is how concentrated the distribution is**: a prompt that ends mid-name can have one token over 0.9 and a count of 1 or 2; a prompt that ends after a full stop can need hundreds of tokens to reach 90%, because almost any word can start a sentence. Your three prompts will differ from each other for exactly that reason, and Reflection Question 1 asks which is which.

<!-- JD: fill one Homer block after a run (the T=1.0 line and its n90 for each of the three prompts). -->

**Step 7. Temperature 0 on the API, twenty times.**

Paste this directly above the `if __name__ == "__main__":` line. One chat completion that continues a prompt, returning the text and the model's top-5 alternatives for its first token.

```python
def api(prompt, temperature, n_tokens=30):
    """ one chat completion that continues the prompt; returns the text and the top-5 alternatives for its first token """
    body = json.dumps({"model": MODEL, "temperature": temperature, "max_tokens": n_tokens,
                       "logprobs": True, "top_logprobs": 5,
                       "messages": [{"role": "user", "content": "Continue this text: " + prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        c = json.load(r)["choices"][0]
    return c["message"]["content"], c["logprobs"]["content"][0]["top_logprobs"]
```

```bash
uv run python sampling/lab.py api0
```

Twenty calls of 30 tokens on your first prompt; `max_tokens=30` is a spend control. The texts and first-token lists are saved in `sampling/api_t0.json`.

*You should see* one line, `N distinct texts in 20`, with `N` between 1 and 20. Most people predict 1. Whatever you get, Reflection Question 2 has the command that compares the first-token `top_logprobs` across the twenty runs: if the five probabilities themselves differ between two runs, the model computed a different distribution for the same input, and temperature 0 faithfully took the top of a different thing. A count of 1 is a real result too; report it as twenty runs, not as "deterministic".

*If it broke* with `HTTP Error 400`, read the message: a reasoning model refuses `temperature` or `logprobs`, so change `MODEL` and write down which model you used. `KeyError: 'OPENAI_API_KEY'` means a new terminal, as in A05.

**Extension — the same three prompts, three temperatures, five runs (ASSIGNED)**

**X1. Predict first.** Create `sampling/RESULTS.md` and paste this in. Fill the prediction table and the two sentences; leave everything under `## Table` empty for now.

```markdown
# A09 results: <your name>

## Prediction (before the table runs)

| prompt | ends with | predicted distinct of 5 at T=0 | at T=0.7 | at T=1.2 |
|---|---|---|---|---|
| 1 | <last five words of PROMPTS[0]> | <n> | <n> | <n> |
| 2 | <last five words of PROMPTS[1]> | <n> | <n> | <n> |
| 3 | <last five words of PROMPTS[2]> | <n> | <n> | <n> |

Why these numbers: <one sentence, using the tokens-for-90% counts from snapshots.txt>
API at temperature 0, same prompt, twenty runs: I predict <n> distinct texts, because <one sentence>

## Table

| prompt | T | distinct of 5 | usable of 5 | predicted distinct |
|---|---|---|---|---|

## Where it got worse

<one line per prompt: the first temperature at which usable fell below 5, or never>
Worst continuation in the table, pasted: <...>
What went wrong in it: <one or two sentences>

## Temperature 0, local vs API

Local sampler, prompt 1, T=0: <n> distinct of 5 (table.txt)
API, prompt 1, T=0: <n> distinct texts of 20, <n> distinct first-token top-5 lists (api_t0.json)
The mechanism my evidence supports for the difference: <one paragraph>

## What temperature for an agent

<one paragraph; the file and line where my v2 agent sets its temperature, and whether this table argues for changing it>
```

Commit it with the sampler, before the table runs:

```bash
git add sampling/lab.py sampling/RESULTS.md sampling/snapshots.txt sampling/api_t0.json
git commit -m "A09: predicted distinct counts, before the temperature table runs"
```

I will check your commit timestamps.

**X2. Run the table.**

```bash
uv run python sampling/lab.py table > sampling/table.txt
grep -c "^## prompt" sampling/table.txt
```

Forty-five generations of thirty tokens each; on a laptop CPU that is a few minutes, so start it and read `snapshots.txt` while it runs. *You should see* `9`: three prompts times three temperatures, each a `## prompt k  T=t  distinct n of 5` heading with five quoted continuations under it.

**X3. Mark it.** Create `scratch/a09-mark.py` and paste this in. It reads the nine headings out of `table.txt` and your prediction table out of `RESULTS.md`, and prints the X3 table with every column filled except `usable of 5`.

```python
# scratch/a09-mark.py
# Run from the repo root:  uv run python scratch/a09-mark.py
# Reads sampling/table.txt (what "lab.py table" printed) and sampling/RESULTS.md, and prints the nine rows
# of the X3 table with the "distinct of 5" and "predicted distinct" columns filled. You fill "usable of 5"
# by reading sampling/table.txt. Run it again after the usable column is filled in RESULTS.md and it also
# prints, per prompt, the lowest temperature at which usable fell below 5.
from pathlib import Path

lines = Path("sampling/table.txt").read_text(encoding="utf-8").splitlines()
blocks = []                                   # one (prompt number, T, distinct, tail) per "## prompt" heading
for line in lines:
    if line.startswith("## prompt "):
        parts = line.split()                  # ['##', 'prompt', '1', 'T=0', 'distinct', '1', 'of', '5', 'ends', "'...'"]
        blocks.append((int(parts[2]), float(parts[3][2:]), int(parts[5]), line.split("ends ", 1)[1]))
assert len(blocks) == 9, f"{len(blocks)} '## prompt' headings in sampling/table.txt; the table run prints 9 (3 prompts x 3 temperatures)"

results = Path("sampling/RESULTS.md")
text = results.read_text(encoding="utf-8") if results.exists() else ""

def table_after(heading):
    """ the cells of every '| ... |' row under a '## heading', up to the next '## ' """
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.strip().lower().startswith(heading.lower())
            continue
        if inside and line.startswith("|") and not line.startswith("|---"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows

predicted = {}                                # (prompt number, T) -> what you predicted, from the Prediction table
for cells in table_after("## Prediction"):
    if cells[0] in ("1", "2", "3") and len(cells) >= 5:
        for t, cell in zip((0, 0.7, 1.2), cells[-3:]):
            predicted[(int(cells[0]), t)] = cell if cell.isdigit() else "?"

usable = {}                                   # (prompt number, T) -> the usable count you filled in under ## Table
for cells in table_after("## Table"):
    if len(cells) >= 4 and cells[0] in ("1", "2", "3") and cells[3].isdigit():
        usable[(int(cells[0]), float(cells[1]))] = int(cells[3])

print("| prompt | T | distinct of 5 | usable of 5 | predicted distinct |")
print("|---|---|---|---|---|")
for k, t, d, tail in blocks:
    print(f"| {k} | {t:g} | {d} | {usable.get((k, t), '')} | {predicted.get((k, t), '?')} |")
missing = [kt for kt in [(k, t) for k, t, _, _ in blocks] if kt not in predicted]
if missing:
    print(f"\n{len(missing)} predicted cells read '?': the Prediction table in sampling/RESULTS.md has no number there yet")
print("\nprompts, as the table run cut them:")
for k, tail in sorted({(k, tail) for k, _, _, tail in blocks}):
    print(f"  prompt {k} ends {tail}")

if len(usable) == 9:
    print("\nwhere it got worse (first temperature at which usable fell below 5):")
    for k in (1, 2, 3):
        worse = [t for t in (0, 0.7, 1.2) if usable[(k, t)] < 5]
        print(f"  prompt {k}: " + (f"T={worse[0]:g}, usable {usable[(k, worse[0])]} of 5" if worse else "never; usable 5 of 5 at every temperature"))
    print(f"  usable in all: {sum(usable.values())} of 45; distinct in all: {sum(d for _, _, d, _ in blocks)} of 45")
elif usable:
    print(f"\n{len(usable)} of 9 usable cells filled under ## Table; fill the rest and run again for the where-it-got-worse lines")
else:
    print("\nnext: paste the table above into sampling/RESULTS.md under ## Table, fill usable of 5 by reading sampling/table.txt, run this again")
```

```bash
uv run python scratch/a09-mark.py
```

*You should see* a nine-row Markdown table with `distinct of 5` filled from `table.txt`, `usable of 5` empty, and `predicted distinct` copied from your X1 table (a `?` means the prediction cell did not hold a plain number; fix the cell, not the script). Paste the nine rows into `RESULTS.md` under `## Table`, below the header rows already there.

Now the one column that is yours. Open `sampling/table.txt` and read all forty-five continuations. **A continuation is usable when all three of these hold:** every word in it is a real word of your corpus's language, spelled the way your corpus spells it; it continues the prompt's sentence grammatically and then stays in your corpus's register (prose stays prose, verse stays verse, a list stays a list, code stays code); and no phrase of three or more words appears in it twice. One miss on any of the three and it is not usable. Two examples, as shapes:

- Usable: `<the rest of the prompt's sentence, ending in a period> <a new sentence that could sit in the same paragraph of your corpus, about the same people or things>`.
- Not usable: `<the rest of the prompt's sentence> <three words> <the same three words> <the same three words again>` (a loop, the usual temperature-0 failure), or `<the sentence finishes> Copyright <year> <a web address>` (a switch of register toward something GPT-2 read more of than your corpus).

Count the usable ones in each block, write the count in the `usable of 5` cell of that row, and run the script again:

```bash
uv run python scratch/a09-mark.py
```

*You should see* the same table with the usable column filled, then three `where it got worse` lines, one per prompt, naming the first temperature at which usable dropped below 5, and a totals line. Whatever your corpus, `distinct of 5` is 1 at temperature 0 for every prompt (argmax asks nothing of the seed) and rises with temperature. The usable column is where your corpus shows: plain modern prose survives 1.2 better than verse, code or a list, and temperature 0 often fails on usability for the opposite reason, by looping.

<!-- JD: fill the Homer nine-row table and the three where-it-got-worse lines after a run. -->

Fill the rest of `RESULTS.md` from what you have: the three lines under `## Where it got worse` are the script's three lines; the worst continuation is the one you would least want in a book, pasted from `table.txt` with the `## prompt` line it sits under; the `## Temperature 0, local vs API` numbers are the `T=0` rows of your table (1 of 5) and the two counts Reflection Question 2's command prints, with one paragraph committing to the mechanism your evidence supports. End with the agent paragraph: the temperature an agent that calls tools should run at, citing the file and line where your v2 agent sets it (`grep -rn -i temperature src/` in the v2 repo finds it).

**X4. Commit, push, PR, sign off.**

```bash
git add sampling/ scratch/a09-video.py scratch/a09-mark.py pyproject.toml uv.lock
git commit -m "A09: sampler by hand, snapshots, API temp-0 x20, 3x3x5 table marked"
git push -u origin dev/sampling
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the cell of your table you marked least confidently.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`sampling/lab.py` (sampler in NumPy, six commands) + `sampling/RESULTS.md` (predictions, table with the usable column, where it got worse, local vs API, agent paragraph) + `sampling/snapshots.txt` + `sampling/api_t0.json` + `sampling/table.txt` + `scratch/a09-video.py` + `scratch/a09-mark.py`.

**Reflection Questions**

1. From `sampling/snapshots.txt`, paste the top three tokens and the tokens-for-90% count for your most concentrated prompt and your least concentrated one, both at `T=1.0`. Quote the last few words of each prompt and say what about where your corpus chunk was cut explains the difference.

   *How to get it:*

   ```bash
   grep -n "T=1.0" -A1 sampling/snapshots.txt
   ```

   prints the three `T=1.0` lines with their `tokens for 90% of the mass` lines; the smallest count is your most concentrated prompt, the largest your least. The prompts themselves are the `PROMPTS` constant in `lab.py`, in the same order as the `## prompt` blocks, and each block's heading ends with the prompt's last forty characters. A prompt cut mid-word or mid-name is concentrated because only a few tokens can finish it; a prompt cut after a period is spread because almost anything can start a sentence.

2. From `sampling/api_t0.json`: how many distinct texts in twenty, and did the first-token `top_logprobs` differ between any two runs? Paste two lists if they did. Your local sampler gave 1 of 5 at temperature 0 on the same prompt. Name the one mechanism your evidence supports for the gap, and the observation you would need to rule it out.

   *How to get it:* the distinct count is what `api0` printed. For the lists:

   ```bash
   uv run python -c "
   import json
   runs = json.load(open('sampling/api_t0.json'))
   lists = [json.dumps([[a['token'], round(a['logprob'], 3)] for a in r['first_token_top5']]) for r in runs]
   print(len(set(r['text'] for r in runs)), 'distinct texts;', len(set(lists)), 'distinct first-token top-5 lists')
   for l in sorted(set(lists)): print(' ', l)
   "
   ```

   *You should see* two counts and then one line per distinct list. One distinct list and several distinct texts means the first distribution was identical and the divergence came later in the text; several distinct lists means the model computed a different distribution for the same input, which is the batching-and-floating-point mechanism: on shared hardware your request is computed alongside other people's, and the order of floating-point additions changes with the batch, so the logits move in the last decimal places and a near-tie at the top can flip. One list and one text is the result that says the serving stack was, for those twenty runs, deterministic. The local 1-of-5 is the `T=0` row for prompt 1 in `sampling/table.txt`.

3. Your X1 prediction row for `T=1.2` (with the commit hash) next to what happened. Paste the worst continuation in your table and say what went wrong in it against the three-part definition of usable. Then give your v2 agent's sampling temperature with the file and line, and say whether this table argues for changing it.

   *How to get it:* the prediction is the `at T=1.2` column of the X1 table in `RESULTS.md`; what happened is the `distinct of 5` cell of each `T=1.2` row, which `scratch/a09-mark.py` printed side by side with it. The hash is the first commit that touched the file:

   ```bash
   git log --oneline --follow -- sampling/RESULTS.md | tail -1
   ```

   The worst continuation is the one you pasted under `## Where it got worse`; name which of the three conditions it failed (a made-up word, a register switch, a repeated phrase). For the agent's temperature, search the v2 repo (`grep -rn -i temperature src/`); in the reference agent it is `TEMPERATURE = 0.2` in `src/llm.ts`. The argument runs through your usable column: an agent's output is parsed by a program, so the question is at which temperature your table stopped giving five usable out of five.
