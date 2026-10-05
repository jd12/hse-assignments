# A09 · Transformer LLMs 10 + Sampling by Hand

**Meetings:** D19 · **Points:** 15 pts

**Watch — 21 min**

**One meeting — 21 min**
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 10, Model Example (9m, code)
[Let's reproduce GPT-2 (124M), Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU) · segment 00:33:31 → 00:45:50 (12m): sampling init, prefix tokens, tokenization, the sampling loop, auto-detecting the device
 · Start the GPT-2 download in Step 2 before you press play. It is about half a gigabyte and it finishes while you watch.

Optional, not in the total: lesson 11, Recent Improvements (10m), and lesson 12, Mixture of Experts (9m). Lesson 12 is where the best explanation for Step 7 comes from.

**During the video**

Make one file now, `scratch/a09-video.py`, and leave it open.

**DLAI lesson 10 · write down the shapes.** Run the notebook in the DLAI page as he goes; the model he loads is too big to be pleasant on a laptop. Every time he prints a shape, write it in your log with what it is. The two that matter are what comes out of the body of the model and what comes out of the head on top of it. Step 3 gets the same two out of GPT-2, and they should look alike with different numbers.

**Karpathy 00:33:31 → 00:45:50 · type everything he types, and run it.** He is using the GPT class he wrote earlier in the video, loaded with GPT-2's weights. You did not write that class, so load the same weights from Hugging Face instead. Put this at the top of `scratch/a09-video.py`:

```python
import tiktoken, torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
```

Then type his code as he types it: `num_return_sequences`, `max_length`, the prompt, `tiktoken.get_encoding("gpt2")`, the `unsqueeze(0).repeat(...)`, the seed, and the `while` loop with `torch.topk(probs, 50, dim=-1)`, `torch.multinomial` and `torch.gather`. One line differs: his model returns the logits, and the Hugging Face one returns an object, so where he writes `logits = model(x)` you write `logits = model(x).logits`. When he adds device detection, type it too; on a Mac it will find `mps`. Run it and read your five continuations.

He samples with top-k and nothing else. Temperature and top-p do not appear in his loop. You add them by hand in Step 5.

**Notes**

**Two tokenizers, two vocabularies.** GPT-2 reads ids from `tiktoken.get_encoding("gpt2")`. Hand it ids from `cl100k_base` (A08's) and you get `IndexError: index out of range in self`, or, for ids that happen to be small enough, fluent garbage.

**Temperature 0 is not a division.** Logits divided by zero are infinities. The sampler treats `temperature == 0` as "take the highest-probability token" before it divides anything.

**The graded sampler is NumPy.** PyTorch appears in one function, `next_logits`, which runs GPT-2 and hands back a NumPy array. Everything that turns logits into a choice is yours, in NumPy.

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
- [ ] GPT-2 downloaded
- [ ] DLAI lesson 10, shapes written down
- [ ] Karpathy 00:33:31 → 00:45:50 typed along in scratch/a09-video.py
- [ ] Steps 3–4, three prompts from my corpus, two shapes printed
- [ ] Steps 5–6, sampler written, distribution snapshots saved
- [ ] Step 7, twenty calls at temperature 0 on the API
- [ ] Extension: predictions committed, 3 prompts × 3 temperatures × 5 runs
- [ ] RESULTS.md, push and open the PR
```

**Step 2. Install and download.**

```bash
uv add --dev transformers
mkdir -p sampling
uv run python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"
```

*You should see* a download progress bar and then nothing. The weights are cached in your home folder; the next load takes seconds.

**Step 3. Load GPT-2 and pick three prompts from your corpus.**

Create `sampling/lab.py`:

```python
# sampling/lab.py
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
import tiktoken, torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from search.search import chunk

enc = tiktoken.get_encoding("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
MODEL = "gpt-4o-mini"  # the model your v2 agent calls, if it takes temperature and logprobs
PROMPTS = []           # filled in once, in Step 3, and never changed after

def pick_prompts(n=3, seed=0, n_tokens=20):
    chunks = chunk((ROOT / "data" / "corpus.txt").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    return [enc.decode(enc.encode(chunks[i])[:n_tokens]) for i in rng.choice(len(chunks), n, replace=False)]

def next_logits(ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1].numpy().astype(np.float64)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd == "pick":
        for p in pick_prompts():
            print(repr(p))
```

```bash
uv run python sampling/lab.py pick
```

*You should see* three quoted strings, each the opening twenty tokens of a random chunk of your corpus. Paste them into `PROMPTS` as a list. From here on every experiment uses those three, and they cannot drift, because they are constants.

*If it broke:* `ModuleNotFoundError: No module named 'search'` means `search/search.py` is not on this branch, which means A06 has not merged. Merge it, then `git merge main` into this branch.

**Step 4. The two shapes from the lesson.**

```bash
uv run python -c "
import sys, torch; sys.path.insert(0, '.')
from sampling.lab import model, enc, PROMPTS
x = torch.tensor([enc.encode(PROMPTS[0])])
print(model.transformer(x).last_hidden_state.shape, model(x).logits.shape)
"
```

*You should see* `torch.Size([1, T, 768])` and `torch.Size([1, T, 50257])`, with `T` near 20. The first is one residual vector per position, 768 wide. The second is one score for every token in GPT-2's vocabulary, at every position. `next_logits` takes the last row of the second.

**Step 5. Write the sampler.**

Add to `lab.py`, above `__main__`:

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    if temperature == 0:
        return int(np.argmax(logits))
    z = logits / temperature
    order = np.argsort(z)[::-1]
    z = z[order]
    p = np.exp(z - z[0])
    p /= p.sum()
    keep = len(p)
    if top_k:
        keep = min(keep, top_k)
    if top_p is not None:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), top_p)) + 1)
    p = p[:keep] / p[:keep].sum()
    return int(order[rng.choice(keep, p=p)])

def generate(prompt, n_new=30, seed=0, **knobs):
    rng = np.random.default_rng(seed)
    start = enc.encode(prompt)
    ids = list(start)
    for _ in range(n_new):
        ids.append(sample(next_logits(ids), rng=rng, **knobs))
    return enc.decode(ids[len(start):])
```

Temperature divides every logit before softmax. Below 1 it stretches the gaps, so the favorite gets more of the probability; above 1 it squashes them. Top-k keeps the `k` most likely tokens. Top-p keeps the fewest tokens whose probabilities add up to at least `p`. Both then renormalize what survived. `z - z[0]` subtracts the largest logit, which is what keeps `np.exp` from overflowing.

Check it reproduces Karpathy's loop, then check temperature 0:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from sampling.lab import generate, PROMPTS
print(repr(generate(PROMPTS[0], seed=1, top_k=50)))
print(repr(generate(PROMPTS[0], seed=1, top_k=50)))
print(repr(generate(PROMPTS[0], seed=2, top_k=50)))
print(repr(generate(PROMPTS[0], seed=9, temperature=0)))
"
```

*You should see* the first two lines identical (same seed), the third different, and the fourth a plain, often repetitive continuation that would come out the same with any seed.

**Step 6. Snapshot the distribution.**

Add:

```python
def snapshot(prompt, temps=(0.7, 1.0, 1.2)):
    z = next_logits(enc.encode(prompt))
    for t in temps:
        p = np.exp((z - z.max()) / t)
        p /= p.sum()
        order = np.argsort(p)[::-1]
        top = ", ".join(f"{enc.decode([int(i)])!r} {p[i]:.3f}" for i in order[:10])
        n90 = int(np.searchsorted(np.cumsum(p[order]), 0.9)) + 1
        print(f"T={t}  top-10: {top}\n       tokens for 90% of the mass: {n90}")
```

and a `snapshot` branch to `__main__` that calls it on each of `PROMPTS`. Run it into a file:

```bash
uv run python sampling/lab.py snapshot > sampling/snapshots.txt
```

*You should see*, for each prompt, the same ten tokens in the same order at all three temperatures, with the top probability highest at 0.7 and lowest at 1.2, and the tokens-for-90% count rising with temperature, never falling. Temperature reorders nothing. It only changes how much of the budget the leaders get. Your three prompts will differ a lot from each other: a prompt that ends mid-name can have one token over 0.9, and a prompt that ends after a full stop can need hundreds of tokens to reach 90%.

**Step 7. Temperature 0 on the API, twenty times.**

Add:

```python
def api(prompt, temperature, n_tokens=30):
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

and this branch to `__main__`:

```python
    elif cmd == "api0":
        runs = [api(PROMPTS[0], 0) for _ in range(20)]
        (ROOT / "sampling" / "api_t0.json").write_text(
            json.dumps([{"text": t, "first_token_top5": lp} for t, lp in runs], indent=1), encoding="utf-8")
        print(len({t for t, _ in runs}), "distinct texts in 20")
```

```bash
uv run python sampling/lab.py api0
```

Twenty calls of 30 tokens; `max_tokens=30` is a spend control.

*You should see* a count between 1 and 20. Most people predict 1. Whatever you get, count it over all twenty runs, and compare the first-token `top_logprobs` across runs: if the probabilities themselves differ between two runs, the model computed a different distribution, and temperature 0 faithfully took the top of a different thing. A count of 1 is a real result too; report it as twenty runs, not as "deterministic".

**Extension — the same prompt, three temperatures, five runs (ASSIGNED)**

**X1. Predict first.** Start `sampling/RESULTS.md` with a table of guesses: for each of your three prompts, how many distinct continuations out of five you expect at temperature 0, 0.7 and 1.2. Commit it alone:

```bash
git add sampling/lab.py sampling/RESULTS.md
git commit -m "A09: predicted distinct counts, before the temperature table runs"
```

I will check your commit timestamps.

**X2. Run the table.** Add a `table` branch to `__main__`:

```python
    elif cmd == "table":
        for p in PROMPTS:
            for t in (0, 0.7, 1.2):
                outs = [generate(p, seed=s, temperature=t) for s in range(5)]
                print(f"\n## {p[-40:]!r}  T={t}  distinct {len(set(outs))} of 5")
                for o in outs:
                    print("  ", repr(o))
```

```bash
uv run python sampling/lab.py table > sampling/table.txt
```

Forty-five generations of thirty tokens each; on a laptop CPU that is a few minutes, so start it and read the snapshots while it runs.

**X3. Mark it.** Read every continuation and mark it usable if it reads as a plausible next thirty tokens of *your corpus*, in its voice. Build:

| prompt | T | distinct of 5 | usable of 5 | predicted distinct |
|---|---|---|---|---|

*You should see* 1 of 5 at temperature 0 for every prompt, rising with temperature. The usable column is where your corpus shows: plain modern prose survives 1.2 better than verse, code or a list.

Then say where it got worse: the lowest temperature at which usable fell below 5 for each prompt, and the worst continuation you got, pasted. Put your local temperature-0 count (1 of 5) next to your API temperature-0 count from Step 7 (out of 20) and commit, in one paragraph, to the mechanism your evidence supports for the difference.

End `RESULTS.md` with one paragraph on what temperature an agent that calls tools should run at, citing the file and line where your v2 agent sets it.

**X4. Commit, push, PR, sign off.**

```bash
git add sampling/ scratch/a09-video.py pyproject.toml uv.lock
git commit -m "A09: sampler by hand, snapshots, API temp-0 x20, 3x3x5 table"
git push -u origin dev/sampling
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the cell of your table you marked least confidently.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`sampling/lab.py` (sampler in NumPy) + `sampling/RESULTS.md` (predictions, table, API temperature-0 result, agent paragraph) + `sampling/snapshots.txt` + `sampling/api_t0.json` + `sampling/table.txt` + `scratch/a09-video.py`.

**Reflection Questions**

1. From `sampling/snapshots.txt`, paste the top three tokens and the tokens-for-90% count for your most concentrated prompt and your least concentrated one, both at `T=1.0`. Quote the last few words of each prompt and say what about where your corpus chunk was cut explains the difference.

   *How to get it:* `grep -n "T=1.0" -A1 sampling/snapshots.txt` prints the three `T=1.0` lines with their tokens-for-90% line; the smallest count is your most concentrated prompt, the largest your least. The prompts themselves are the `PROMPTS` constant in `lab.py`, in the same order as the snapshot blocks.

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

   One distinct list and several distinct texts means the first distribution was identical and the divergence came later; several distinct lists means the model computed a different distribution for the same input, which is the batching-and-floating-point mechanism. The local 1-of-5 is the first row of `table.txt`.

3. Your X1 prediction row for `T=1.2` (with the commit hash) next to what happened. Paste the worst continuation in your table and say what went wrong in it, token by token if you can. Then give your v2 agent's sampling temperature with the file and line, and say whether this table argues for changing it.

   *How to get it:* the prediction row is the `T=1.2` column of the table you wrote in `RESULTS.md` before running; the hash:

   ```bash
   git log --oneline --follow -- sampling/RESULTS.md | tail -1
   ```

   What happened is the `distinct` number on each `T=1.2` heading in `sampling/table.txt`. For the agent's temperature, search the v2 repo (`grep -rn -i temperature src/`); in the reference agent it is `TEMPERATURE = 0.2` in `src/llm.ts`.
