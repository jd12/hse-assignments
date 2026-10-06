# A09 · Transformer LLMs 10 + Sampling by Hand

**Meetings:** D19 · **Points:** 15 pts

**Watch — 21 min**

**One meeting — 21 min**
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 10, Model Example (9m, code)
[Let's reproduce GPT-2 (124M), Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU) · segment 00:33:31 → 00:45:50 (12m): sampling init, prefix tokens, tokenization, the sampling loop, auto-detecting the device
 · Do Steps 2 and 3 before you press play: the GPT-2 download is about half a gigabyte and finishes while you watch, and `scratch/a09-video.py` needs the prompts from Step 3.

Optional, not in the total: lesson 11, Recent Improvements (10m), and lesson 12, Mixture of Experts (9m). Lesson 12 is where the best explanation for Step 7 comes from.

**During the video**

He samples from a model he wrote, with a prompt he made up. `scratch/a09-video.py` runs the same loop on the Hugging Face GPT-2 with the first prompt from your corpus. Run it once before you press play, after Step 3:

```bash
uv run python scratch/a09-video.py
```

The script encodes your prompt, stacks it five times, and runs his `while` loop: softmax, `topk` of 50, `multinomial`, append, until each row is 50 tokens. *You should see* the device line, your prompt in quotes, `x (5, T)` with your `T` near 20, `x after the loop (5, 50): 30 new tokens per row`, and five continuations starting with `>`, all five different. Paste the five under `## During the video` in `evidence/A09.md`. *If it broke* with `ModuleNotFoundError: No module named 'sampling'`, Step 3 is not done. `FileNotFoundError: data/corpus.txt` means the terminal is not in the repo root; `pwd` should end in `foundations-<your-username>`. <!-- JD: fill the Homer T and one of the five continuations after a run. -->

**DLAI lesson 10 · write down the shapes.** No typing. Run the notebook in the DLAI page as he goes, and every time he prints a shape, write it in your log with what it is. The two that matter are what comes out of the body of the model and what comes out of the head on top of it; Step 4 prints the same two out of GPT-2. **Karpathy 00:33:31 → 00:45:50 · read along; the file has already run.** Five pause points, one line in your log each:

1. `num_return_sequences = 5`, `max_length = 30`: your file has `max_length = 50`. Write down why (his prompt is 8 tokens, yours is 20, both want about thirty new ones).
2. `tiktoken.get_encoding("gpt2")` and `unsqueeze(0).repeat(...)`: your `x (5, T)` line is the same shape with your prompt's length in place of his 8. Write down both shapes.
3. `logits = model(x)`: the one line that differs. Hugging Face returns an object, so your line reads `model(x).logits`. Write down what `logits[:, -1, :]` keeps and why only that row matters.
4. `torch.topk(probs, 50, dim=-1)` and `torch.multinomial`: that is top-k, the only knob in his loop. **Temperature and top-p do not appear.** Step 5 adds both by hand.
5. His five continuations are about language models, because that is what his prompt said. Read yours again and write one line on whether GPT-2 stayed in your corpus's voice or drifted toward something it read more of.

**Notes**

**Two tokenizers, two vocabularies.** GPT-2 reads ids from `tiktoken.get_encoding("gpt2")`, 50,257 of them. Hand it ids from `cl100k_base` (A08's, 100,277) and you get `IndexError: index out of range in self`, or, for ids that happen to be small enough, fluent garbage.

**Temperature 0 is not a division.** Logits divided by zero are infinities. The sampler treats `temperature == 0` as "take the highest-probability token" before it divides anything.

**The API is not your laptop.** Temperature 0 on GPT-2 on your machine is one computation, done the same way every time. Temperature 0 on the API runs on shared hardware alongside other people's requests. Step 7 is about the difference.

**Walkthrough — temperature, top-k and top-p on your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/sampling
```

Step 2 of A05b's template update, or the pull request I opened on your repo, put `a09-video.py` and `a09-mark.py` in `scratch/`, and `evidence/A09.md`; `ls scratch evidence` should show them.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Steps 2–4, GPT-2 downloaded, lab.py started, three prompts from my corpus in PROMPTS, two shapes printed
- [ ] a09-video.py run, five continuations in evidence/A09.md; DLAI shapes and five Karpathy pause-point lines in the log
- [ ] Steps 5–6, sampler pasted and checked, snapshots saved
- [ ] Step 7, twenty calls at temperature 0 on the API, output in evidence/A09.md
- [ ] Extension: predictions committed, table run, usable column filled
- [ ] Fill every slot in evidence/A09.md (check_evidence.py: all slots filled); push and open the PR
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

Create `sampling/lab.py` (right-click `sampling`, **New File**) and paste this in. It is the whole file except the four functions Steps 5 and 6 paste above the main block: the model, the tokenizer, the two functions that touch GPT-2, and the main block with one branch per command. The branches that call functions you have not pasted yet will not run until you paste them; `pick` needs none of them.

```python
# sampling/lab.py
# Run from the repo root:  uv run python sampling/lab.py <command>, one command per step: pick, shapes, check, snapshot, api0, table
# The lines you edit: PROMPTS (once, in Step 3) and MODEL (only if the API refuses the knobs).
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
import tiktoken, torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent.parent   # this file's folder is sampling/; its parent is the repo root
sys.path.insert(0, str(ROOT))                    # so "from search.search" works from any folder
from search.search import chunk                  # A06's chunker, so the prompts are cut the way the corpus was chunked

enc = tiktoken.get_encoding("gpt2")              # GPT-2's tokenizer. Not cl100k_base: different ids, different vocabulary.
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()   # the 124M-parameter GPT-2, loaded from the Step 2 download
MODEL = "gpt-4o-mini"  # the model your v2 agent calls, if it takes temperature and logprobs
PROMPTS = []           # filled in once, in Step 3, from what "pick" printed, and never changed after

# In: how many prompts, the seed, how many tokens each.  Out: a list of n strings cut from your corpus.
def pick_prompts(n=3, seed=0, n_tokens=20):
    """ the first n_tokens GPT-2 tokens of n chunks of your corpus, chosen with a fixed seed """
    chunks = chunk((ROOT / "data" / "corpus.txt").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)                         # a random number generator that always draws the same way for this seed
    picked = rng.choice(len(chunks), n, replace=False)        # n different chunk numbers, drawn at random
    prompts = []
    for i in picked:
        ids = enc.encode(chunks[i])[:n_tokens]                # the chunk as token ids, cut to the first n_tokens
        prompts.append(enc.decode(ids))                       # and back to text
    return prompts

# In: a list of token ids.  Out: GPT-2's scores for what comes next, one per token in its vocabulary.
def next_logits(ids):
    """ run GPT-2 on a list of token ids and hand back the logits for the next token, as a NumPy array of 50,257 numbers """
    with torch.no_grad():                                     # no gradients: we only read the model, we never train it
        logits = model(torch.tensor([ids])).logits            # shape (1, T, 50257): a row of scores at every position
        return logits[0, -1].numpy().astype(np.float64)       # the last row only, as NumPy: the earlier rows predict tokens we already have

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"       # the word after the script name, or "pick" if there is none
    if cmd != "pick":                                         # every other command needs the three prompts
        assert len(PROMPTS) == 3, "paste the three strings that 'pick' printed into PROMPTS first (Step 3)"
    if cmd == "pick":                                         # Step 3: print the three prompts to paste into PROMPTS
        for p in pick_prompts():
            print(repr(p))
    elif cmd == "shapes":                                     # Step 4: the two shapes from the lesson, out of GPT-2
        x = torch.tensor([enc.encode(PROMPTS[0])])
        with torch.no_grad():
            hidden = model.transformer(x).last_hidden_state   # what comes out of the body of the model
            logits = model(x).logits                          # what comes out of the head on top of it
        print("hidden state", tuple(hidden.shape), "  logits", tuple(logits.shape))
        print(f"T = {x.shape[1]} tokens in PROMPTS[0]; next_logits keeps the last of the {x.shape[1]} rows of logits, {logits.shape[-1]} numbers")
    elif cmd == "check":                                      # Step 5: same seed twice, a new seed, then temperature 0
        print(repr(generate(PROMPTS[0], seed=1, top_k=50)))
        print(repr(generate(PROMPTS[0], seed=1, top_k=50)))
        print(repr(generate(PROMPTS[0], seed=2, top_k=50)))
        print(repr(generate(PROMPTS[0], seed=9, temperature=0)))
    elif cmd == "snapshot":                                   # Step 6: the top-10 next tokens for each prompt at three temperatures
        for k, p in enumerate(PROMPTS, 1):                    # k counts 1, 2, 3 and p is the prompt
            print(f"## prompt {k}  ends {p[-40:]!r}")         # p[-40:] is the last 40 characters; !r shows them in quotes
            snapshot(p)
    elif cmd == "api0":                                       # Step 7: the API twenty times at temperature 0, saved to a file
        runs = []
        for _ in range(20):
            runs.append(api(PROMPTS[0], 0))
        (ROOT / "sampling" / "api_t0.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
        texts, lists = set(), set()                           # a set keeps one copy of each thing, so its size is the distinct count
        for r in runs:
            texts.add(r["text"])
            top5 = []
            for a in r["first_token_top5"]:                   # the five alternatives: the token and its log probability, rounded
                top5.append([a["token"], round(a["logprob"], 3)])
            lists.add(json.dumps(top5))                       # as one string, so the set can compare whole lists
        print(len(texts), "distinct texts in 20;", len(lists), "distinct first-token top-5 lists")
        for l in sorted(lists):
            print("  ", l)
    elif cmd == "table":                                      # Extension X2: 3 prompts x 3 temperatures x 5 seeds
        for k, p in enumerate(PROMPTS, 1):
            for t in (0, 0.7, 1.2):
                outs = []
                for s in range(5):                            # seeds 0 to 4
                    outs.append(generate(p, seed=s, temperature=t))
                print(f"\n## prompt {k}  T={t}  distinct {len(set(outs))} of 5  ends {p[-40:]!r}")   # set(outs) drops duplicates
                for o in outs:
                    print("  ", repr(o))
```

```bash
uv run python sampling/lab.py pick
```

*You should see* three quoted strings, each the opening twenty GPT-2 tokens of a chunk of your corpus, chosen by a fixed seed, so everyone with your corpus gets the same three. On Homer the seed picks chunks 1363, 1094 and 1820 of 2142, which open `Hector hurried from the house when she had done speaking, ...`, `These, then, went on board and sailed their ways over the sea. ...` and `The Trojans with Hector at their head charged in a body. ...`, cut at twenty tokens, often mid-sentence. <!-- JD: fill the three exact Homer strings after a run; the chunk indices and openings are exact (computed offline), the 20-token cut needs the gpt2 tokenizer. -->

Paste the three strings into `PROMPTS`, between the square brackets, with commas between them, so the line reads `PROMPTS = ['...', '...', '...']`. **Every experiment from here on uses those three**, and they cannot drift, because they are constants.

*If it broke:* `ModuleNotFoundError: No module named 'search'` means `search/search.py` is not on this branch, which means A06 has not merged. Merge it, then `git merge main` into this branch. `OSError: ... gpt2 ... not found` means Step 2's download did not finish.

**Step 4. The two shapes from the lesson.**

```bash
uv run python sampling/lab.py shapes
```

*You should see* `hidden state (1, T, 768)   logits (1, T, 50257)` with your `T` near 20, then `T = ... tokens in PROMPTS[0]`. **The first shape is one residual vector per position, 768 wide: the body. The second is one score for every token in GPT-2's vocabulary, at every position: the head.** `next_logits` takes the last row of the second, the only row that predicts a token you have not seen yet. Compare both with the two shapes from the lesson: same three dimensions, his width and vocabulary in place of 768 and 50,257. Paste both lines under `## Step 4` in `evidence/A09.md`. <!-- JD: fill the Homer T after a run; it is 20 unless decode-then-encode of the cut changes the token count. -->

**Step 5. The sampler.**

Paste these two functions directly **above** the `if __name__ == "__main__":` line. `sample()` turns one row of logits into one token id and holds all three knobs; `generate()` calls it once per new token. Read `sample()` top to bottom once, with its comments.

```python
# In: one row of 50,257 logits, the three knobs, and a random number generator.  Out: one token id.
def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """ turn one row of logits into one token id. Everything here is NumPy. """
    if temperature == 0:
        return int(np.argmax(logits))            # temperature 0 is "take the top": argmax is the position of the biggest score, decided before any division
    z = logits / temperature                     # temperature divides every score: below 1 stretches the gaps, above 1 squashes them
    order = np.argsort(z)[::-1]                  # argsort gives the positions sorted smallest first; [::-1] reverses, so the most likely token id comes first
    z = z[order]                                 # the scores in that same order, biggest first
    p = np.exp(z - z[0])                         # softmax, step 1: e to the power of each score, with the biggest subtracted first so exp cannot overflow
    p /= p.sum()                                 # softmax, step 2: divide by the total, so the probabilities add up to one
    keep = len(p)                                # how many tokens survive; start with all of them
    if top_k:
        keep = min(keep, top_k)                  # top-k: only the k most likely survive
    if top_p is not None:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), top_p)) + 1)   # top-p: cumsum adds up the probabilities as it goes; searchsorted finds where the running total first reaches top_p; +1 because positions count from 0
    p = p[:keep] / p[:keep].sum()                # renormalize what survived, so the survivors share the whole budget
    return int(order[rng.choice(keep, p=p)])     # rng.choice draws one position from 0 to keep-1, weighted by p; order[...] turns it back into a token id

# In: a prompt, how many tokens to add, a seed, the three knobs.  Out: the new tokens as text.
def generate(prompt, n_new=30, seed=0, temperature=1.0, top_k=None, top_p=None):
    """ n_new tokens after the prompt, one sample() per token, with the knobs passed through to sample() """
    rng = np.random.default_rng(seed)            # same seed, same draws, same text
    start = enc.encode(prompt)                   # the prompt as token ids
    ids = list(start)                            # a copy that grows by one id per loop
    for _ in range(n_new):
        ids.append(sample(next_logits(ids), temperature=temperature, top_k=top_k, top_p=top_p, rng=rng))
    return enc.decode(ids[len(start):])          # ids[len(start):] skips the prompt, so only the new tokens come back
```

**Temperature divides every logit before softmax**, so it changes no order: the most likely token at `T=0.7` is the most likely token at `T=1.2`. **Top-k keeps the `k` most likely tokens; top-p keeps the fewest tokens whose probabilities add up to at least `p`.** Both then renormalize what survived. `topk` and `multinomial` in the video file are `top_k` and `rng.choice` here.

```bash
uv run python sampling/lab.py check
```

*You should see* four quoted continuations of your first prompt, thirty tokens each: the first two identical (same seed, same top-k of 50), the third different (seed 2), and the fourth a plain, often repetitive continuation that would come out the same with any seed, because temperature 0 never asks the random number generator anything. The four take about a minute on a laptop CPU. Paste them under `## Step 5` in `evidence/A09.md`. *If it broke* with `NameError: name 'generate' is not defined`, the two functions landed below the `if __name__` block instead of above it. <!-- JD: fill the four Homer check lines after a run; shape only here. -->

**Step 6. Snapshot the distribution.**

Paste these two functions directly above the `if __name__ == "__main__":` line: `snapshot()` for this step, `api()` for Step 7. `snapshot()` runs GPT-2 once on a prompt and shows what the sampler is choosing from at three temperatures; `api()` is one chat completion that continues a prompt. Read `api()` once: it is what you are sending, and `MODEL` is the model you are paying for.

```python
# In: a prompt and the temperatures to try.  Out: nothing returned; prints the top-10 next tokens at each temperature.
def snapshot(prompt, temps=(0.7, 1.0, 1.2)):
    """ the top-10 next tokens after the prompt at each temperature, and how many tokens hold 90% of the probability """
    z = next_logits(enc.encode(prompt))          # GPT-2 runs once; every temperature below reuses the same scores
    for t in temps:
        p = np.exp((z - z.max()) / t)            # softmax at temperature t, in one line: subtract the biggest, divide by t, exp
        p /= p.sum()
        order = np.argsort(p)[::-1]              # token ids, most likely first (see sample())
        parts = []
        for i in order[:10]:                     # the ten most likely: the token as text and its probability to 3 places
            parts.append(f"{enc.decode([int(i)])!r} {p[i]:.3f}")
        top = ", ".join(parts)                   # one string, the parts separated by commas
        n90 = int(np.searchsorted(np.cumsum(p[order]), 0.9)) + 1   # how many tokens, most likely first, it takes to reach 90% (same trick as top-p)
        print(f"T={t}  top-10: {top}\n       tokens for 90% of the mass: {n90}")

# In: a prompt and a temperature.  Out: a dict with the API's text and its top-5 alternatives for the first token.
def api(prompt, temperature, n_tokens=30):
    """ one chat completion that continues the prompt; returns the text and the top-5 alternatives for its first token """
    body = json.dumps({"model": MODEL, "temperature": temperature, "max_tokens": n_tokens,   # max_tokens=30 is a spend control
                       "logprobs": True, "top_logprobs": 5,                                 # ask for the five most likely tokens at every position
                       "messages": [{"role": "user", "content": "Continue this text: " + prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],   # the key from your terminal's environment, as in A05
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:       # send it and wait for the reply
        c = json.load(r)["choices"][0]           # the reply is JSON; "choices" holds one answer
    first_token_top5 = c["logprobs"]["content"][0]["top_logprobs"]   # position 0 of the answer: its five most likely tokens and their log probabilities
    return {"text": c["message"]["content"], "first_token_top5": first_token_top5}
```

```bash
uv run python sampling/lab.py snapshot | tee sampling/snapshots.txt
```

*You should see* three blocks, one per prompt, each headed `## prompt k  ends '...'` and holding three `T=` lines with a `tokens for 90% of the mass` line under each. Whatever your corpus, in every block: **the same ten tokens in the same order at all three temperatures**; the first token's probability highest at 0.7 and lowest at 1.2; the tokens-for-90% count rising with temperature, never falling. **The tokens-for-90% count is how concentrated the distribution is**: a prompt that ends mid-name can have a count of 1 or 2; a prompt that ends after a full stop can need hundreds, because almost any word can start a sentence. Reflection Question 1 asks which of your three is which. <!-- JD: fill one Homer block after a run (the T=1.0 line and its n90 for each of the three prompts). -->

**Step 7. Temperature 0 on the API, twenty times.**

```bash
uv run python sampling/lab.py api0
```

Twenty calls of 30 tokens on your first prompt; every run is saved in `sampling/api_t0.json`. *You should see* `N distinct texts in 20; M distinct first-token top-5 lists`, then one line per distinct list: the five most likely first tokens with their log probabilities. Paste all of it under `## Step 7` in `evidence/A09.md`. Most people predict `N` of 1. **If `M` is above 1, the model computed a different distribution for the same input**, and temperature 0 faithfully took the top of a different thing; if `M` is 1 and `N` is not, the first distribution was identical and the divergence came later in the text. A count of 1 is a real result too; report it as twenty runs, not as "deterministic". *If it broke* with `HTTP Error 400`, a reasoning model refuses `temperature` or `logprobs`: change `MODEL` and write down which model you used. `KeyError: 'OPENAI_API_KEY'` means a new terminal, as in A05.

**Extension — the same three prompts, three temperatures, five runs (ASSIGNED)**

**X1. Predict first.** The prediction table is already in `evidence/A09.md` under `## Prediction`. Fill its `<   >` cells with whole numbers from 1 to 5, the three `ends with` cells, and the two sentences under it; leave every section below it for later.

Commit it before the table runs:

```bash
git add sampling/lab.py evidence/A09.md sampling/snapshots.txt sampling/api_t0.json
git commit -m "A09: predicted distinct counts, before the temperature table runs"
```

I will check your commit timestamps.

**X2. Run the table.**

```bash
uv run python sampling/lab.py table > sampling/table.txt
grep -c "^## prompt" sampling/table.txt
```

Forty-five generations of thirty tokens; on a laptop CPU that is a few minutes, so read `snapshots.txt` while it runs. *You should see* `9`: three prompts times three temperatures, each a `## prompt k  T=t  distinct n of 5` heading with five quoted continuations under it.

**X3. Mark it.**

```bash
uv run python scratch/a09-mark.py
```

The script reads the nine headings out of `table.txt` and your prediction table out of `evidence/A09.md`. *You should see* a nine-row Markdown table with `distinct of 5` filled, `usable of 5` empty, and `predicted distinct` copied from X1 (a `?` means that prediction cell did not hold a plain number; fix the cell, not the script). Paste the whole table under `## Extension X3` in `evidence/A09.md`, in place of the first slot there.

Now the one column that is yours. Open `sampling/table.txt` and read all forty-five continuations. **A continuation is usable when all three of these hold:** every word in it is a real word of your corpus's language, spelled the way your corpus spells it; it continues the prompt's sentence grammatically and then stays in your corpus's register (prose stays prose, verse stays verse, a list stays a list, code stays code); and no phrase of three or more words appears in it twice. One miss and it is not usable. Two shapes. Usable: `<the rest of the prompt's sentence, ending in a period> <a new sentence that could sit in the same paragraph of your corpus>`. Not usable: `<the rest of the sentence> <three words> <the same three words> <the same three words again>` (a loop, the usual temperature-0 failure), or `<the sentence finishes> Copyright <year> <a web address>` (a switch of register toward something GPT-2 read more of than your corpus).

Count the usable ones in each block, write the count in that row's `usable of 5` cell, and run `uv run python scratch/a09-mark.py` again. *You should see* the same table with the usable column filled, then three `where it got worse` lines, one per prompt, naming the first temperature at which usable dropped below 5, and a totals line. Paste those four lines into the fence under the table in `evidence/A09.md`. Whatever your corpus, **`distinct of 5` is 1 at temperature 0 for every prompt** (argmax asks nothing of the seed) and rises with temperature. The usable column is where your corpus shows: plain modern prose survives 1.2 better than verse, code or a list, and temperature 0 often fails on usability the other way, by looping. <!-- JD: fill the Homer nine-row table and the three where-it-got-worse lines after a run. -->

**X4. Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/A09.md` (the questions are below) and run `uv run python scripts/check_evidence.py A09` until it says `all slots filled`. Then:

```bash
git add sampling/ evidence/A09.md pyproject.toml uv.lock
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
`sampling/lab.py` (sampler in NumPy, six commands) + `evidence/A09.md` with every slot filled (the predictions committed before the table runs, the five video continuations, the Step 4, 5 and 7 output, the table with the usable column, where it got worse, the three reflection answers: snapshots compared, local vs API, the worst continuation and the agent paragraph) + `sampling/snapshots.txt` + `sampling/api_t0.json` + `sampling/table.txt`.

**Reflection Questions** (answer under `## Reflection` in `evidence/A09.md`)

1. From `sampling/snapshots.txt`, paste the top three tokens and the tokens-for-90% count for your most concentrated prompt and your least concentrated one, both at `T=1.0`. Quote the last few words of each prompt and say what about where your corpus chunk was cut explains the difference.

   *How to get it:* `grep -n "T=1.0" -A1 sampling/snapshots.txt` prints the three `T=1.0` lines with their `tokens for 90% of the mass` lines; the smallest count is your most concentrated prompt, the largest your least. Each `## prompt` heading ends with that prompt's last forty characters. A prompt cut mid-word or mid-name is concentrated because only a few tokens can finish it; a prompt cut after a period is spread because almost anything can start a sentence.

2. From `sampling/api_t0.json`: how many distinct texts in twenty, and did the first-token `top_logprobs` differ between any two runs? Paste two lists if they did. Your local sampler gave 1 of 5 at temperature 0 on the same prompt. Name the one mechanism your evidence supports for the gap, and the observation you would need to rule it out.

   *How to get it:* both counts and the distinct lists are what `api0` printed, pasted under `## Step 7` in `evidence/A09.md`; `sampling/api_t0.json` holds all twenty runs if you need more. Several lists is the batching-and-floating-point mechanism: on shared hardware your request is computed alongside other people's, the order of floating-point additions changes with the batch, and a near-tie at the top can flip. One list and one text says the serving stack was, for those twenty runs, deterministic.

3. Your X1 prediction row for `T=1.2` (with the commit hash) next to what happened. Paste the worst continuation in your table and say what went wrong in it against the three-part definition of usable. Then give your v2 agent's sampling temperature with the file and line, and say whether this table argues for changing it.

   *How to get it:* the prediction is the `at T=1.2` column of the X1 table; what happened is the `distinct of 5` cell of each `T=1.2` row, which `a09-mark.py` printed beside it, and the hash is the commit whose message says predicted distinct counts: `git log --oneline --grep="predicted distinct counts" -- evidence/A09.md | tail -1`. The worst continuation is the one in `sampling/table.txt` you would least want in a book, pasted with its `## prompt` line; name which of the three conditions it failed (a made-up word, a register switch, a repeated phrase). For the agent's temperature, `grep -rn -i temperature src/` in the v2 repo; in the reference agent it is `TEMPERATURE = 0.2` in `src/llm.ts`, and the argument runs through your usable column: an agent's output is parsed by a program, so the question is at which temperature your table stopped giving five usable out of five.
