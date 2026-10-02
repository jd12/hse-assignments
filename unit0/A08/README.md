# A08 · 3B1B Attention (Ch. 6) + Transformer LLMs 09 + One Head on Your Corpus

**Meetings:** D17–D18 · **Points:** 15 pts

**Watch — 53 min**

**Day 1 — 26 min**
[Attention in transformers, step-by-step, 3Blue1Brown Deep Learning Ch. 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) (26m)
 · Paper, a pencil, and your A07 diagram beside you.

**Day 2 — 27 min**
[Let's build GPT: from scratch, in code, spelled out, Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) · segment 01:02:00 → 01:19:11 (17m): self-attention v4, then his six notes, ending with scaled attention
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 9, Self-Attention (10m)
 · `scratch/a08-video.py` open before you press play. Karpathy first, then DLAI.

**During the video**

**3B1B Ch. 6 · draw the grid.** No code. When the grid of scores for *a fluffy blue creature roamed the verdant forest* appears, pause and draw it: the words along the top and down the side, and every cell he blanks out shaded. Then write, under it, whether the word doing the asking runs along the top or down the side, and what happens to the shaded cells before anything is added up. On Day 2 you print this grid for a sentence of your own, and Karpathy's code puts the asking word down the side, one row each. If your drawing is the other way round, redraw it transposed so the two can be compared.

**Karpathy 01:02:00 → 01:19:11 · type everything he types, and run it.** He is in a notebook; you are in `scratch/a08-video.py`, run with `uv run python scratch/a08-video.py`. He starts from the averaging version you did not watch, so put these four lines at the top before you start:

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
```

Then type what he types: the `B, T, C` toy tensor, `head_size`, the three `nn.Linear`s, `wei`, the `tril` mask, the softmax, and `out = wei @ v`. Print `wei[0]` every time he does. From 01:11:38 he talks through six notes; type the two he demonstrates in code at 01:16:56 (the variance of `wei` with and without the scale, and the softmax of a small vector multiplied by 8), and write one sentence in your log for each of the other four. His correction in the description applies from here on: scale by `head_size`, not `C`.

**DLAI lesson 9 · write one row down.** No code. When a single token's scores are turned into weights, pause and write that row of weights in your log with the tokens it belongs to. Tonight's row is yours.

**Notes**

**`softmax(..., dim=-1)` and nothing else.** With `dim=0` the columns sum to one instead of the rows, nothing errors, and the grid is wrong. The row-sums print in Step 4 is what catches it.

**`k.transpose(-2, -1)`, not `k.T`.** On a 3-D tensor `.T` reverses every dimension and you get a shape error from the `@`, or a deprecation warning and a wrong answer.

**Mask where `tril == 0`.** Masking `tril == 1` hides the past and keeps the future. The symptom is a grid with zeros *below* the diagonal.

**These are tokens, not words.** `cl100k_base` splits a long or rare name into several pieces. If your referent is three tokens, the weight on the referent is the sum of three columns.

**A random head has learned nothing.** Its weights are drawn, not trained, so its grid looks close to even and is different for every seed. That is the correct output and the extension is built on it.

**The essay bans notation.** No matrices, no softmax, no `d_k`, no square roots, no `Q`, `K` or `V`. The words query, key and value are allowed, as English: what a token is asking for, what it advertises, and what it hands over when something matches.

**Walkthrough — one head, one sentence from your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/attention
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: 3B1B Ch. 6, grid drawn and read the right way round
- [ ] Day 1: Step 2, sentence chosen from my corpus
- [ ] Day 1: Step 5 started, first draft of attention.md
- [ ] Day 2: Karpathy 01:02:00 → 01:19:11 typed along in scratch/a08-video.py
- [ ] Day 2: DLAI lesson 9, one row written down
- [ ] Day 2: Steps 3–4, head_on_corpus.py printing a labeled grid
- [ ] Extension: prediction committed, five seeds, table
- [ ] Step 5 finished at ~400 words, push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Choose the sentence from your corpus (Day 1).**

```bash
uv add --dev tiktoken
mkdir -p transformer scratch
```

Create `scratch/a08-find.py`:

```python
import random, re
from pathlib import Path

text = " ".join(Path("data/corpus.txt").read_text(encoding="utf-8").split())
sents = re.split(r"(?<=[.!?])\s+", text)
PRON = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}
cands = [s for s in sents if 8 <= len(s.split()) <= 16
         and PRON & set(re.findall(r"[a-z]+", s.lower()))]
random.seed(0)
for s in random.sample(cands, min(10, len(cands))):
    print("-", s)
print(len(cands), "candidates")
```

```bash
uv run python scratch/a08-find.py
```

*You should see* ten sentences and a candidate count in the hundreds or more. Change the seed until you find one where the pronoun's referent is in the same sentence, before it. If your corpus is not in English, change `PRON` to that language's pronouns.

Write the sentence in your log, and underline the pronoun and its referent. The essay in Step 5 and the extension both use this sentence and no other.

**Step 3. Run one head on it (Day 2, after the video).**

Create `transformer/head_on_corpus.py`. It is the head you typed in the video, with the toy tensor replaced by your sentence:

```python
# transformer/head_on_corpus.py
import sys
import tiktoken, torch
import torch.nn as nn
from torch.nn import functional as F

SENTENCE = "PASTE YOUR SENTENCE HERE"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1337
C, head_size = 32, 16

enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode(SENTENCE)
labels = [enc.decode([i]).strip() or "_" for i in ids]
T = len(ids)

torch.manual_seed(SEED)
tok_emb = nn.Embedding(enc.n_vocab, C)
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

x = tok_emb(torch.tensor(ids))                       # (T, C)
q, k = query(x), key(x)                              # (T, head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5      # (T, T)
wei = wei.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ value(x)                                 # (T, head_size)

wei = wei.detach()
print(f"T={T} seed={SEED} out={tuple(out.shape)}")
print(" " * 10 + "".join(f"{l[:6]:>7}" for l in labels))
for r, l in enumerate(labels):
    print(f"{l[:9]:>9} " + "".join(f"{w:7.2f}" for w in wei[r].tolist()))
print("row sums:", [round(s, 3) for s in wei.sum(dim=-1).tolist()])
```

The only line that is new is `tok_emb`: a table with one random row per token in `cl100k_base`, which is how a token id becomes a vector before any training has happened.

```bash
uv run python transformer/head_on_corpus.py
```

**Step 4. Read the grid.**

*You should see* a `T × T` grid with your tokens labeling both the rows and the columns, and these four things, whatever your sentence:

| Invariant | Why |
|---|---|
| Every cell above the diagonal is `0.00` | The mask. A token cannot see what comes after it. |
| The first row is `1.00` followed by zeros | The first token can only see itself, so all of its budget goes there. |
| Every row sum prints as `1.0` | Softmax along the row. Attention is a budget: more on one token is less on another. |
| From a few rows down, the weights in row `r` sit near `1/(r+1)`, none of them dominant | Random weights produce small, similar scores, and softmax of similar scores is close to even. The first two or three rows have too few cells to average out; `0.76 0.24` in row 1 is normal. |

*If it broke:* zeros below the diagonal mean you masked `tril == 1`. Row sums that are not all 1.0 mean `dim=0`. A `T` much larger than the word count is fine; count the labels and you will find the words that split.

The fourth row of that table is the whole story of a random head. It is Karpathy's averaging version with some noise on it. Nothing in it knows what a pronoun is.

**Step 5. Write ~400 words in `transformer/attention.md`.**

Plain English, exactly one head, zero equations, about your sentence from Step 2 throughout. Describe the head you would want: one that connects your pronoun to its referent. Every position puts out a request, every position advertises what it has, and how well they match decides how much of each earlier position's value is added to the asking position's residual stream. Information flows *into* the asking token. Say whose vector changed and what was added to it.

Say what the mask does in your sentence, naming the words, and why the budget matters. Say why there are many heads rather than one wide one: a head that tracks your pronoun is not also doing some other job in that layer.

The test: someone who has taken AP CS A and no machine learning reads it once and can say what would change if you removed the mask. Run that test on someone.

```bash
wc -w transformer/attention.md
```

*You should see* something near 400. Six hundred means you have not cut yet.

**Extension — what a random head does to your pronoun (ASSIGNED)**

**X1. Predict, and commit the prediction.** Create `transformer/HEAD_RUN.md` with your sentence, the pronoun token and its position in the grid, the referent token(s) and their columns, and two predictions: the weight this random head's pronoun row puts on the referent (summed over its tokens), and the weight a trained pronoun head would put there. Then:

```bash
git add transformer/head_on_corpus.py transformer/HEAD_RUN.md
git commit -m "A08: prediction for the pronoun row, before running seeds"
```

I will check your commit timestamps.

**X2. Five seeds.**

```bash
for s in 1 2 3 4 5; do uv run python transformer/head_on_corpus.py $s; done > transformer/head_runs.txt
```

The baseline is the even share: if the pronoun is at position `p` (counting from 0) and the referent is `n` tokens, a head that spreads its budget evenly puts `n/(p+1)` on the referent. Build this table in `HEAD_RUN.md` from the pronoun's row in each run:

| seed | weight on referent | even share | ratio | biggest weight in the row, and which token |
|---|---|---|---|---|

**X3. Say what it shows.** Give the mean and the range of the referent weight across the five seeds, and how many seeds put *less* than the even share on the referent. Then say plainly, in two or three sentences, what a random head shows and what a trained one would: a random head spreads its budget close to evenly and which token wins moves with the seed; a trained pronoun head would put most of the row on the referent and would do it the same way every time, because its weights were learned rather than drawn. Put your X1 prediction next to what happened.

**X4. Commit, push, PR, sign off.**

```bash
git add transformer/attention.md transformer/HEAD_RUN.md transformer/head_runs.txt scratch/a08-video.py scratch/a08-find.py pyproject.toml uv.lock
git commit -m "A08: attention in 400 words, one random head on a corpus sentence over 5 seeds"
git push -u origin dev/attention
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the sentence in `attention.md` you are least sure is true.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`transformer/attention.md` (~400 words, no notation, your sentence) + `transformer/head_on_corpus.py` + `transformer/HEAD_RUN.md` + `transformer/head_runs.txt` + `scratch/a08-video.py`.

**Reflection Questions**

1. Paste the pronoun's row from seed 1 with the column labels above it. Which token got the biggest weight, and by how much did it beat the even share? Then quote the sentence in your `attention.md` that describes what that row should look like in a trained model, and say which number in the pasted row it disagrees with.

   *How to get it:* seed 1 is the first block of `transformer/head_runs.txt`, or run it again. With your pronoun's position `p` from X1 (counting from 0), the row you want is the one labeled with the pronoun token; the column labels are line 2 of the output:

   ```bash
   uv run python transformer/head_on_corpus.py 1 | grep -n "^ *he \|^  *But Apollo"
   ```

   Put your pronoun in place of `he` and the first two column labels of your sentence in place of `But Apollo`. The even share for one token in that row is `1/(p+1)`; "beat it by" is the biggest weight minus that.

2. Your X1 prediction for the random head, with the commit hash, next to the five-seed mean and range. How many seeds went below the even share? If your prediction was far off, say what you assumed about an untrained head that the run showed was wrong.

   *How to get it:* the prediction is the number you wrote in `HEAD_RUN.md` before any run; the mean, range and below-the-share count are the X3 lines under your table. The hash is the first commit that touched the file:

   ```bash
   git log --oneline --follow -- transformer/HEAD_RUN.md | tail -1
   ```

3. Change `dim=-1` to `dim=0` in the softmax line, run seed 1, and paste the first three rows and the `row sums:` line. Say which invariant from the Step 4 table broke and which still held. Then name the claim in your `attention.md` that would be false for a head built that way. Put the line back before you commit.

   *How to get it:* change only the `F.softmax(wei, dim=-1)` line; the `wei.sum(dim=-1)` line below it stays, or the row-sums print will lie to you. Then:

   ```bash
   uv run python transformer/head_on_corpus.py 1 | sed -n '2,5p;$p'
   ```

   *You should see* the first row no longer `1.00`, and row sums that climb from near zero toward one instead of all reading `1.0`; the zeros above the diagonal are still there. Afterwards `git diff transformer/head_on_corpus.py` should show nothing but that one line, and `git restore transformer/head_on_corpus.py` puts it back.
