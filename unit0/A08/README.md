# A08 · 3B1B Attention (Ch. 6) + Transformer LLMs 09 + One Head on Your Corpus

**Meetings:** D17–D18 · **Points:** 15 pts

**Watch — 53 min**

**Day 1 — 26 min**
[Attention in transformers, step-by-step, 3Blue1Brown Deep Learning Ch. 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) (26m)
 · Paper, a pencil, and your A07 `SHAPES.md` beside you.

**Day 2 — 27 min**
[Let's build GPT: from scratch, in code, spelled out, Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) · segment 01:02:00 → 01:19:11 (17m): self-attention v4, then his six notes, ending with scaled attention
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 9, Self-Attention (10m)
 · The output of `scratch/a08-video.py` in your terminal before you press play. Karpathy first, then DLAI.

**During the video**

**3B1B Ch. 6 · read the grid the right way round.** No code, no drawing. When the grid of scores for *a fluffy blue creature roamed the verdant forest* appears, pause and write two lines in your log: whether the word doing the asking runs along the top or down the side, and what happens to the cells he blanks out before anything is added up. On Day 2 you print this grid for a sentence of your own, and Karpathy's code puts the asking word down the side, one row each. **If his grid is the other way round, say so in your log**, so you do not read your own grid transposed.

**Karpathy 01:02:00 → 01:19:11 · read along; the file has already run.** He is in a notebook; your `scratch/a08-video.py` from Step 3 has his v4 head on his toy tensor in Part 1 and the same head on your sentence in Part 2, and you ran it before pressing play. He starts from the averaging version you did not watch, then builds the head piece by piece: the `B, T, C` toy tensor, `head_size`, the three `nn.Linear`s, `wei`, the `tril` mask, the softmax, `out = wei @ v`. Pause when he prints `wei[0]` and compare it with your Part 1 grid: his numbers differ from yours because his random draw differs, but the shape of the grid, the `1.00` in the corner and the zeros above the diagonal are the same. Then look at Part 2, which is the same three matrices on your sentence, with your tokens on the rows.

From 01:11:38 he talks through six notes. Part 3 of your output has one numbered line for each; pause at each note and read its line:

1. *Attention is communication*, nodes in a graph aggregating from the nodes that point at them. Line 1 counts, for every position in your sentence, how many positions send into it: `1, 2, 3, ...` up to `T`.
2. *No notion of space.* Line 2 shows your sentence reversed gives the same vectors in reverse order, `True`: nothing in `x` says where a token sits. Positions come later, in A09.
3. *Batch elements never talk.* Line 3 zeroes out sequence 1 of his toy tensor and shows sequence 0's grid did not move.
4. *Delete the mask line and you have an encoder.* Line 4 is row 0 of your sentence with the mask deleted: no longer `1.00` and zeros, but a spread over all `T` tokens, still summing to one.
5. *Self-attention means keys and values come from the same place as queries.* Line 5 takes keys and values from a different text and the grid stops being square.
6. *Scale by one over the square root of head size.* Line 6 is his demonstration, the variance of `wei` with and without the scale and the softmax of a small vector times 8, and then the same thing on your sentence: the biggest weight in your last row with the scale and without it. His correction in the description applies from here on: scale by `head_size`, not `C`.

Write one sentence in your log for each of the six, saying what the line shows in your own words.

**DLAI lesson 9 · write one row down.** No code. When a single token's scores are turned into weights, pause and write that row of weights in your log with the tokens it belongs to. Tonight's row is yours.

**Notes**

**`softmax(..., dim=-1)` and nothing else.** With `dim=0` the columns sum to one instead of the rows, nothing errors, and the grid is wrong. The row-sums print in Step 5 is what catches it.

**`k.transpose(-2, -1)`, not `k.T`.** On a 3-D tensor `.T` reverses every dimension and you get a shape error from the `@`, or a deprecation warning and a wrong answer.

**Mask where `tril == 0`.** Masking `tril == 1` hides the past and keeps the future. The symptom is a grid with zeros *below* the diagonal.

**These are tokens, not words.** `cl100k_base` splits a long or rare name into several pieces. If your referent is three tokens, the weight on the referent is the sum of three columns, and `a08-table.py` takes a list of positions for exactly that reason.

**A random head has learned nothing.** Its weights are drawn, not trained, so its grid looks close to even and is different for every seed. That is the correct output and the extension is built on it.

**The paragraph bans notation.** No matrices, no softmax, no `d_k`, no square roots, no `Q`, `K` or `V`. The words query, key and value are allowed, as English: what a token is asking for, what it advertises, and what it hands over when something matches.

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
- [ ] Day 1: 3B1B Ch. 6, which way the grid reads written in the log
- [ ] Day 1: Step 2, sentence with a pronoun and its referent chosen from my corpus
- [ ] Day 1: Step 3, scratch/a08-video.py run once, output in the log
- [ ] Day 1: Step 6 started, prompts 1 and 2 drafted in attention.md
- [ ] Day 2: Karpathy 01:02:00 → 01:19:11 read along, six sentences in the log
- [ ] Day 2: DLAI lesson 9, one row written down
- [ ] Day 2: Steps 4–5, head_on_corpus.py printing a labeled grid, four invariants checked
- [ ] Day 2: Step 6 finished, 150 to 200 words, four prompts in order
- [ ] Extension: prediction committed, five seeds, table printed and pasted
- [ ] Push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Choose the sentence from your corpus (Day 1).**

`torch` and `tiktoken` are installed from A07. Create `scratch/a08-find.py` (right-click `scratch`, **New File**) and paste this in. It splits your corpus into sentences and prints ten of eight to sixteen words that contain a pronoun.

```python
# scratch/a08-find.py
# Run from the repo root:  uv run python scratch/a08-find.py
import random, re
from pathlib import Path

text = " ".join(Path("data/corpus.txt").read_text(encoding="utf-8").split())
sents = re.split(r"(?<=[.!?])\s+", text)        # splits after a period, exclamation mark or question mark followed by a space
PRON = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}
cands = [s for s in sents if 8 <= len(s.split()) <= 16
         and PRON & set(re.findall(r"[a-z]+", s.lower()))]    # the sentence's words, lowercased, letters only
random.seed(0)
for s in random.sample(cands, min(10, len(cands))):
    print("-", s)
print(len(cands), "candidates")
```

```bash
uv run python scratch/a08-find.py
```

*You should see* ten sentences and a candidate count in the hundreds or more. **You need a sentence where the pronoun's referent is in the same sentence, before it**: a name or a noun earlier in the sentence that the pronoun stands for. Change `random.seed(0)` to `1`, `2`, and so on until one of the ten fits. On Homer, seed 0 printed `664 candidates` and, among its ten, the sentence used in every example below: `But Apollo looked down from Pergamus and called aloud to the Trojans, for he was displeased.` The pronoun is `he`, the referent is `Apollo`. If your corpus is not in English, change `PRON` to that language's pronouns.

Write the sentence in your log, and underline the pronoun and its referent. The paragraph in Step 6 and the extension both use this sentence and no other.

**Step 3. Create `scratch/a08-video.py` and run it once (Day 1, so it is ready for Day 2).**

Paste all of this in. Part 1 is Karpathy's head as he types it in the segment, on his toy tensor. Part 2 is the same head, same three matrices, on your sentence. Part 3 is one line per note.

```python
# scratch/a08-video.py
# Run from the repo root:  uv run python scratch/a08-video.py
# Part 1 is Karpathy's self-attention v4 on his toy tensor, as he types it from 01:02:00.
# Part 2 is the same head, same three matrices, on one sentence from your corpus.
# Part 3 is one line of output for each of his six notes, starting at 01:11:38.
import tiktoken, torch
import torch.nn as nn
from torch.nn import functional as F

SENTENCE = "PASTE YOUR SENTENCE HERE"      # the sentence you chose in Step 2

torch.manual_seed(1337)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=140)

# ------------------------------------------------------------ Part 1: his toy tensor, 01:02:00 -> 01:11:38
B, T, C = 4, 8, 32           # batch, time, channels: four sequences of eight positions, 32 numbers each
x = torch.randn(B, T, C)     # random numbers standing in for token vectors

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)                                   # (B, T, 16)  what every position advertises
q = query(x)                                 # (B, T, 16)  what every position is asking for
wei = q @ k.transpose(-2, -1) * head_size**-0.5    # (B, T, T) scores; head_size, not C: his correction in the description

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))    # a position cannot see what comes after it
wei = F.softmax(wei, dim=-1)                       # every row becomes a budget that sums to one
v = value(x)                                       # (B, T, 16)  what every position hands over
out = wei @ v                                      # (B, T, 16)

print("PART 1: his toy tensor")
print("x", tuple(x.shape), " k, q, v", tuple(k.shape), " wei", tuple(wei.shape), " out", tuple(out.shape))
print("wei[0], the grid for the first of his four sequences:")
print(wei[0].detach())

# ------------------------------------------------------------ Part 2: the same head on your sentence
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode(SENTENCE)
labels = [enc.decode([i]).strip() or "_" for i in ids]
Ts = len(ids)
tok_emb = nn.Embedding(enc.n_vocab, C)             # one random row of 32 numbers per token id
xs = tok_emb(torch.tensor([ids]))                  # (1, Ts, C): your sentence in his shape, B = 1

def attend(xq, xkv, masked=True):
    """ queries from xq, keys and values from xkv; the mask only makes sense when they are the same """
    q, k, v = query(xq), key(xkv), value(xkv)
    wei = q @ k.transpose(-2, -1) * head_size**-0.5
    if masked:
        Tq = xq.shape[1]
        wei = wei.masked_fill(torch.tril(torch.ones(Tq, Tq)) == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei.detach(), (wei @ v).detach()

wei_s, out_s = attend(xs, xs)
print(f"\nPART 2: the same head on your sentence, {Ts} tokens")
print("xs", tuple(xs.shape), " wei", tuple(wei_s.shape), " out", tuple(out_s.shape))
print("the first four rows, with your tokens on them:")
print(" " * 9 + "".join(f"{l[:6]:>7}" for l in labels[:4]))
for r in range(4):
    print(f"{labels[r][:8]:>8} " + "".join(f"{w:7.2f}" for w in wei_s[0, r, :4].tolist()))

# ------------------------------------------------------------ Part 3: his six notes, 01:11:38 -> 01:19:11
print("\nPART 3: the six notes")

# note 1: attention is communication. Every token is a node; the nonzero weights in its row are the arrows pointing into it.
edges = (wei_s[0] > 0).sum(dim=-1).tolist()
print(f"1. nodes that send into each position, row by row: {edges}")
print(f"   the first token hears from {edges[0]} node (itself); the last, '{labels[-1]}', hears from all {edges[-1]}")

# note 2: no notion of space. x is built from the token ids alone, so the same token gets the same vector wherever it sits.
xs_rev = tok_emb(torch.tensor([ids[::-1]]))
print(f"2. your sentence reversed gives the same vectors in reverse order: {torch.equal(xs.flip(1), xs_rev)}  (nothing in x says where a token is; positions are added later)")

# note 3: the B sequences never talk. Wipe out sequence 1 of his toy tensor and sequence 0's grid does not move.
x_wiped = x.clone()
x_wiped[1] = 0
wei_wiped, _ = attend(x_wiped, x_wiped)
print(f"3. after zeroing sequence 1 of his toy tensor, sequence 0's grid is unchanged: {torch.allclose(wei[0].detach(), wei_wiped[0])}")

# note 4: encoder = delete the masking line. Every token then sees the whole sentence, including what comes after it.
wei_enc, _ = attend(xs, xs, masked=False)
print(f"4. with the mask line deleted, row 0 of your sentence becomes: {[round(w, 2) for w in wei_enc[0, 0].tolist()]}")
print(f"   it still sums to {wei_enc[0, 0].sum().item():.2f}, and the first token now hears from all {int((wei_enc[0, 0] > 0).sum())} tokens")

# note 5: self-attention means k and v come from the same x as q. Take them from a different text and the grid stops being square.
OTHER = " ".join(open("data/corpus.txt", encoding="utf-8").read().split()[:12])      # the first twelve words of your corpus
xo = tok_emb(torch.tensor([enc.encode(OTHER)]))
wei_cross, _ = attend(xs, xo, masked=False)
print(f"5. queries from your sentence, keys and values from another text ({xo.shape[1]} tokens): wei is {tuple(wei_cross.shape)}, not square. That is cross-attention.")

# note 6: scale by 1/sqrt(head_size). His demonstration, then yours.
kk, qq = torch.randn(B, T, head_size), torch.randn(B, T, head_size)
raw = qq @ kk.transpose(-2, -1)
print(f"6. variance of k {kk.var():.2f}, of q {qq.var():.2f}, of q @ k^T without the scale {raw.var():.2f}, with the scale {(raw * head_size**-0.5).var():.2f}")
small = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print(f"   softmax of a small vector      {[round(w, 2) for w in F.softmax(small, dim=-1).tolist()]}")
print(f"   softmax of the same vector * 8 {[round(w, 2) for w in F.softmax(small * 8, dim=-1).tolist()]}")
q_s, k_s = query(xs), key(xs)
unscaled = F.softmax((q_s @ k_s.transpose(-2, -1)).masked_fill(torch.tril(torch.ones(Ts, Ts)) == 0, float("-inf")), dim=-1).detach()
print(f"   biggest weight in your last row, '{labels[-1]}': with the scale {wei_s[0, -1].max():.2f}, without it {unscaled[0, -1].max():.2f}")
```

Put your sentence from Step 2 in the `SENTENCE =` slot. Then:

```bash
uv run python scratch/a08-video.py
```

*You should see* three parts. Part 1: four shapes and an 8 × 8 grid with `1.00` in the top-left corner and zeros above the diagonal. Part 2: three shapes with your `T`, and the first four rows of your grid with your tokens on them, `1.00` first. Part 3: six numbered lines. Whatever your sentence, line 1 counts up from 1 to your `T`, lines 2 and 3 say `True`, line 4's row sums to `1.00` over all `T` tokens, line 5's shape is not square, and line 6's unscaled variance is near `head_size`, 16, while the scaled one is near 1. On Homer:

```text
PART 2: the same head on your sentence, 23 tokens
xs (1, 23, 32)  wei (1, 23, 23)  out (1, 23, 16)
the first four rows, with your tokens on them:
             But Apollo looked   down
     But    1.00   0.00   0.00   0.00
  Apollo    0.42   0.58   0.00   0.00
  looked    0.27   0.47   0.26   0.00
    down    0.33   0.16   0.14   0.37

PART 3: the six notes
1. nodes that send into each position, row by row: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
   the first token hears from 1 node (itself); the last, '.', hears from all 23
2. your sentence reversed gives the same vectors in reverse order: True  (nothing in x says where a token is; positions are added later)
3. after zeroing sequence 1 of his toy tensor, sequence 0's grid is unchanged: True
4. with the mask line deleted, row 0 of your sentence becomes: [0.04, 0.03, 0.03, 0.04, 0.03, 0.05, 0.06, 0.06, 0.03, 0.06, 0.03, 0.08, 0.04, 0.04, 0.03, 0.02, 0.03, 0.05, 0.04, 0.03, 0.05, 0.05, 0.1]
   it still sums to 1.00, and the first token now hears from all 23 tokens
5. queries from your sentence, keys and values from another text (16 tokens): wei is (1, 23, 16), not square. That is cross-attention.
6. variance of k 1.07, of q 1.14, of q @ k^T without the scale 18.00, with the scale 1.12
   softmax of a small vector      [0.19, 0.14, 0.24, 0.14, 0.29]
   softmax of the same vector * 8 [0.03, 0.0, 0.16, 0.0, 0.8]
   biggest weight in your last row, '.': with the scale 0.08, without it 0.28
```

Paste the whole output into your log. Line 6's last line is the whole reason for the scale: without it, one token in your last row takes a quarter of the budget for no reason but the size of the numbers.

*If it broke* with `ModuleNotFoundError`, A07's Step 2 did not finish. `FileNotFoundError: data/corpus.txt` means the terminal is not in the repo root; `pwd` should end in `foundations-<your-username>`.

**Step 4. Run one head on it (Day 2, after the video).**

Create `transformer/head_on_corpus.py`. It is the head from Part 1, with the toy tensor replaced by your sentence and a seed on the command line:

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

Put the same sentence in the `SENTENCE =` slot. The only line that is new against Part 1 is `tok_emb`: a table with one random row per token in `cl100k_base`, which is how a token id becomes a vector before any training has happened. A07's `block_shapes.py` used the same line.

```bash
uv run python transformer/head_on_corpus.py
```

**Step 5. Read the grid.**

*You should see* a `T × T` grid with your tokens labeling both the rows and the columns, and these four things, whatever your sentence:

| Invariant | Why |
|---|---|
| Every cell above the diagonal is `0.00` | The mask. A token cannot see what comes after it. |
| The first row is `1.00` followed by zeros | The first token can only see itself, so all of its budget goes there. |
| Every row sum prints as `1.0` | Softmax along the row. Attention is a budget: more on one token is less on another. |
| From a few rows down, the weights in row `r` sit near `1/(r+1)`, none of them dominant | Random weights produce small, similar scores, and softmax of similar scores is close to even. The first two or three rows have too few cells to average out; `0.76 0.24` in row 1 is normal. |

On Homer the first line is `T=23 seed=1337 out=(23, 16)`, the `he` row near the bottom has 19 nonzero cells, the biggest `0.10`, and the row-sums line is twenty-three `1.0`s.

*If it broke:* zeros below the diagonal mean you masked `tril == 1`. Row sums that are not all 1.0 mean `dim=0`. A `T` much larger than the word count is fine; count the labels and you will find the words that split: on Homer, `Pergamus` is `P`, `erg`, `amus` and `Trojans` is `Tro`, `j`, `ans`.

The fourth row of that table is the whole story of a random head. It is Karpathy's averaging version with some noise on it. Nothing in it knows what a pronoun is.

**Step 6. Write the guided paragraph in `transformer/attention.md`.**

Create `transformer/attention.md` and paste this in. One paragraph of 150 to 200 words, plain English, exactly one head, zero notation, answering the four prompts in order, every one of them about your sentence from Step 2. Prompts 1 and 2 can be drafted on Day 1 from the 3B1B video; prompts 3 and 4 need your grid.

```markdown
# One head on my sentence: <your name>

Sentence: <your sentence>
Pronoun: <token> at position <p>. Referent: <token(s)> at position(s) <   >.

<One paragraph, 150 to 200 words, answering these four in order. Delete the prompts when you are done.
1. What the pronoun asks for. At the pronoun's position a head that resolves pronouns puts out a request;
   say in English what that request is in your sentence.
2. What the referent advertises. Say what the referent's position announces that matches the request, what
   it hands over when they match, and whose vector changes because of it: the referent's or the pronoun's.
3. What the mask hides. Name the words in your sentence after the pronoun that the pronoun cannot see, and
   say why a model that predicts the next token must not see them.
4. Why the weights sum to one and what that costs. Every row of your grid adds up to one; say what that means
   for the referent when other tokens in the row also get some weight, and what a trained head would do about it.>
```

The test: someone who has taken AP CS A and no machine learning reads it once and can say what would change if you removed the mask. Run that test on someone.

```bash
wc -w transformer/attention.md
```

*You should see* a count between about 185 and 235: your 150 to 200 words plus roughly 35 in the three header lines. Three hundred means you have not cut yet.

**Extension — what a random head does to your pronoun (ASSIGNED)**

**X1. Predict, and commit the prediction.** First find the positions. The grid labels tokens, not words, so count tokens, from 0, with your sentence in the slot:

```bash
uv run python -c "
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')
SENTENCE = 'PASTE YOUR SENTENCE HERE'
for i, t in enumerate(enc.encode(SENTENCE)): print(i, repr(enc.decode([t])))
"
```

*You should see* one token per line, numbered from 0. On Homer, 23 lines: `0 'But'`, `1 ' Apollo'`, ..., `17 ' for'`, `18 ' he'`, `19 ' was'`, `20 ' disple'`, `21 'ased'`, `22 '.'`. So on Homer the pronoun is at position 18 and the referent at position 1. A referent that split into three tokens, like `Pergamus` at 5, 6, 7, is three positions, and its weight is the sum of three columns.

Now the prediction. **The even share is the baseline**: if the pronoun is at position `p` (counting from 0) it can see `p + 1` tokens, so a head that spreads its budget evenly puts `1/(p+1)` on each, and `n/(p+1)` on a referent of `n` tokens. On Homer that is `1/19 = 0.0526`. Create `transformer/HEAD_RUN.md` and paste this in, filled:

```markdown
# A08 head run: <your name>

Sentence: <your sentence>
Pronoun: <token> at position <p>. Referent: <token(s)> at position(s) <   >, n = <   > token(s).
Even share on the referent: n / (p + 1) = <   > / <   > = <0.xxxx>

## Prediction (before any run)

Weight this random head's pronoun row will put on the referent, summed over its tokens: <0.xx>
Weight a trained pronoun head would put there: <0.xx>
Why these two numbers: <one sentence each>
```

Commit before you run a single seed:

```bash
git add transformer/head_on_corpus.py transformer/HEAD_RUN.md
git commit -m "A08: prediction for the pronoun row, before running seeds"
```

I will check your commit timestamps.

**X2. Five seeds, read by a script.**

```bash
for s in 1 2 3 4 5; do uv run python transformer/head_on_corpus.py $s; done > transformer/head_runs.txt
wc -l transformer/head_runs.txt
```

*You should see* five times `T + 3` lines; on Homer, `130 transformer/head_runs.txt`. Five grids of 23 rows is too many numbers to read by eye, so a script reads them. Create `scratch/a08-table.py` and paste this in:

```python
# scratch/a08-table.py
# Run from the repo root:  uv run python scratch/a08-table.py
# Reads the five runs in transformer/head_runs.txt and prints the X2 table for HEAD_RUN.md.
# Run it once with the two slots empty: it prints every token with its position. Fill the
# slots from that list, run it again, and it prints the table, the summary lines, and the
# pronoun's row from seed 1 with its labels.
from pathlib import Path

PRONOUN_POS = None             # position of your pronoun token, counting from 0; e.g. 18
REFERENT_POSITIONS = []        # position(s) of the referent's token(s), e.g. [1] or [5, 6, 7]

text = Path("transformer/head_runs.txt").read_text(encoding="utf-8").splitlines()
runs = []                                                   # one (seed, labels, grid) per block of the file
i = 0
while i < len(text):
    if text[i].startswith("T="):
        T = int(text[i].split()[0][2:])
        seed = int(text[i].split()[1][5:])
        header = text[i + 1]
        labels = [header[10 + 7 * c: 17 + 7 * c].strip() for c in range(T)]      # the column labels, 7 characters each
        grid = [[float(w) for w in text[i + 2 + r][10:].split()] for r in range(T)]   # row r: label in 9 characters, then T weights
        runs.append((seed, labels, grid))
        i += 2 + T
    i += 1
assert runs, "transformer/head_runs.txt is empty or not in head_on_corpus.py's format"
seed0, labels, grid0 = runs[0]

if PRONOUN_POS is None or not REFERENT_POSITIONS:
    print(f"{len(runs)} runs of T={len(labels)} tokens. Positions, counting from 0:")
    for p, l in enumerate(labels):
        print(f"{p:>3}  {l}")
    print("\nPut the pronoun's position in PRONOUN_POS and the referent's position(s) in REFERENT_POSITIONS, then run again.")
    raise SystemExit

p, n = PRONOUN_POS, len(REFERENT_POSITIONS)
assert all(r < p for r in REFERENT_POSITIONS), "the referent has to come before the pronoun, or the mask hides it"
even_one = 1 / (p + 1)                 # a head that spreads its budget evenly gives each visible token this much
even = n * even_one                    # and the referent, over its n tokens, this much
print(f"pronoun '{labels[p]}' at {p}: {p + 1} tokens visible, even share {even_one:.4f} each, {even:.4f} on the referent ({n} token(s): {[labels[r] for r in REFERENT_POSITIONS]})\n")

print("| seed | weight on referent | even share | ratio | biggest weight in the row, and which token |")
print("|---|---|---|---|---|")
weights = []
for seed, lab, grid in runs:
    row = grid[p][: p + 1]
    w = sum(row[r] for r in REFERENT_POSITIONS)
    big = max(range(p + 1), key=lambda c: row[c])
    weights.append(w)
    print(f"| {seed} | {w:.2f} | {even:.4f} | {w / even:.2f} | {row[big]:.2f} on '{lab[big]}' (position {big}) |")

below = sum(w < even for w in weights)
print(f"\nmean {sum(weights) / len(weights):.3f}   range {min(weights):.2f} to {max(weights):.2f}   seeds below the even share: {below} of {len(weights)}")

print(f"\nseed {seed0}, the pronoun's row with its column labels (visible tokens only):")
print(" " * 9 + "".join(f"{l[:6]:>7}" for l in labels[: p + 1]))
print(f"{labels[p][:8]:>8} " + "".join(f"{w:7.2f}" for w in grid0[p][: p + 1]))
```

Run it once with the two slots as they are:

```bash
uv run python scratch/a08-table.py
```

*You should see* your tokens again, one per line, numbered from 0, read this time from the column labels in `head_runs.txt`, and a line telling you to fill the slots. The numbers match the ones you found in X1; the labels are the first six characters of each token, as the grid prints them, so `disple` is `displeased` cut short. Put your pronoun's position in `PRONOUN_POS` and the referent's position or positions in `REFERENT_POSITIONS`; on Homer, `PRONOUN_POS = 18` and `REFERENT_POSITIONS = [1]`. Run it again:

```bash
uv run python scratch/a08-table.py
```

*You should see* one line naming your pronoun and the even share, a five-row Markdown table, one summary line, and the pronoun's row from seed 1 with its labels. Whatever your sentence, the `even share` column is the same number in every row, the `ratio` column hovers around 1 and moves with the seed, and the biggest weight in the row changes token from seed to seed. On Homer:

```text
pronoun 'he' at 18: 19 tokens visible, even share 0.0526 each, 0.0526 on the referent (1 token(s): ['Apollo'])

| seed | weight on referent | even share | ratio | biggest weight in the row, and which token |
|---|---|---|---|---|
| 1 | 0.06 | 0.0526 | 1.14 | 0.13 on 'from' (position 4) |
| 2 | 0.06 | 0.0526 | 1.14 | 0.10 on 'P' (position 5) |
| 3 | 0.04 | 0.0526 | 0.76 | 0.07 on 'looked' (position 2) |
| 4 | 0.05 | 0.0526 | 0.95 | 0.08 on 'j' (position 14) |
| 5 | 0.08 | 0.0526 | 1.52 | 0.08 on 'Apollo' (position 1) |

mean 0.058   range 0.04 to 0.08   seeds below the even share: 2 of 5

seed 1, the pronoun's row with its column labels (visible tokens only):
             But Apollo looked   down   from      P    erg   amus    and called  aloud     to    the    Tro      j    ans      ,    for     he
      he    0.08   0.06   0.03   0.07   0.13   0.06   0.03   0.03   0.02   0.03   0.08   0.03   0.03   0.09   0.02   0.03   0.07   0.07   0.03
```

Seed 5 put its biggest weight on `Apollo`, and it is 0.08, half again the even share: that is what a lucky draw looks like, and seed 3 undoes it. Paste the table and the summary line into `HEAD_RUN.md` under `## X2. Five seeds`.

*If it broke* with `AssertionError: the referent has to come before the pronoun`, your referent is after the pronoun and the mask hides it; choose a different sentence in Step 2. A `ValueError` on the `float(w)` line means `head_runs.txt` was written by a changed `head_on_corpus.py`; put the print lines back and re-run the `for` loop.

**X3. Say what it shows.** Under `## X3`, the mean and the range from the summary line, how many seeds put *less* than the even share on the referent, and then two or three plain sentences: what a random head shows and what a trained one would. A random head spreads its budget close to evenly and which token wins moves with the seed; a trained pronoun head would put most of the row on the referent and would do it the same way every time, because its weights were learned rather than drawn. Put your X1 prediction next to what happened.

**X4. Commit, push, PR, sign off.**

```bash
git add transformer/attention.md transformer/HEAD_RUN.md transformer/head_runs.txt scratch/a08-video.py scratch/a08-find.py scratch/a08-table.py
git commit -m "A08: one head in 200 words, one random head on a corpus sentence over 5 seeds"
git push -u origin dev/attention
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the sentence in `attention.md` you are least sure is true.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`transformer/attention.md` (150 to 200 words, four prompts in order, no notation, your sentence) + `transformer/head_on_corpus.py` + `transformer/HEAD_RUN.md` (prediction committed first, the printed table, X3) + `transformer/head_runs.txt` + `scratch/a08-video.py` + `scratch/a08-find.py` + `scratch/a08-table.py`.

**Reflection Questions**

1. Paste the pronoun's row from seed 1 with the column labels above it. Which token got the biggest weight, and by how much did it beat the even share for one token? Then quote the sentence in your `attention.md` that says what that row should look like in a trained head, and say which number in the pasted row it disagrees with.

   *How to get it:* the row and its labels are the last two lines `uv run python scratch/a08-table.py` prints; the biggest weight and its token are in the seed 1 row of the table above it. The even share for one token is `1/(p+1)`, printed on the script's first line as `even share ... each`; "beat it by" is the biggest weight minus that. On Homer: `0.13` on `from`, against `0.0526`, so by about `0.08`.

2. Your X1 prediction for the random head, with the commit hash, next to the five-seed mean and range. How many seeds went below the even share? If your prediction was far off, say what you assumed about an untrained head that the run showed was wrong.

   *How to get it:* the prediction is the number you wrote in `HEAD_RUN.md` before reading any run; the mean, range and below-the-share count are the script's summary line, pasted in X2. The hash is the first commit that touched the file:

   ```bash
   git log --oneline --follow -- transformer/HEAD_RUN.md | tail -1
   ```

3. Change `dim=-1` to `dim=0` in the softmax line of `transformer/head_on_corpus.py`, run seed 1, and paste the first three rows and the `row sums:` line. Say which invariant from the Step 5 table broke and which still held. Then name the claim in your `attention.md` that would be false for a head built that way. Put the line back before you commit.

   *How to get it:* change only the `F.softmax(wei, dim=-1)` line; the `wei.sum(dim=-1)` line below it stays, or the row-sums print will lie to you. Then:

   ```bash
   uv run python transformer/head_on_corpus.py 1 | sed -n '2,5p;$p'
   ```

   *You should see* the first row no longer `1.00`, and row sums that start near zero and climb past one instead of all reading `1.0`; the zeros above the diagonal are still there, because the mask is applied before the softmax and does not care which way the softmax runs. On Homer the first row read `0.05` and the row sums ran from `0.053` up to `3.517`; it is the columns that sum to one now. Afterwards `git diff transformer/head_on_corpus.py` should show nothing but that one line, and `git restore transformer/head_on_corpus.py` puts it back.
