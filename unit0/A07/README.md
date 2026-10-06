# A07 · 3B1B Transformers (Ch. 5) + Transformer LLMs 07–08 + Build the Block

**Meetings:** D15–D16 · **Points:** 15 pts

**Watch — 55 min**

**Day 1 — 27 min**
[Transformers, the tech behind LLMs, 3Blue1Brown Deep Learning Ch. 5](https://www.youtube.com/watch?v=wjZofJX0v4M) (27m)
 · Paper and a pencil beside you for the tally. Nothing to type during this one; Steps 2–5 come after it.

**Day 2 — 28 min**
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 7, Architectural Overview (6m) · lesson 8, The Transformer Block (6m)
[Let's build GPT: from scratch, in code, spelled out, Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) · segment 01:21:59 → 01:37:49 (16m): multi-headed self-attention, feedforward, residual connections, layernorm
 · Your printed shape table from Day 1 open beside you, and `transformer/block_shapes.py` open in VS Code.

Reference, not assigned: [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/). Use it to check a shape you are unsure of. Its figures are encoder-decoder; see Notes.

**During the video**

**3B1B Ch. 5 · write down the tally.** He counts GPT-3's weights as he goes, matrix by matrix, keeping a running total. Each time he adds a matrix to the tally, pause and write its name and its shape in your log, with his numbers. When he finishes, go back over your list and mark which number is the vocabulary size, which is the embedding dimension, and which is the context length. **Three numbers, three jobs**, and Step 5 puts all three next to the three your script prints.

**DLAI lessons 7–8 · compare, do not copy.** No code. When lesson 8 puts the block on screen, pause and hold your printed SHAPE TABLE next to it. Write down one row of your table the picture has no box for, and one box in the picture your table has no row for. Lesson 7's picture of the whole model, with the block repeated, is what the `x in` and `x out` rows are for.

**Karpathy 01:21:59 → 01:37:49 · read along; the file is already on disk.** He types four classes in this stretch and you have all four in `transformer/block_shapes.py` from Step 4, so you type nothing. Each time he adds a piece, pause and find it in the file: `MultiHeadAttention` with its `ModuleList` of heads, then `FeedFoward` (his spelling; the file keeps it), then `Block`, then the four things he goes back to add: the `proj` linear after the concat, the four-times-wider hidden layer, the residual `x = x + ...`, and the two `LayerNorm`s. For each of the four, find its row in your SHAPE TABLE too. He is training on Shakespeare and you are not, so skip his training runs and loss numbers. Stop at 01:37:49, where he starts scaling up; the Extension runs your block at two of the sizes he discusses.

**Notes**

**PyTorch enters here, as a scratch dependency only.** `uv add --dev torch tiktoken` puts both in the dev group. Graded code in `search/` and `sampling/` still imports NumPy and the standard library and nothing else. On a Mac the install is an ordinary download. On Linux or Windows the default wheel carries CUDA and runs to gigabytes; use the CPU index instead, shown in Step 2.

**Karpathy's classes read globals.** `Head` and `Block` use `n_embd` and `block_size` without being handed them. That is why `CONFIGS` sits at the top of the file and the classes below it, and why changing the config is one word on the command line rather than an edit inside a class.

**`block_size` caps `T`.** The mask is a `block_size × block_size` buffer, sliced to `T × T`. Feed a sentence longer than `block_size` and `masked_fill` fails with a `RuntimeError` saying the size of tensor a must match the size of tensor b. That is the context length, enforced by one line. The script prints a warning just before it happens; Reflection Question 3 makes it happen on purpose.

**Scale by `head_size`, not `C`.** In the video, at 01:20:05, the scaling uses `C`. His own correction in the description says `head_size`. The `Head` in the file already does it right, and says so in a comment.

**LayerNorm goes before the sublayer in this code.** `x = x + self.sa(self.ln1(x))`. The 2017 paper, and every Alammar figure, puts it after. Your table's rows are in the order the code runs them, so `ln1(x)` comes before every attention row and `ln2(x)` before every feedforward row. Annotate the order your code runs.

**Alammar draws an encoder and a decoder.** GPT-style models are decoder-only. A decoder-only block has masked self-attention and a feedforward layer, and no cross-attention. If a picture has an arrow coming in from an encoder, your table has no row for it, and that is right.

**Walkthrough — the block, run on your sentence and then annotated**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/transformer-block
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: 3B1B Ch. 5, weight tally written in the log
- [ ] Day 1: Steps 2–4, torch and tiktoken installed, sentence chosen, block_shapes.py run
- [ ] Day 1: Step 5, SHAPES.md started: table pasted, GPT-3's three numbers beside mine
- [ ] Day 2: DLAI lessons 7–8, one missing row and one missing box written down
- [ ] Day 2: Karpathy 01:21:59 → 01:37:49 read along in transformer/block_shapes.py
- [ ] Day 2: Steps 6–7, every row annotated, both + rows say what was added
- [ ] Extension: hand count committed, then run and checked
- [ ] Push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Install PyTorch and tiktoken (Day 1).**

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv add --dev torch tiktoken
uv run python -c "import torch, tiktoken; print(torch.__version__, tiktoken.get_encoding('cl100k_base').n_vocab)"
```

On Linux or Windows, use the CPU wheel instead of the first line:

```bash
uv add --dev torch tiktoken --index pytorch-cpu=https://download.pytorch.org/whl/cpu
```

*You should see* a version number starting with `2.` and then `100277`, the number of tokens `cl100k_base` knows. If `uv add` has been downloading for five minutes, you are pulling the CUDA build; stop it and use the second command.

**Step 3. Choose a sentence from your corpus.**

The block needs an input, and a tensor of random numbers tells you nothing. Your input is one sentence from `data/corpus.txt`, turned into token ids and then into vectors, the same way A08 will do it. This prints five sentences of eight to sixteen words:

```bash
uv run python -c "
import random
from pathlib import Path
text = ' '.join(Path('data/corpus.txt').read_text(encoding='utf-8').split())
sents = [s.strip() + '.' for s in text.split('. ') if 8 <= len(s.split()) <= 16]
random.seed(0)
for s in random.sample(sents, 5): print('-', s)
print(len(sents), 'sentences of 8 to 16 words')
"
```

*You should see* five sentences and a count in the hundreds or more. Change `random.seed(0)` to `1`, `2`, and so on until one of the five is a whole sentence of real text, not a heading, a page number or a line of a table of contents. On Homer, seed 0 printed `- For he was as one possessed, and was thirsting after glory.` among its five, with 926 sentences to choose from. Write your sentence in your log. A08 will want a sentence with a pronoun whose referent comes earlier in the same sentence, so if one of the five has that, take it now and you will reuse it. If your corpus is not in English, it still works; the split is on a period and a space.

<!-- JD: the Homer You-should-see values below all use the A08 sentence, "But Apollo looked down from Pergamus and called aloud to the Trojans, for he was displeased." (16 words, 23 tokens), so A07 and A08 print consistent numbers. -->

**Step 4. Create `transformer/block_shapes.py` and run it once.**

```bash
mkdir -p transformer
```

Create `transformer/block_shapes.py` (right-click `transformer`, **New File**) and paste all of this in. The four classes are Karpathy's, exactly as he has them at 01:37:49; the part below them is yours.

```python
# transformer/block_shapes.py
# Run from the repo root:  uv run python transformer/block_shapes.py
#                     or:  uv run python transformer/block_shapes.py gpt2      (the Extension)
# Builds Karpathy's decoder block (Head, MultiHeadAttention, FeedFoward, Block, as he writes
# them in the video) and pushes one sentence from your corpus through it once. Prints a
# SHAPE TABLE with one row per arrow, then every parameter tensor with its size, then the total.
import sys
import tiktoken, torch
import torch.nn as nn
from torch.nn import functional as F

SENTENCE = "PASTE YOUR SENTENCE HERE"          # one sentence from data/corpus.txt; Step 3 says how to find it

CONFIGS = {              # name: (n_embd = d, n_head = h, block_size = context length)
    "walkthrough": (64, 8, 32),
    "gpt2":        (768, 12, 1024),           # GPT-2 small, from openai-community/gpt2 config.json
    "tiny":        (32, 4, 32),               # Karpathy's width and head count in the segment; block_size raised so a sentence fits
}
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "walkthrough"
n_embd, n_head, block_size = CONFIGS[CONFIG]
torch.manual_seed(1337)

# ------------------------------------------------------------ Karpathy's classes, 01:21:59 -> 01:37:49
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q, k = self.query(x), self.key(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5      # head_size, not C: his correction in the description
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):                   # his spelling, kept
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ------------------------------------------------------------ your sentence becomes x
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode(SENTENCE)
B, T, d, h = 1, len(ids), n_embd, n_head
d_head = d // h
tok_emb = nn.Embedding(enc.n_vocab, n_embd)              # one random row per token id, as in A08
x = tok_emb(torch.tensor([ids]))                         # (B, T, d): one sentence, T tokens, d numbers each
blk = Block(n_embd, n_head)

print(f"config {CONFIG}: vocabulary {enc.n_vocab}   d {d}   context (block_size) {block_size}")
print(f"sentence: {T} tokens from {len(SENTENCE.split())} words  ->  B={B} T={T} d={d} h={h} d_head={d_head} 4d={4*d}")
if T > block_size:
    print(f"T={T} is longer than block_size={block_size}: the mask is {block_size} x {block_size} and cannot cover it. Watch what happens.")

# ------------------------------------------------------------ the arrows, in the order Block.forward runs them
rows = []
def row(name, t, symbols, sublayer):
    rows.append((name, symbols, str(tuple(t.shape)), sublayer))

with torch.no_grad():
    y = blk(x)                                           # the whole block once; the rows below retrace it one arrow at a time
    row("x in", x, "(B, T, d)", "residual stream")
    a = blk.ln1(x)
    row("ln1(x)", a, "(B, T, d)", "attention")
    h0 = blk.sa.heads[0]
    q, k, v = h0.query(a), h0.key(a), h0.value(a)
    row("q, k, v (one head)", q, "(B, T, d_head)", "attention")
    scores = q @ k.transpose(-2, -1) * h0.head_size**-0.5
    row("scores q @ k^T (one head)", scores, "(B, T, T)", "attention")
    mask = h0.tril[:T, :T]
    row("mask tril[:T, :T]", mask, "(T, T)", "attention")
    wei = F.softmax(scores.masked_fill(mask == 0, float("-inf")), dim=-1)
    row("softmax weights (one head)", wei, "(B, T, T)", "attention")
    row("wei @ v (one head out)", wei @ v, "(B, T, d_head)", "attention")
    all_scores = torch.stack([hd.query(a) @ hd.key(a).transpose(-2, -1) for hd in blk.sa.heads], dim=1)
    row("scores, all heads stacked", all_scores, "(B, h, T, T)", "attention")
    cat = torch.cat([hd(a) for hd in blk.sa.heads], dim=-1)
    row("heads concat", cat, "(B, T, h*d_head)", "attention")
    sa_out = blk.sa.proj(cat)
    row("proj", sa_out, "(B, T, d)", "attention")
    x1 = x + sa_out
    row("+ residual 1:  x + sa(ln1(x))", x1, "(B, T, d)", "residual stream")
    b = blk.ln2(x1)
    row("ln2(x)", b, "(B, T, d)", "feedforward")
    hid = blk.ffwd.net[0](b)
    row("ffwd hidden (before ReLU)", hid, "(B, T, 4d)", "feedforward")
    ff_out = blk.ffwd.net[2](F.relu(hid))
    row("ffwd out", ff_out, "(B, T, d)", "feedforward")
    x2 = x1 + ff_out
    row("+ residual 2:  x + ffwd(ln2(x))  = x out", x2, "(B, T, d)", "residual stream")
    assert torch.allclose(x2, y, atol=1e-5), "the arrows above do not add up to Block.forward"

print("\nSHAPE TABLE")
print(f"{'arrow':<42} {'shape in symbols':<18} {'shape in numbers':<18} sublayer")
for name, sym, num, sub in rows:
    print(f"{name:<42} {sym:<18} {num:<18} {sub}")

# ------------------------------------------------------------ parameters: every tensor that training would change
print("\nPARAMETERS")
total = 0
for name, p in blk.named_parameters():
    print(f"{name:<24} {str(tuple(p.shape)):<14} {p.numel():>10,}")
    total += p.numel()
print(f"{'total':<24} {'':<14} {total:>10,}   ({len(list(blk.parameters()))} tensors)")
print(f"not parameters: {h} mask buffers tril, each {tuple(h0.tril.shape)}, never trained")
```

Put your sentence from Step 3 inside the quotes on the `SENTENCE =` line, in place of `PASTE YOUR SENTENCE HERE`. Then:

```bash
uv run python transformer/block_shapes.py
```

The part below the classes does three things. `tok_emb` is a table with one random row of `d` numbers per token in `cl100k_base`, so `x` is your sentence as `T` vectors, stacked, with a batch dimension of 1 in front. The `with torch.no_grad()` block runs the block once and then retraces it one arrow at a time, in the order `Block.forward` runs them, recording the shape of each; the `assert` at the end proves the retrace lands on the same output. `named_parameters()` lists every tensor training would change.

*You should see* two header lines, a SHAPE TABLE of fifteen rows, a PARAMETERS list of 34 lines, a total, and one line about the masks. Whatever your sentence, `T` in every row is your sentence's token count, every row tagged `residual stream` has the same shape as `x in`, and the total is 49,792. On Homer, with the 23-token sentence `But Apollo looked down from Pergamus and called aloud to the Trojans, for he was displeased.`:

```text
config walkthrough: vocabulary 100277   d 64   context (block_size) 32
sentence: 23 tokens from 16 words  ->  B=1 T=23 d=64 h=8 d_head=8 4d=256

SHAPE TABLE
arrow                                      shape in symbols   shape in numbers   sublayer
x in                                       (B, T, d)          (1, 23, 64)        residual stream
ln1(x)                                     (B, T, d)          (1, 23, 64)        attention
q, k, v (one head)                         (B, T, d_head)     (1, 23, 8)         attention
scores q @ k^T (one head)                  (B, T, T)          (1, 23, 23)        attention
mask tril[:T, :T]                          (T, T)             (23, 23)           attention
softmax weights (one head)                 (B, T, T)          (1, 23, 23)        attention
wei @ v (one head out)                     (B, T, d_head)     (1, 23, 8)         attention
scores, all heads stacked                  (B, h, T, T)       (1, 8, 23, 23)     attention
heads concat                               (B, T, h*d_head)   (1, 23, 64)        attention
proj                                       (B, T, d)          (1, 23, 64)        attention
+ residual 1:  x + sa(ln1(x))              (B, T, d)          (1, 23, 64)        residual stream
ln2(x)                                     (B, T, d)          (1, 23, 64)        feedforward
ffwd hidden (before ReLU)                  (B, T, 4d)         (1, 23, 256)       feedforward
ffwd out                                   (B, T, d)          (1, 23, 64)        feedforward
+ residual 2:  x + ffwd(ln2(x))  = x out   (B, T, d)          (1, 23, 64)        residual stream

PARAMETERS
sa.heads.0.key.weight    (8, 64)               512
sa.heads.0.query.weight  (8, 64)               512
sa.heads.0.value.weight  (8, 64)               512
...                                                  (the same three lines for heads 1 to 7)
sa.proj.weight           (64, 64)            4,096
sa.proj.bias             (64,)                  64
ffwd.net.0.weight        (256, 64)          16,384
ffwd.net.0.bias          (256,)                256
ffwd.net.2.weight        (64, 256)          16,384
ffwd.net.2.bias          (64,)                  64
ln1.weight               (64,)                  64
ln1.bias                 (64,)                  64
ln2.weight               (64,)                  64
ln2.bias                 (64,)                  64
total                                       49,792   (34 tensors)
not parameters: 8 mask buffers tril, each (32, 32), never trained
```

The `...` line is mine; your output has all 24 head lines. Note the one shape the old shape prints never showed: `(B, h, T, T)`, the attention scores of every head at once. It is `T × T` per head, which is why context length costs what it costs. Paste the whole output into your log.

*If it broke:* `ModuleNotFoundError: No module named 'torch'` or `'tiktoken'` means Step 2 did not finish, or you typed `python` instead of `uv run python`. A second line reading `sentence: 6 tokens from 4 words` means `PASTE YOUR SENTENCE HERE` is still in the slot. A `KeyError` on the `CONFIGS[CONFIG]` line means the config word on the command line is misspelled; the three names are `walkthrough`, `gpt2` and `tiny`. A `RuntimeError` about tensor sizes means your sentence is longer than 32 tokens; pick a shorter one in Step 3, because Reflection Question 3 needs the walkthrough config to run clean first.

**Step 5. Start `transformer/SHAPES.md` (Day 1).**

Create `transformer/SHAPES.md` and paste this in. Fill the angle brackets you can fill today: the sentence, `T`, the pasted table, and the three GPT-3 numbers from your tally. The annotation rows wait for Day 2.

```markdown
# A07 shapes: <your name>

Sentence: <your sentence>
T = <   > tokens from <   > words. Config: walkthrough, d = 64, h = 8, d_head = 8, block_size = 32.

## Three numbers, three jobs

| | my script | GPT-3, from the 3B1B tally |
|---|---|---|
| vocabulary size | 100277 | <   > |
| d, the embedding dimension | 64 | <   > |
| context length | 32 | <   > |

## The printed table

    <paste the SHAPE TABLE exactly as printed, indented four spaces, header line included>

## One line per arrow, in my words

| arrow | what this arrow carries |
|---|---|
| x in | <   > |
| ln1(x) | <   > |
| q, k, v (one head) | <   > |
| scores q @ k^T (one head) | <   > |
| mask tril[:T, :T] | a T x T triangle of ones cut from the 32 x 32 buffer; the zeros above the diagonal are the positions each token is not allowed to see |
| softmax weights (one head) | <   > |
| wei @ v (one head out) | <   > |
| scores, all heads stacked | <   > |
| heads concat | the 8 head outputs laid side by side, 8 numbers each, back to width 64 |
| proj | <   > |
| + residual 1 | <what was added to the stream, and where it came from> |
| ln2(x) | <   > |
| ffwd hidden (before ReLU) | <   > |
| ffwd out | <   > |
| + residual 2 = x out | <what was added to the stream, and where it came from> |

## From DLAI lesson 8

One row of my table the picture has no box for: <   >
One box in the picture my table has no row for: <   >
```

Two rows are filled in as the standard: one sentence, in words, that says what the numbers in that row are. `d` is not the vocabulary size and it is not the context length; the three-number table is there so you never confuse them again.

<!-- JD: GPT-3 from the 3B1B tally: vocabulary 50,257; d 12,288; context 2,048. -->

**Step 6. Annotate every row (Day 2, after the videos).**

Fill the thirteen empty rows of the annotation table, one line each, in your own words. The test for a line: a reader who has your printed table but not the code could say, from your line alone, why the shape is what it is. `(1, 23, 8)` on `q, k, v (one head)` is not explained by "the query"; it is explained by "my 23 tokens, each shrunk from 64 numbers to 8, because `d_head = d / h = 64 / 8`". Use your own `T`.

**The two block-boundary rows have the same shape**, `x in` and `x out`. That is not a coincidence: block 2 takes what block 1 hands it, so a stack can only be built if every block returns the shape it received. Say that on the `x in` line or the `x out` line.

The two diagrams in `transformer/examples/` draw A05's probe, not this block. Compare your printed table with them anyway: they have a shape on every arrow, letters first with the real numbers beside them, and a note wherever the shape changes. That is what your table is, by construction, and what your annotations have to add is the note.

**Step 7. The two `+` rows.**

The rows tagged `residual stream` are the line running straight through the block. **Nothing ever replaces it**; two things are added to it. Write, on the `+ residual 1` line, what was added and where it came from: attention writes into each position from other positions, so on Homer the thing added at position 18, `he`, was built from positions 0 to 18, `But` through `he`. Write, on the `+ residual 2` line, what was added and where it came from: the feedforward layer sees one position at a time, so the thing added at position 18 was built from `he` alone, after attention had already written into it. Name a position in your own sentence in each line, with the token that sits there; this prints the positions:

```bash
uv run python -c "
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')
SENTENCE = 'PASTE YOUR SENTENCE HERE'
for i, t in enumerate(enc.encode(SENTENCE)): print(i, repr(enc.decode([t])))
"
```

*You should see* your sentence as one token per line, numbered from 0; on Homer, 23 lines, `0 'But'`, `1 ' Apollo'`, through `22 '.'`, with `Pergamus` split into `' P'`, `'erg'`, `'amus'`. That is the `T` in every row of your table.

**Extension — count the block before PyTorch does (CHOOSE)**

Pick one. Say which in your log, and why that one. Each option is the same script with a different config word on the command line; no code changes.

| Option | Command | Config |
|---|---|---|
| **A. GPT-2 small** | `uv run python transformer/block_shapes.py gpt2` | d 768, h 12, d_head 64, block_size 1024, from `openai-community/gpt2`'s `config.json` |
| **B. Tiny** | `uv run python transformer/block_shapes.py tiny` | d 32, h 4, d_head 8, block_size 32: Karpathy's width and head count during the segment, with the context raised so your sentence fits |
| **C. Both** | both commands | both tables, and the ratio between them |

**X1. Count by hand, commit, then run.** Do not run your option's command yet. Create `transformer/COUNT.md`, paste this in, and fill it from the config numbers in the table above and nothing else. The arithmetic is laid out; you supply the numbers and the products. Option C fills the template twice, once per config.

```markdown
# A07 hand count: <your name>, option <A / B / C>

Config <gpt2 / tiny>: d = <   >   h = <   >   d_head = d / h = <   >   4d = <   >

## X1. Hand count, before running

| sublayer | tensors | arithmetic | parameters |
|---|---|---|---|
| heads: key, query, value, no bias | h x 3 matrices, each d_head x d | <h> x 3 x <d_head> x <d> = | <   > |
| projection: weight and bias | d x d, plus d | <d> x <d> + <d> = | <   > |
| feedforward layer 1: weight and bias | 4d x d, plus 4d | <4d> x <d> + <4d> = | <   > |
| feedforward layer 2: weight and bias | d x 4d, plus d | <d> x <4d> + <d> = | <   > |
| LayerNorm 1: weight and bias | d, plus d | <d> + <d> = | <   > |
| LayerNorm 2: weight and bias | d, plus d | <d> + <d> = | <   > |
| **hand count** | | sum of the column | **<   >** |

Tensors I expect in the printed list: <   >  (3 per head, 2 for the projection, 4 for the feedforward, 4 for the two LayerNorms)
```

Check your arithmetic against the walkthrough config, where you already know the answer: at `d = 64, h = 8` the same six rows read 8 × 3 × 8 × 64 = 12,288; 4,096 + 64 = 4,160; 16,384 + 256 = 16,640; 16,384 + 64 = 16,448; 128; 128; sum 49,792, which is the total Step 4 printed, from 34 tensors. Then:

```bash
git add transformer/COUNT.md
git commit -m "A07: hand count of one block, before running it"
```

I will check your commit timestamps.

**X2. Run it.** Your option's command from the table. Paste the PARAMETERS list and the total under a `## X2. PyTorch's count` heading in `COUNT.md`, then one line: `Hand count <   > · PyTorch <   > · difference <   >`.

*You should see* the two agree, or disagree by an amount that is exactly one row of your table: a missing bias is `d` or `4d`, a missing LayerNorm is `2d`, a missing projection is `d × d + d`. Find the row by comparing your table's rows with the printed tensor lines; each printed line belongs to exactly one row. On Homer, `gpt2` printed a total of `7,085,568   (46 tensors)` and `tiny` printed `12,608   (22 tensors)`; the shape table changed in the third column only, and in no row did the `T` of 23 move.

**X3. The comparison.** One paragraph under `## X3`, by option.

*Option A.* Here is the real GPT-2 small block, one layer of the published weights (`h.0.*` in the model file), which you cannot download in this room and do not need to:

| GPT-2 tensor | shape | parameters |
|---|---|---|
| `attn.c_attn` weight and bias | 768 × 2304, plus 2304 | 1,771,776 |
| `attn.c_proj` weight and bias | 768 × 768, plus 768 | 590,592 |
| `mlp.c_fc` weight and bias | 768 × 3072, plus 3072 | 2,362,368 |
| `mlp.c_proj` weight and bias | 3072 × 768, plus 768 | 2,360,064 |
| `ln_1`, `ln_2` | 2 × (768 + 768) | 3,072 |
| **real per-block count** | | **7,087,872** |

Subtract your PyTorch total from 7,087,872 and write the gap. Then say which GPT-2 tensor has no match in your printed list, and which word in `transformer/block_shapes.py` is the reason. The gap is three times `d`, and it is one `bias=False`.

*Option B.* The attention weights are not in the parameter count; they are the `(B, h, T, T)` row, recomputed for every input. Count them at your `block_size` and at four times it:

```bash
uv run python -c "h, bs = 4, 32; print('weights per block at block_size', bs, '=', h*bs*bs, ' at', 4*bs, '=', h*(4*bs)**2, ' parameters in the block: 12608')"
```

*You should see* `4096` and `65536`. Write which of the two is more than the whole block's parameter count, and why the second number is sixteen times the first and not four.

*Option C.* Predict the ratio of the two totals from the widths before you divide: `(768 / 32)` squared. Then divide the two printed totals:

```bash
uv run python -c "print('predicted', (768/32)**2, ' actual', 7085568/12608)"
```

*You should see* `576.0` and a number just under it. Say which rows of your two hand-count tables grow with `d` rather than `d` squared; those are the reason the actual ratio falls short.

Finish `COUNT.md` with one line: where your hand count was wrong, and by how many parameters, or, if it was right, which single printed line you would most likely have missed and why.

**X4. Commit, push, PR, sign off.**

```bash
git add transformer/block_shapes.py transformer/SHAPES.md transformer/COUNT.md pyproject.toml uv.lock
git commit -m "A07: decoder block on one corpus sentence, shape table annotated, parameter count checked"
git push -u origin dev/transformer-block
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the row of your SHAPES.md annotation table you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`transformer/SHAPES.md` (the printed table, every row annotated, both `+` rows saying what was added to the residual stream, the three-number table) + `transformer/COUNT.md` (hand count committed before the run, PyTorch's count, the comparison) + `transformer/block_shapes.py`.

**Reflection Questions**

1. Paste the seven rows of your SHAPE TABLE whose shape in numbers is not the shape of `x in`. For each, name the number in it that is neither `1` nor your `T`, and give the arithmetic that produced it: `d_head = d / h`, `4d`, `h` itself, or `T` a second time. Then name the one row you would have guessed wrong on Day 1 if you had been asked to write the table before running it, and say what you would have written.

   *How to get it:* `uv run python transformer/block_shapes.py` prints the table again; the seven rows are `q, k, v (one head)`, `scores q @ k^T (one head)`, `mask tril[:T, :T]`, `softmax weights (one head)`, `wei @ v (one head out)`, `scores, all heads stacked` and `ffwd hidden (before ReLU)`. The second header line prints `d_head` and `4d` for your config, so the arithmetic is `d_head = 64 / 8 = 8` and `4d = 4 × 64 = 256`; the `T × T` rows have `T` twice because every token scores every token. The row most people would have guessed wrong is the stacked scores, `(B, h, T, T)`, because `h` sits in front and nothing else in the table has four dimensions.

2. From `COUNT.md`: your hand count, PyTorch's count, and the hash of the commit that holds the hand count. Name the row where you were wrong and by how many parameters, and say what you had forgotten. If you were exactly right, give the one printed tensor line you would most likely have missed, and why.

   *How to get it:* both counts are in your `COUNT.md` under X2. The hash is the first commit that touched the file:

   ```bash
   git log --oneline --follow -- transformer/COUNT.md | tail -1
   ```

   The tensor lines are the PARAMETERS section of your X2 run. The two lines most often forgotten are `sa.proj.bias`, because the heads have no bias and the habit carries over, and the four LayerNorm lines, because a picture draws LayerNorm as a line, not a box: on the walkthrough config that is 64 and 256 parameters.

3. Make the context length too short for your sentence: in `CONFIGS`, change the walkthrough line's `32` to a number smaller than your `T` (on Homer `T` is 23, so `16` works), run the script, and paste the last line of the error. Say which line of `Head` raised it. Then give the context length of GPT-3 from your 3B1B tally, and say how many attention weights one head computes at that length versus at half of it. Put the `32` back before you commit.

   *How to get it:* `T` is in the second line the script prints, `sentence: 23 tokens`. Change `"walkthrough": (64, 8, 32)` to `"walkthrough": (64, 8, 16)` and run `uv run python transformer/block_shapes.py`. The script prints its own warning line first, then the traceback ends with `RuntimeError: The size of tensor a (16) must match the size of tensor b (23) at non-singleton dimension 2`, with your `T` in place of 23, and the frame above it is `in forward`, pointing at the `masked_fill` line of `Head.forward`, line 38 if you pasted the file unchanged. For the last part, one head's attention matrix is `T × T` (your table printed `(1, 8, 23, 23)` for all eight heads at `T = 23`), so the count at the context length is that length squared, and at half of it a quarter as many. Afterwards `git diff transformer/block_shapes.py` should show nothing, or `git restore transformer/block_shapes.py` puts it back.
