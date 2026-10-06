# A07 · 3B1B Transformers (Ch. 5) + Transformer LLMs 07–08 + Build the Block

**Meetings:** D15–D16 · **Points:** 15 pts

**Watch — 55 min**

**Day 1 — 27 min**
[Transformers, the tech behind LLMs, 3Blue1Brown Deep Learning Ch. 5](https://www.youtube.com/watch?v=wjZofJX0v4M) (27m)
 · Nothing to type during this one; Steps 2–5 come after it.

**Day 2 — 28 min**
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 7, Architectural Overview (6m) · lesson 8, The Transformer Block (6m)
[Let's build GPT: from scratch, in code, spelled out, Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) · segment 01:21:59 → 01:37:49 (16m): multi-headed self-attention, feedforward, residual connections, layernorm
 · Your printed shape table from Day 1 open beside you, and `transformer/block_shapes.py` open in VS Code.

Reference, not assigned: [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/). Use it to check a shape you are unsure of. Its figures are encoder-decoder; see Notes.

**During the video**

**3B1B Ch. 5 · three numbers.** He gives GPT-3's vocabulary size, embedding dimension and context length as he goes. Write the three down in your log with which is which. **Three numbers, three jobs**; Step 5 puts them next to the three your script prints, and confusing them is the most common mistake on this assignment.

**DLAI lessons 7–8 · watch.** No code, nothing to write. Lesson 8's block on screen is your SHAPE TABLE as a picture; lesson 7's stack of blocks is why the `x in` and `x out` rows have to match.

**Karpathy 01:21:59 → 01:37:49 · read along; the file is already on disk.** All four classes he types are in `transformer/block_shapes.py`, so you type nothing. Each time he adds a piece, pause and find it in the file, then find its row in your SHAPE TABLE: `MultiHeadAttention`, `FeedFoward` (his spelling; the file keeps it), `Block`, then the four things he goes back to add: the `proj` linear after the concat, the four-times-wider hidden layer, the residual `x = x + ...`, and the two `LayerNorm`s. Skip his training runs and loss numbers. Stop at 01:37:49.

**Notes**

**PyTorch enters here, as a scratch dependency only.** `uv add --dev torch tiktoken` puts both in the dev group; graded code in `search/` and `sampling/` still imports NumPy and the standard library and nothing else. On Linux or Windows the default wheel carries CUDA and runs to gigabytes; use the CPU index shown in Step 2.

**`block_size` caps `T`.** The mask is a `block_size × block_size` buffer, sliced to `T × T`. A sentence longer than `block_size` makes `masked_fill` fail with a `RuntimeError` about tensor sizes. That is the context length, enforced by one line; Reflection Question 3 makes it happen on purpose. One more thing the file fixes: at 01:20:05 he scales by `C`, and his correction in the description says `head_size`; `Head` does it right and says so in a comment.

**LayerNorm goes before the sublayer in this code**, `x = x + self.sa(self.ln1(x))`; the paper and every Alammar figure put it after. Your table's rows are in the order the code runs them. Alammar also draws an encoder and a decoder; GPT-style models are decoder-only, so if a picture has an arrow coming in from an encoder, your table has no row for it, and that is right.

**Walkthrough — the block, run on your sentence and then annotated**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/transformer-block
```

Step 2 of A05b's template update, or the pull request I opened on your repo, put `block_shapes.py` in `transformer/`, and `evidence/A07.md`; `ls transformer evidence` should show them.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: 3B1B Ch. 5, GPT-3's three numbers in the log
- [ ] Day 1: Steps 2–5, torch and tiktoken installed, sentence chosen, block_shapes.py run, evidence/A07.md started
- [ ] Day 2: DLAI lessons 7–8 and Karpathy 01:21:59 → 01:37:49, read along in block_shapes.py
- [ ] Day 2: Step 6, every row annotated, both + rows say what was added
- [ ] Extension: hand count committed, then run and checked
- [ ] Fill every slot in evidence/A07.md (check_evidence.py: all slots filled); push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Install PyTorch and tiktoken (Day 1).** Run the `cd`, then one of the two `uv add` lines, the plain one on a Mac and the `--index` one on Linux or Windows (it picks the CPU build), then the check:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv add --dev torch tiktoken
uv add --dev torch tiktoken --index pytorch-cpu=https://download.pytorch.org/whl/cpu
uv run python -c "import torch, tiktoken; print(torch.__version__, tiktoken.get_encoding('cl100k_base').n_vocab)"
```

*You should see* a version number starting with `2.` and then `100277`, the number of tokens `cl100k_base` knows. If `uv add` has been downloading for five minutes, you are pulling the CUDA build; stop it and use the `--index` line.

**Step 3. Choose a sentence from your corpus.**

The block's input is one sentence from `data/corpus.txt`, turned into token ids and then into vectors, the same way A08 will do it. This prints five sentences of eight to sixteen words:

```bash
uv run python -c "
import random; from pathlib import Path
text = ' '.join(Path('data/corpus.txt').read_text(encoding='utf-8').split())   # the whole corpus as one line
sents = [s.strip() + '.' for s in text.split('. ') if 8 <= len(s.split()) <= 16]   # same as: split on '. ', keep the pieces of 8 to 16 words
random.seed(0)
for s in random.sample(sents, 5): print('-', s)
print(len(sents), 'sentences of 8 to 16 words')
"
```

*You should see* five sentences and a count in the hundreds or more. Change `random.seed(0)` to `1`, `2`, and so on until one of the five is a whole sentence of real text, not a heading or a page number. On Homer, seed 0 printed `- For he was as one possessed, and was thirsting after glory.` among its five, with 926 sentences to choose from. Write your sentence in your log. **A08 will want a sentence with a pronoun whose referent comes earlier in the same sentence**; if one of the five has that, take it now and you will reuse it.

<!-- JD: the Homer You-should-see values below all use the A08 sentence, "But Apollo looked down from Pergamus and called aloud to the Trojans, for he was displeased." (16 words, 23 tokens), so A07 and A08 print consistent numbers. -->

**Step 4. Run `transformer/block_shapes.py` on your sentence.**

Open the file. The four classes in it are Karpathy's, exactly as he has them at 01:37:49, with comments added; the part below them pushes your sentence through the block once. `Head` and `Block` are the two to read now, so here they are, with the file's longer comments cut:

```python
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)      # what each token advertises
        self.query = nn.Linear(n_embd, head_size, bias=False)    # what each token is asking for
        self.value = nn.Linear(n_embd, head_size, bias=False)    # what each token hands over
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))   # a triangle of ones; a buffer stays with the model but is never trained
        self.head_size = head_size

    # In: x of shape (B, T, n_embd). Out: (B, T, head_size), each token's mix of the values it can see.
    def forward(self, x):
        B, T, C = x.shape                                        # read the three sizes off the tensor
        q, k = self.query(x), self.key(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5     # every query against every key, one score per pair: (B, T, T); head_size, not C
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))   # later tokens get -inf, so softmax gives them weight 0
        wei = F.softmax(wei, dim=-1)                             # each row becomes weights that sum to one
        return wei @ self.value(x)                               # weighted mix of the values, (B, T, head_size)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head           # d_head = d / h; // is whole-number division
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)        # normalizes each token's vector before attention
        self.ln2 = nn.LayerNorm(n_embd)        # normalizes each token's vector before the feedforward

    # In: x of shape (B, T, n_embd). Out: the same shape. The two "x = x + ..." lines are the residual stream.
    def forward(self, x):
        x = x + self.sa(self.ln1(x))           # attention's output is added to x, never put in its place
        x = x + self.ffwd(self.ln2(x))         # the feedforward's output is added too
        return x
```

Put your sentence from Step 3 inside the quotes on the `SENTENCE =` line, in place of `PASTE YOUR SENTENCE HERE`. Then:

```bash
uv run python transformer/block_shapes.py
```

The part below the classes does three things. `tok_emb` is a table with one random row of `d` numbers per token in `cl100k_base`, so `x` is your sentence as `T` vectors with a batch dimension of 1 in front. The `with torch.no_grad()` block runs the block once, then retraces it one arrow at a time in the order `Block.forward` runs them, recording each shape; every `row(...)` line has a comment naming the line of `Head.forward` or `Block.forward` it mirrors, and the `assert` at the end proves the retrace lands on the same output. `named_parameters()` lists every tensor training would change. *You should see* two header lines, a SHAPE TABLE of fifteen rows, a PARAMETERS list of 34 lines, a total, and one line about the masks. Whatever your sentence, `T` in every row is your sentence's token count, every row tagged `residual stream` has the same shape as `x in`, and the total is 49,792. On Homer, with the 23-token sentence `But Apollo looked down from Pergamus and called aloud to the Trojans, for he was displeased.`:

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
...                                                  (the same three lines for heads 1 to 7, then sa.proj, ffwd.net and ln lines)
total                                       49,792   (34 tensors)
not parameters: 8 mask buffers tril, each (32, 32), never trained
```

The `...` line is mine; your output has all 34 lines, one per tensor, in the order `sa.heads.*`, `sa.proj.*`, `ffwd.net.*`, `ln1.*`, `ln2.*`. Note `(B, h, T, T)`, the attention scores of every head at once: `T × T` per head, which is why context length costs what it costs. Paste the whole output under `## Step 4` in `evidence/A07.md`. *If it broke:* `ModuleNotFoundError: No module named 'torch'` or `'tiktoken'` means Step 2 did not finish, or you typed `python` instead of `uv run python`. A second line reading `sentence: 6 tokens from 4 words` means `PASTE YOUR SENTENCE HERE` is still in the slot. A `KeyError` on the `CONFIGS[CONFIG]` line means the config word is misspelled; the three names are `walkthrough`, `gpt2` and `tiny`. A `RuntimeError` about tensor sizes means your sentence is longer than 32 tokens; pick a shorter one in Step 3.

**Step 5. Start the shapes section (Day 1).**

The shapes section is already in `evidence/A07.md` under `## Shapes`. Fill what you can today: the sentence, `T`, and the three GPT-3 numbers from your log, in the `<   >` cells. The annotation rows wait for Day 2; two of them are filled in as the standard, one sentence, in words, that says what the numbers in that row are.

<!-- JD: GPT-3 from 3B1B: vocabulary 50,257; d 12,288; context 2,048. -->

**Step 6. Annotate every row (Day 2, after the videos).**

Fill the thirteen empty rows of the annotation table under `## Shapes` in `evidence/A07.md`, one line each, in your own words. The test for a line: a reader who has your printed table but not the code could say, from your line alone, why the shape is what it is. `(1, 23, 8)` on `q, k, v (one head)` is not explained by "the query"; it is explained by "my 23 tokens, each shrunk from 64 numbers to 8, because `d_head = d / h = 64 / 8`". **`x in` and `x out` have the same shape** because block 2 takes what block 1 hands it; say that on one of those two lines. The two diagrams in `transformer/examples/` draw A05's probe, not this block, but they show the standard: a shape on every arrow, letters first with the real numbers beside them, and a note wherever the shape changes.

The rows tagged `residual stream` are the line running straight through the block. **Nothing ever replaces it**; two things are added to it. On the `+ residual 1` line, write what was added and where it came from: attention writes into each position from other positions, so on Homer the thing added at position 18, `he`, was built from positions 0 to 18, `But` through `he`. On the `+ residual 2` line: the feedforward layer sees one position at a time, so the thing added at position 18 was built from `he` alone, after attention had already written into it. Name a position in your own sentence in each line, with the token that sits there; this prints the positions:

```bash
uv run python -c "
import tiktoken; enc = tiktoken.get_encoding('cl100k_base')
SENTENCE = 'PASTE YOUR SENTENCE HERE'
for i, t in enumerate(enc.encode(SENTENCE)): print(i, repr(enc.decode([t])))   # enumerate numbers the tokens from 0
"
```

*You should see* one token per line, numbered from 0; on Homer, 23 lines, `0 'But'`, `1 ' Apollo'`, through `22 '.'`, with `Pergamus` split into `' P'`, `'erg'`, `'amus'`. That is the `T` in every row of your table.

**Extension — count the block before PyTorch does (CHOOSE)**

Pick one and say which on the `Option:` line under `## Prediction` in `evidence/A07.md`. Each option is the same script with a different config word on the command line; no code changes.

| Option | Command | Config |
|---|---|---|
| **A. GPT-2 small** | `uv run python transformer/block_shapes.py gpt2` | d 768, h 12, d_head 64, block_size 1024, from `openai-community/gpt2`'s `config.json` |
| **B. Tiny** | `uv run python transformer/block_shapes.py tiny` | d 32, h 4, d_head 8, block_size 32: Karpathy's width and head count during the segment |
| **C. Both** | both commands | both tables, and the ratio between them |

**X1. Count by hand, commit, then run.** Do not run your option's command yet. The hand-count table is already in `evidence/A07.md` under `## Prediction`; fill its `<   >` cells from the config numbers in the table above and nothing else, the arithmetic in numbers and then the result. Option C copies the config line and the table once more below them and fills them for the second config.

Check your arithmetic against the walkthrough config, where you know the answer: at `d = 64, h = 8` the six rows read 12,288; 4,160; 16,640; 16,448; 128; 128; sum 49,792 from 34 tensors, which is what Step 4 printed. Then:

```bash
git add evidence/A07.md
git commit -m "A07: hand count of one block, before running it"
```

I will check your commit timestamps.

**X2. Run it.** Your option's command from the table. Paste the PARAMETERS list and the total under `## Extension X2` in `evidence/A07.md`, then fill the line under it: `Hand count <   > · PyTorch <   > · difference <   >`.

*You should see* the two agree, or disagree by an amount that is exactly one row of your table: a missing bias is `d` or `4d`, a missing LayerNorm is `2d`, a missing projection is `d × d + d`. On Homer, `gpt2` printed a total of `7,085,568   (46 tensors)` and `tiny` printed `12,608   (22 tensors)`; the shape table changed in the third column only, and in no row did the `T` of 23 move.

**X3. The comparison.** The command line (or, for option A, the subtraction) and one paragraph under `## Extension X3` in `evidence/A07.md`, by option, ending with one line: where your hand count was wrong and by how many parameters, or, if it was right, which single printed line you would most likely have missed and why.

*Option A.* The real GPT-2 small block, one layer of the published weights (`h.0.*`), has 7,087,872 parameters: `attn.c_attn` 768 × 2304 plus a bias of 2304 (1,771,776), `attn.c_proj` 768 × 768 plus 768 (590,592), `mlp.c_fc` 768 × 3072 plus 3072 (2,362,368), `mlp.c_proj` 3072 × 768 plus 768 (2,360,064), and `ln_1` and `ln_2` at 768 + 768 each (3,072). Subtract your PyTorch total from 7,087,872 and write the gap. Then say which GPT-2 tensor has no match in your printed list, and which word in `transformer/block_shapes.py` is the reason. The gap is three times `d`, and it is one `bias=False`.

*Option B.* The attention weights are not in the parameter count; they are the `(B, h, T, T)` row, recomputed for every input. The first command below counts them at your `block_size` and at four times it. *You should see* `4096` and `65536`. Write which of the two is more than the whole block's parameter count, and why the second number is sixteen times the first and not four.

*Option C.* Predict the ratio of the two totals from the widths before you divide: `(768 / 32)` squared. Then the second command below divides the two printed totals. *You should see* `576.0` and a number just under it. Say which rows of your two hand-count tables grow with `d` rather than `d` squared; those are why the actual ratio falls short.

```bash
uv run python -c "h, bs = 4, 32; print('weights per block at block_size', bs, '=', h*bs*bs, ' at', 4*bs, '=', h*(4*bs)**2, ' parameters in the block: 12608')"   # option B
uv run python -c "print('predicted', (768/32)**2, ' actual', 7085568/12608)"   # option C
```

**X4. Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/A07.md` (the questions are below) and run `uv run python scripts/check_evidence.py A07` until it says `all slots filled`. Then:

```bash
git add transformer/block_shapes.py evidence/A07.md pyproject.toml uv.lock
git commit -m "A07: decoder block on one corpus sentence, shape table annotated, parameter count checked"
git push -u origin dev/transformer-block
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the row of your `## Shapes` annotation table you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/A07.md` with every slot filled (the hand count committed before the run, the printed table, every annotation row, both `+` rows saying what was added to the residual stream, the three-number table, PyTorch's count, the comparison, the three reflection answers) + `transformer/block_shapes.py` with your sentence in it.

**Reflection Questions** (answer under `## Reflection` in `evidence/A07.md`)

1. Paste the seven rows of your SHAPE TABLE whose shape in numbers is not the shape of `x in`. For each, name the number in it that is neither `1` nor your `T`, and give the arithmetic that produced it: `d_head = d / h`, `4d`, `h` itself, or `T` a second time. Then name the one row you would have guessed wrong on Day 1 if you had been asked to write the table before running it, and say what you would have written.

   *How to get it:* `uv run python transformer/block_shapes.py` prints the table again; the seven rows are `q, k, v (one head)`, `scores q @ k^T (one head)`, `mask tril[:T, :T]`, `softmax weights (one head)`, `wei @ v (one head out)`, `scores, all heads stacked` and `ffwd hidden (before ReLU)`. The second header line prints `d_head` and `4d` for your config, so the arithmetic is `d_head = 64 / 8 = 8` and `4d = 4 × 64 = 256`, and the `T × T` rows have `T` twice because every token scores every token. The row most people would have guessed wrong is the stacked scores, `(B, h, T, T)`, because `h` sits in front and nothing else in the table has four dimensions.

2. From `## Prediction` and `## Extension X2` in `evidence/A07.md`: your hand count, PyTorch's count, and the hash of the commit that holds the hand count. Name the row where you were wrong and by how many parameters, and say what you had forgotten. If you were exactly right, give the one printed tensor line you would most likely have missed, and why.

   *How to get it:* both counts are on the `Hand count` line under `## Extension X2` in `evidence/A07.md`, and the tensor lines are the PARAMETERS section of your X2 run. The two lines most often forgotten are `sa.proj.bias`, because the heads have no bias and the habit carries over, and the four LayerNorm lines, because a picture draws LayerNorm as a line, not a box. The hash is the commit whose message says hand count: `git log --oneline --grep="hand count" -- evidence/A07.md | tail -1`.

3. Make the context length too short for your sentence: in `CONFIGS`, change the walkthrough line's `"block_size": 32` to a number smaller than your `T` (on Homer `T` is 23, so `16` works), run the script, and paste the last line of the error. Say which line of `Head` raised it. Then give the context length of GPT-3 from your log, and say how many attention weights one head computes at that length versus at half of it. Put the `32` back before you commit.

   *How to get it:* `T` is in the second line the script prints, `sentence: 23 tokens`. After the change, `uv run python transformer/block_shapes.py` prints its own warning line first, then the traceback ends with `RuntimeError: The size of tensor a (16) must match the size of tensor b (23) at non-singleton dimension 2`, with your `T` in place of 23, and the frame above it is `in forward`, pointing at the `masked_fill` line of `Head.forward`, line 65 of the file. One head's attention matrix is `T × T` (your table printed `(1, 8, 23, 23)` for all eight heads at `T = 23`), so the count at the context length is that length squared, and at half of it a quarter as many. Afterwards put the `32` back; `git diff transformer/block_shapes.py` should print no line containing `block_size`.
