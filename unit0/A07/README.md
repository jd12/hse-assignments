# A07 · 3B1B Transformers (Ch. 5) + Transformer LLMs 07–08 + Build the Block

**Meetings:** D15–D16 · **Points:** 15 pts

**Watch — 55 min**

**Day 1 — 27 min**
[Transformers, the tech behind LLMs, 3Blue1Brown Deep Learning Ch. 5](https://www.youtube.com/watch?v=wjZofJX0v4M) (27m)
 · Paper and a pencil beside you. Nothing to type today.

**Day 2 — 28 min**
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 7, Architectural Overview (6m) · lesson 8, The Transformer Block (6m)
[Let's build GPT: from scratch, in code, spelled out, Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) · segment 01:21:59 → 01:37:49 (16m): multi-headed self-attention, feedforward, residual connections, layernorm
 · Your Day 1 drawing open beside you, and `scratch/a07-video.py` open in VS Code with Step 5 already done.

Reference, not assigned: [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/). Use it to check a shape you are unsure of. Its figures are encoder-decoder; see Notes.

**During the video**

**3B1B Ch. 5 · write down the tally.** He counts GPT-3's weights as he goes, matrix by matrix, keeping a running total. Each time he adds a matrix to the tally, pause and write its name and its shape in your log, with his numbers. When he finishes, go back over your list and mark which number is the vocabulary size, which is the embedding dimension, and which is the context length. Three numbers, three jobs, and Step 3 needs all three.

**DLAI lessons 7–8 · compare, do not copy.** No code. When lesson 8 puts the block on screen, pause and write down one component in it that is missing from your Day 1 drawing, and one arrow in your drawing that it does not show.

**Karpathy 01:21:59 → 01:37:49 · type everything he types, and run it.** He pastes nothing in this stretch, so neither do you. Type into `scratch/a07-video.py`, which Step 5 has already set up with a `Head` class and the constants his code expects. He writes `MultiHeadAttention`, then `FeedFoward` (his spelling; keep it or fix it, but be consistent), then `Block`, then goes back to add the projection, the four-times-wider hidden layer, the residual `x = x + ...`, and the two `LayerNorm`s. Follow each change as he makes it. He is training on Shakespeare and you are not, so skip his training runs and loss numbers. Stop at 01:37:49, where he starts scaling up.

**Notes**

**PyTorch enters here, as a scratch dependency only.** `uv add --dev torch` puts it in the dev group. Graded code in `search/` and `sampling/` still imports NumPy and the standard library and nothing else. On a Mac the install is an ordinary download. On Linux or Windows the default wheel carries CUDA and runs to gigabytes; use the CPU index instead, shown in Step 4.

**Karpathy's classes read globals.** `Head` and `Block` use `n_embd` and `block_size` without being handed them. Type a class before those constants exist and the first run dies with `NameError: name 'n_embd' is not defined`. The constants go at the top of the file.

**`block_size` caps `T`.** The mask is a `block_size × block_size` buffer, sliced to `T × T`. Feed a sequence longer than `block_size` and `masked_fill` fails with a `RuntimeError` saying the size of tensor a must match the size of tensor b. That is the context length, enforced by one line.

**Scale by `head_size`, not `C`.** In the video, at 01:20:05, the scaling uses `C`. His own correction in the description says `head_size`. The `Head` in Step 5 already does it right.

**LayerNorm goes before the sublayer in this code.** `x = x + self.sa(self.ln1(x))`. The 2017 paper, and every Alammar figure, puts it after. Draw the one your code runs.

**Alammar draws an encoder and a decoder.** GPT-style models are decoder-only. A decoder-only block has masked self-attention and a feedforward layer, and no cross-attention. If your diagram has an arrow coming in from an encoder, it is a diagram of a model you do not use.

**Walkthrough — the block, drawn and then run**

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
- [ ] Day 1: Steps 2–3, block boundary and every arrow labeled
- [ ] Day 2: DLAI lessons 7–8, one missing component written down
- [ ] Day 2: Steps 4–5, torch installed, Head typed
- [ ] Day 2: Karpathy 01:21:59 → 01:37:49 typed along in scratch/a07-video.py
- [ ] Day 2: Steps 6–8, shapes printed, diagram corrected, photographed
- [ ] Extension: hand count committed, then checked
- [ ] Push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Draw the boundary first (Day 1).**

On paper, one box: a single decoder block. One arrow in at the bottom, one arrow out at the top. Label both with a shape in terms of `B` (batch), `T` (sequence length) and `d` (model dimension).

*You should see*, when you are done, the same shape on both arrows. If they differ, the stack cannot be built, because block 2 takes what block 1 hands it.

**Step 3. Fill the box, and label every arrow.**

Inside the box: the masked multi-head attention sublayer, the feedforward sublayer, the two residual additions, and the two layer norms. Every arrow gets a shape in `B`, `T`, `d`, `h` (heads) and `d_head`. The two places people go quiet are the split into heads and the concatenation coming back out; label both.

`d` is not the vocabulary size and it is not the context length. Your 3B1B tally has all three for GPT-3. Write them in the corner of the page.

Then draw the residual stream as a line running straight up the side of the box, and at each of the two `+` signs write what has been added to it. Attention writes into it from other positions. The feedforward layer writes into it from the token itself. Nothing ever replaces it.

**Step 4. Install PyTorch (Day 2).**

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv add --dev torch
uv run python -c "import torch; print(torch.__version__)"
```

On Linux or Windows, use the CPU wheel instead of the first line:

```bash
uv add --dev torch --index pytorch-cpu=https://download.pytorch.org/whl/cpu
```

*You should see* a version number starting with `2.`. If `uv add` has been downloading for five minutes, you are pulling the CUDA build; stop it and use the second command.

**Step 5. Set up the scratch file before the video.**

Create `scratch/a07-video.py`. `Head` is from the four minutes just before today's segment; A08 is the assignment where you type it from the video. Type it from here now:

```python
# scratch/a07-video.py
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B, T = 2, 7
n_embd, n_head = 64, 8
block_size = 16

class Head(nn.Module):
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
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)
```

Now play the Karpathy segment and type `MultiHeadAttention`, `FeedFoward` and `Block` below this, as he does.

**Step 6. Print every shape you labeled.**

When the segment ends, add this at the bottom:

```python
x = torch.randn(B, T, n_embd)
blk = Block(n_embd, n_head)
a = blk.ln1(x)
print("x in        ", tuple(x.shape))
print("one head    ", tuple(blk.sa.heads[0](a).shape))
print("heads concat", tuple(torch.cat([h(a) for h in blk.sa.heads], dim=-1).shape))
print("ffwd hidden ", tuple(blk.ffwd.net[0](blk.ln2(x)).shape))
print("x out       ", tuple(blk(x).shape))
print("params      ", sum(p.numel() for p in blk.parameters()))
```

```bash
uv run python scratch/a07-video.py
```

*You should see*, with these constants:

```
x in         (2, 7, 64)
one head     (2, 7, 8)
heads concat (2, 7, 64)
ffwd hidden  (2, 7, 256)
x out        (2, 7, 64)
params       49792
```

*If it broke:* `AttributeError: 'Block' object has no attribute 'ln1'` means you stopped before 01:32:51; finish the segment. `'FeedFoward' object has no attribute 'net'` means you wrote the layers without his `nn.Sequential`. A `params` number other than 49,792 means a missing projection, a missing bias, or a hidden layer that is not four times wide: print `[(n, p.numel()) for n, p in blk.named_parameters()]` and find it.

**Step 7. The one shape PyTorch did not print.**

`Head` returns the output of attention, not the weights, so the `T × T` matrix never appeared above. Check it with NumPy:

```bash
uv run python -c "
import numpy as np
B,T,d,h = 2,7,64,8
x = np.random.randn(B,T,d)
q = x.reshape(B,T,h,d//h).transpose(0,2,1,3)
print('x', x.shape, '-> heads', q.shape)
scores = q @ q.transpose(0,1,3,2)
print('scores', scores.shape)
"
```

*You should see* `x (2, 7, 64) -> heads (2, 8, 7, 8)` and `scores (2, 8, 7, 7)`. The attention matrix is `T × T` per head, which is why context length costs what it costs.

**Step 8. Correct the diagram, then save it.**

Go arrow by arrow against Steps 6 and 7. Where a label was wrong, cross it out and write the right one beside it rather than redrawing; the crossing-out is evidence and question 1 asks about it. Photograph it in decent light and check it is readable at 100% zoom, because I read it at 100% zoom. Save it as `transformer/block.png`.

**Extension — count the block before PyTorch does (CHOOSE)**

Pick one. Say which in your log, and why that one.

| Option | Config | What you draw |
|---|---|---|
| **A. A real model** | Look up a decoder-only model's `config.json` on Hugging Face (GPT-2 is `openai-community/gpt2`). Write down its width, heads, layers, context length and vocabulary, and the URL. | A second copy of your diagram with its real numbers on every arrow. |
| **B. Your own tiny config** | Choose `n_embd`, `n_head` and `block_size` yourself. `n_embd` must divide by `n_head`, and it may not be 64. | The same diagram with your numbers, checked by re-running Step 7 with them. |
| **C. Two sizes of one family** | Two configs from the same family, small and larger. | One table of every arrow's shape for both, with the labels that did not change marked. |

**X1. Count by hand, commit, then run.** From your diagram alone, count the parameters in one block for your chosen config: every weight matrix and every bias, sublayer by sublayer, in a table in `transformer/COUNT.md`. Do not run anything yet.

```bash
git add transformer/COUNT.md
git commit -m "A07: hand count of one block, before running it"
```

I will check your commit timestamps.

**X2. Run it.** Change the constants in `scratch/a07-video.py` to your config (for A and C, set `T` small, like 8, so it runs fast) and run it. Put PyTorch's `params` next to your hand count.

*You should see* the two agree, or disagree by an amount you can trace to one sublayer with `named_parameters()`.

**X3. The comparison.** Option A: find the model's real per-block count (sum the tensors named for one layer in its published weights, or compute it from its architecture) and compare it with Karpathy's `Block` on the same numbers. They are not equal, and the gap has an exact explanation. Option B: count the attention weights, `h × T × T`, at your `block_size` and at four times it. Option C: the ratio of the two per-block counts, predicted from the widths before you computed it, then actual.

Finish `COUNT.md` with where your hand count was wrong, and by how many parameters.

**X4. Commit, push, PR, sign off.**

```bash
git add transformer/block.png transformer/COUNT.md scratch/a07-video.py pyproject.toml uv.lock
git commit -m "A07: decoder block with shapes, typed Block, parameter count checked"
git push -u origin dev/transformer-block
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the arrow on your diagram you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`transformer/block.png` (decoder-only block, every arrow shaped, residual stream traced) + `transformer/COUNT.md` + `scratch/a07-video.py`.

**Reflection Questions**

1. Paste the six lines Step 6 printed. Then, from the photo in `transformer/block.png`, name every arrow you crossed out in Step 8 and what you had written there first. If you crossed out nothing, name the arrow you were least sure of on Day 1 and say which printed line settled it.

   *How to get it:* `uv run python scratch/a07-video.py` prints the six lines again. The crossed-out arrows are on your own photo; each one matches one printed line (`one head` is the split, `heads concat` the join, `ffwd hidden` the wide layer, `x in` and `x out` the two block boundaries).

2. From `COUNT.md`: your hand count, PyTorch's count, and the hash of the commit that holds the hand count. Name the sublayer where you were wrong and by how many parameters, and say what you had forgotten. If you were exactly right, give the one line of `named_parameters()` output you would most likely have got wrong, and why.

   *How to get it:* both counts are in your `COUNT.md` table from X2. The hash is the first commit that touched the file:

   ```bash
   git log --oneline --follow -- transformer/COUNT.md | tail -1
   ```

   To see every tensor with its shape and size, which is where a missing bias or projection shows up:

   ```bash
   uv run python -c "
   import sys; sys.argv = ['x']
   exec(open('scratch/a07-video.py').read().split('x = torch.randn')[0])
   for n, p in Block(n_embd, n_head).named_parameters(): print(f'{n:28} {tuple(p.shape)!s:12} {p.numel()}')
   "
   ```

   *You should see* 34 lines (24 head matrices, the projection weight and bias, four feedforward tensors, four layer-norm tensors) that sum to your `params` line.

3. Set `T` to one more than `block_size` in `scratch/a07-video.py`, run it, and paste the last line of the error. Say which line of `Head` raised it. Then give the context length of the model in your 3B1B tally (or your extension config), and say how many attention weights one head computes at that length versus at half of it.

   *How to get it:* `block_size` is set below `B, T` in the file, so write the number in, not the name: change `B, T = 2, 7` to `B, T = 2, 17` and run `uv run python scratch/a07-video.py`. The traceback ends with a `RuntimeError` about tensor sizes, and the line above it names the `masked_fill` line in `Head.forward`. Put `T` back to 7 afterwards. For the last part, one head's attention matrix is `T × T` (Step 7 printed `(2, 8, 7, 7)` for `T = 7`), so the count at the context length is that length squared.
