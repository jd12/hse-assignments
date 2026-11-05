# T17 · The MLP: Three Characters In, an Embedding Table, a Hidden Layer, and Logits Out

**Meetings:** D41 · **Points:** 15 pts

**Watch — 38 min**
[Karpathy, Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) · 00:00:00 → 00:37:56, "intro", "Bengio et al. 2003 (MLP language model) paper walkthrough", "(re-)building our training dataset", "implementing the embedding lookup table", "implementing the hidden layer + internals of torch.Tensor: storage, views", "implementing the output layer", "implementing the negative log likelihood loss", "summary of the full network" and "introducing F.cross_entropy and why" (38m)
 · `scratch/t17-video.py` (new; it opens `scratch/names.txt` like T13's), `evidence/T17.md`, terminal in the repo root. The longest video of the track; the walkthrough is one script.

**During the video**

**Type everything he types, and run it.** The dataset with three characters of context, the embedding table `C`, the hidden layer, the output layer, the loss, then `F.cross_entropy`. Run the scratch file with `uv run python scratch/t17-video.py` after each cell.

**00:09:03 · compute before he does.** When he builds the dataset for the first name with a block size of 3, pause and write under `## During the video` in `evidence/T17.md` the contexts and targets for `emma`: five lines of three characters and the character that follows. Then unpause and check.

**00:12:19 → 00:29:15 · compute before he does.** Before he prints each of these shapes, pause and write it: `C[X]` for his `X`, the result of `.view(-1, 6)`, `h`, and `logits`. Then, before he counts, the parameter count for `C` 27×2, `W1` 6×100, `b1` 100, `W2` 100×27, `b2` 27. Six answers under `## During the video`.

**00:32:49 → 00:37:56.** Why `F.cross_entropy` instead of the three lines of softmax and log: the same number, faster, and no overflow when a logit is large. Write under `## During the video` the one word for what happens to `exp(100)` in the hand-written version.

**Notes**

**Three characters of context instead of one.** T14's bigram model read one character and predicted the next from a 27×27 table. Three characters of context would need a 27×27×27 table, most of it zeros on 508 names. The MLP's answer is to turn each character into a few numbers (the embedding) and let a layer of neurons read the three embeddings side by side. The table it learns is 27×2, not 27³.

**`C[X]` is the whole embedding lookup.** `X` is a tensor of integer contexts, shape `(n, 3)`; indexing `C` with it returns shape `(n, 3, 2)`, each character replaced by its row of `C`. `.view(-1, 6)` lays the three embeddings end to end so the hidden layer can read all six numbers at once. His segment on storage and views is why `.view` costs nothing: the numbers do not move.

**The network is your micrograd MLP with a lookup table in front.** `tanh(x @ W1 + b1) @ W2 + b2` is exactly `MLP(6, [100, 27])` from T11 without the final `tanh`, because the output goes to softmax instead. 3,481 parameters against T12's 37.

**Walkthrough — one forward pass on your names, every shape printed**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T16 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/mlp-names
```

*If T16 is not approved yet:* `git switch dev/net && git switch -c dev/mlp-names`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] makemore 2, 0:00 → 37:56 watched, typed along, the shapes and the count written before he printed them
- [ ] Step 2: the shapes and count for MY names in evidence/T17.md and committed, then mlp.py
- [ ] Step 3: the untrained loss read against the uniform guess
- [ ] Fill every slot in evidence/T17.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The network on your names, by hand first.**

Open `names/mlp.py` and read `build_dataset`, `make_parameters` and `forward`: his three cells as three functions. Before you run it, write under `## Step 2` in `evidence/T17.md`, by hand: how many examples your names make (every name contributes its length plus one), the contexts and targets of the first name in your `names/names.txt`, the five parameter shapes, the parameter total, and the shapes `emb`, `view`, `h`, `logits` take for that first name's examples. Commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/mlp.py
```

*You should see* the example count and the shapes of `X` and `Y`, the first name's contexts, the five parameter shapes and the total, the chain of shapes through the forward pass, the first letter's row of `C`, the untrained loss on that name, and the probability the untrained net gave each real next character. On Homer:

```text
508 names -> 4125 examples   X (4125, 3)   Y (4125,)
the contexts of 'abantes':
    ... --> a
    ..a --> b
    .ab --> a
    aba --> n
    ban --> t
    ant --> e
    nte --> s
    tes --> .
C (27, 2)   W1 (6, 100)   b1 (100,)   W2 (100, 27)   b2 (27,)
parameters: 3481
emb (8, 3, 2) -> view (8, 6) -> h (8, 100) -> logits (8, 27) -> probs (8, 27)
row of C for 'a': [-0.027, -1.101]
untrained loss on 'abantes''s 8 examples: 13.9934   (a uniform guess scores 3.2958)
probability the untrained net gave the real next character, per context: [0.0, 0.0173, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026]
```

`4125` examples is T13's bigram total: the same number, because each example is one "next character" and there is one per character of every name plus one `.`. The contexts slide: three dots, then two dots and `a`, and so on to `tes --> .`. Hold your page next to the output and mark each shape right or wrong. Paste everything under `## Step 2`.

*If it broke* with `KeyError: '<some character>'`, a name in your file has a character the T14 vocabulary does not; `mine.py` keeps only letters, so `names.txt` was edited by hand. *If it broke* with `RuntimeError: shape '[-1, 6]' is invalid`, `BLOCK * EMB` was changed in one place and not the other.

**Step 3. Read the untrained loss.**

A uniform guess, `1/27` for every character, scores `3.30`. Your untrained net scores far worse, `14` on Homer, and gives the real next character a probability of `0.0` in most positions. Random weights of size 1 through a 100-neuron layer produce logits in the tens, and softmax of logits in the tens is a spike on one character, almost always the wrong one: the untrained net is confidently wrong, which is worse than being uniformly unsure. Write under `## Step 3` your untrained loss, the uniform score, and one sentence on why random is worse than uniform here. T18 fixes both the loss and, in its extension, the confidence.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T17.md` (the questions are below) and run `uv run python scripts/check_evidence.py T17` until it says `all slots filled`. Then:

```bash
git add scratch/t17-video.py evidence/T17.md
git commit -m "T17: the MLP on my names, every shape by hand and then printed, the untrained loss"
git push -u origin dev/mlp-names
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the shape you got wrong on paper, or the one you were least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`scratch/t17-video.py` through 37:56 + `evidence/T17.md` with every slot filled, the Step 2 page committed before `mlp.py` ran.

**Reflection Questions** (answer under `## Reflection` in `evidence/T17.md`)

1. Paste your Step 2 page beside the printed shapes. Say what `C[X]` does in one sentence that uses the word "row", and why `.view(-1, 6)` is free while a copy would not be.

   *How to get it:* each integer in `X` is replaced by that row of `C`. `.view` only changes how the same numbers in memory are read, as 8 rows of 6 instead of 8 rows of 3 rows of 2; nothing is written.

2. Write the parameter count as five products and sums, then say what the count becomes if the embedding size goes from 2 to 10 and everything else stays, and which two tensors changed.

   *How to get it:* `27×2 + 6×100 + 100 + 100×27 + 27 = 3481`. At `EMB = 10`: `C` is `27×10` and `W1` is `30×100`, so `270 + 3000 + 100 + 2700 + 27 = 6097`.

3. Paste your untrained loss line and the per-context probabilities. Say why most of them print as `0.0`, in terms of what softmax does to logits that differ by tens, and what that means for the first gradient step.

   *How to get it:* `exp(30) / exp(45)` is `3e-7`, which rounds to `0.0` at four places. A probability of nearly zero on the right answer gives `-log` of nearly `15`, so the loss is enormous and the first steps are big, which T18's sweep shows.
