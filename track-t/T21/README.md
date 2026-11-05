# T21 · ★ Milestone B1: An Autograd Engine From Nothing, and the Net It Trained on Your Corpus

**Meetings:** D36 · **Points:** 15 pts

**Watch — none**
 · `micrograd/` open, `evidence/T21.md`, terminal in the repo root, and a second terminal for the clean clone in Step 5.

No video today. Everything in `micrograd/` was typed by you between T08 and T12: a `Value` class that records every operation and runs the chain rule backwards through it, a network built from it, and a hundred steps of gradient descent on forty slices of your corpus. The milestone is proving all of that works from nothing: five tests pass, a derivation page you wrote by hand agrees with the engine on a network with a hidden layer, and a stranger can clone your repo and train the net with one command.

**Notes**

**The tests say what the engine must do, in five lines each.** `micrograd/test_engine.py` came with the template: his T08 expression, a reused node, `tanh` two ways, division, and `sum`. Each test states the paper answer in its comment. A failing test names the operation whose `_backward` is wrong; that is the whole value of a test with a known answer.

**A hidden layer is where the chain rule earns its name.** T09's page went from `o` back to `w1` through one `tanh`. Today's page goes through two: the T09 neuron's output feeds a second neuron, and `do2/dw1` is a product of five factors, two of them `1 - tanh²` terms. Everything in makemore is this page with more neurons.

**A clean clone is the only honest test of "it runs."** Your working copy has files that were never committed, a `.venv` that was built over weeks, and a `data/` folder that git ignores. A fresh clone has none of that. If the net trains from a fresh clone with the three commands in Step 5, it works; if not, something you rely on never made it into git.

**Walkthrough — prove it**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T12 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/milestone
```

*If T12 is not approved yet:* `git switch dev/train && git switch -c dev/milestone`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: five tests pass
- [ ] Step 3: the derivation page, two neurons in a chain, typed in evidence/T21.md and committed
- [ ] Step 4: chain.py checks the page
- [ ] Step 5: clean clone, three commands, the net trains
- [ ] Step 6: micrograd/MILESTONE.md
- [ ] Fill every slot in evidence/T21.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Five tests.**

```bash
uv run pytest micrograd -q
```

*You should see* `5 passed`. Paste the last line under `## Step 2` in `evidence/T21.md`.

*If it broke* with one test failing, read its comment: it says the paper answer and names the operation. `test_reuse` failing is a `=` somewhere; `test_radd_for_sum` failing is a missing `__radd__`; `test_division` failing is `__pow__`'s `_backward`. Fix the engine, not the test, and run again until five pass. Write under `## Step 2` which test failed first, if any, and the line you changed.

**Step 3. The derivation page, by hand, committed before the engine checks it.**

Two neurons in a chain, on chapter 2's two features from T09:

```text
n1 = x1·w1 + x2·w2 + b1      o1 = tanh(n1)
n2 = o1·w3 + b2              o2 = tanh(n2)
```

with `w1 = 0.5`, `w2 = -0.3`, `b1 = 0.1`, `w3 = 0.8`, `b2 = -0.2`, and your `x1`, `x2` from T09 Step 2. Under `## Step 3` in `evidence/T21.md`, first the forward pass: `n1`, `o1`, `n2`, `o2` to four decimals (`o1` and `n1` are T09's `o` and `n`). Then every gradient of `o2`, from the output back, one line each with the local rule and the number: `do2/do2`, `do2/dn2`, `do2/db2`, `do2/dw3`, `do2/do1`, `do2/dn1`, `do2/db1`, `do2/dw1`, `do2/dw2`, `do2/dx1`, `do2/dx2`. Eleven numbers. The two that need care: `do2/dn2 = 1 - o2²` and `do2/dn1 = (1 - o1²) · do2/do1`, each using its own neuron's output. Then one line that writes `do2/dw1` as the product of all its factors, from `o2` to `w1`, with the numbers in. Commit the file by itself:

```bash
git add evidence/T21.md
git commit -m "T21: the derivation page, two neurons in a chain"
```

I will check your commit timestamps: the page has to exist before `chain.py` runs.

**Step 4. The engine checks the page.**

```bash
uv run python micrograd/chain.py
```

*You should see* the forward pass, the eleven gradients, and the two `tanh` factors. On Homer:

```text
x1 = -1.28   x2 = -0.36   w1 = 0.5  w2 = -0.3  b1 = 0.1   w3 = 0.8  b2 = -0.2
n1 = -0.4320   o1 = -0.4070   n2 = -0.5256   o2 = -0.4820
do2/do2 =  1.0000
do2/dn2 =  0.7677
do2/db2 =  0.7677
do2/dw3 = -0.3124
do2/do1 =  0.6141
do2/dn1 =  0.5124
do2/db1 =  0.5124
do2/dw1 = -0.6559
do2/dw2 = -0.1845
do2/dx1 =  0.2562
do2/dx2 = -0.1537
the two tanh factors: 1 - o2^2 = 0.7677   1 - o1^2 = 0.8344
```

Hold your page next to it. `do2/dw1` on Homer is `0.7677 × 0.8 × 0.8344 × (-1.28) = -0.6559`: the second `tanh`, the weight `w3`, the first `tanh`, the input `x1`. Where your page disagrees, do not change it; write the right number next to the wrong one with the rule you misapplied. Paste the output under `## Step 4`.

**Step 5. The clean clone.**

In the second terminal, somewhere outside your repo:

```bash
cd /tmp
rm -rf milestone-check
git clone https://github.com/Sierra-Canyon/hse-2026-2027-gpt-<your-username>.git milestone-check
cd milestone-check
git switch dev/milestone
```

The corpus is git-ignored, so bring it the way T01 did, then the three commands a stranger would run:

```bash
cp ~/version_control/hse-2026-2027-gpt-<your-username>/data/corpus.txt data/
uv sync
uv run pytest micrograd -q
uv run python micrograd/train.py
```

*You should see* `5 passed` and the same training output as T12, to the digit, ending at the same loss and the same misses. If the clone's numbers match your working copy's, the net is reproducible: same seed, same slices, same engine. Paste the `after training` line from the clone under `## Step 5`.

*If it broke* with `ModuleNotFoundError: No module named 'engine'` in the clone, `micrograd/engine.py` or `nn.py` was never committed; `git status` in your working copy will show it untracked. *If it broke* with the corpus missing, `data/` is ignored on purpose; the `cp` line is the fix. *If it broke* with `fatal: invalid reference: dev/milestone`, you have not pushed the branch yet; push first, then clone.

**Step 6. `micrograd/MILESTONE.md`.**

Half a page, four parts: the engine, as a table of every operation `Value` supports with its local derivative in one column (`+` passes through, `*` gives the other factor, `tanh` gives `1 - t²`, `exp` gives itself, `**k` gives `k·x^(k-1)`); the network, `MLP(2, [4, 4, 1])`, 37 parameters, what its inputs and labels are on your corpus; the result, T12's final loss and count with the misses explained in one line each; and the clean-clone commands from Step 5 that reproduce it. The derivation page stays in `evidence/T21.md`; link to it.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T21.md` (the questions are below) and run `uv run python scripts/check_evidence.py T21` until it says `all slots filled`. Then:

```bash
git add micrograd/MILESTONE.md evidence/T21.md
git commit -m "Milestone B1: engine tests, the two-neuron derivation page, clean clone, MILESTONE.md"
git push -u origin dev/milestone
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, title it **Milestone B1**, **Create pull request**, stop. The PR body is the clean-clone `after training` line.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`micrograd/` complete: `engine.py`, `nn.py`, `test_engine.py` passing, `train.py` reproducing T12 from a clean clone, `MILESTONE.md` + `evidence/T21.md` with every slot filled, the derivation page committed before `chain.py` ran. This is the engine the rest of the semester is built on; makemore starts on D37 with PyTorch doing what your engine does, at scale.

**Reflection Questions** (answer under `## Reflection` in `evidence/T21.md`)

1. Paste the `chain.py` output next to your page. Write `do2/dw1` as its product of factors with your numbers, naming the node each factor belongs to, and say which two factors would both shrink toward zero if both neurons were pushed near `±1`.

   *How to get it:* `do2/dw1 = (1 - o2²) × w3 × (1 - o1²) × x1`. The two `tanh` factors are the ones that die at `±1`; with two saturated neurons in a chain the gradient reaching `w1` is the product of two small numbers.

2. Paste the clean clone's `after training` line next to your T12 one. Say what three things have to be identical for the two runs to match to the digit, and name one change to the repo that would make them differ without any code changing.

   *How to get it:* the seed, the slices (so the corpus and the chapter cuts), and the engine. Changing `WORD2` in `chapter_vectors.py` changes every feature and every number after it, with no line of `micrograd/` touched.

3. Name the one operation whose `_backward` you would be least able to derive again from nothing, and write its local derivative from memory, then check it against `engine.py`.

   *How to get it:* close the file first. `__pow__` is the usual answer: `k · x^(k-1) · out.grad`, and the `-1` in `__truediv__` is why division works at all.
