# T07 · The Slope of a Cosine, by Nudging and by Rule

**Meetings:** D30 · **Points:** 15 pts

**Watch — 19 min**
[Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) · 00:00:00 → 00:19:09, "intro", "micrograd overview", "derivative of a simple function with one input" and "derivative of a function with multiple inputs" (19m)
 · `evidence/T07.md` open, and a terminal in the repo root.

**During the video**

**00:08:08 · compute before he does.** His function is `3x² - 4x + 5`. When he evaluates it at `x = 3.0`, pause and write the value yourself, then the derivative by the power rule, `6x - 4`, at `3.0`, at `-3.0` and at `2/3`. Unpause: he gets each slope by nudging `x` by a small `h` and dividing. Yours by the rule and his by nudging should agree in the first few digits. Write the four numbers under `## During the video` in `evidence/T07.md`.

**00:14:12 · the same thing with three inputs.** `d = a*b + c` with `a = 2.0`, `b = -3.0`, `c = 10.0`. Before he nudges each one, write `dd/da`, `dd/db` and `dd/dc` as numbers. The rule: the derivative of `a*b` with respect to `a` is `b`, so `dd/da = -3.0`; by the same rule `dd/db = 2.0`; `c` is added straight in, so `dd/dc = 1.0`. Unpause and check each one as he nudges it. Three more numbers under `## During the video`.

Linear algebra is behind you and today starts micrograd, Karpathy's tiny engine that computes derivatives of any expression. Today he computes them by nudging; by T10 the engine computes them by rule, and the whole rest of the year runs on that.

**Notes**

**A slope is "how much the output moves per one unit of input," measured right where you are.** Nudging is the honest way to measure it: change the input by `h`, see how much the output changed, divide by `h`. The rule is the shortcut AP Calc teaches. Today's function is the cosine between two of your chapters from T06, and the input that moves is one word's count in chapter 2. Raise the count of `WORD2` in chapter 2 by one and the cosine moves by a tiny amount; that amount is the slope.

**The rule for today's slope has two pieces, and they pull in opposite directions.** The cosine is the dot product divided by the two lengths. Raising chapter 2's count of a word by 1 adds chapter 7's count of that word to the dot product, which pushes the cosine up by `b[k] / (|A| |B|)`. It also makes chapter 2's vector longer, which pushes the cosine down by `(A·B) · a[k] / (|A|³ |B|)`. `slope_by_rule` in `micrograd/slope.py` is those two pieces subtracted. A word that chapter 7 uses heavily and chapter 2 uses lightly raises the cosine; a word chapter 2 already overuses lowers it. Step 3 finds both kinds in your corpus.

**`h` has a right size.** Too large and you measure a chord, not the tangent. Too small and the two cosines you subtract are equal in every digit a float can hold, and the slope comes out as exactly zero. Step 4 shows both ends; `1e-6` is the safe middle, and it is what the script uses.

**Walkthrough — one word's count moves**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T06 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/slope
```

*If T06 is not approved yet:* `git switch dev/bands && git switch -c dev/slope`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] micrograd 00:00 → 19:09 watched, the seven derivatives computed before he nudged
- [ ] Step 2: two predictions in evidence/T07.md and committed, then slope.py run
- [ ] Step 3: the gradient, all 51 words
- [ ] Step 4: the h table
- [ ] Extension: how far the straight line stays right
- [ ] Fill every slot in evidence/T07.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. One word, one slope.**

`micrograd/slope.py` came with the template update; open it and read the top. It takes T06's top-50 vocabulary, adds your `WORD2` if it is not already in it, builds chapter 2 and chapter 7 as count vectors, and defines `f(j, t)`: the cosine between the two chapters after chapter 2's count of word `j` is raised by `t`.

Before you run anything, write two predictions under `## Step 2` in `evidence/T07.md` and commit that file: whether raising chapter 2's count of your `WORD2` by one makes the cosine go up or down, and which of the 51 words would raise it most if its count went up. Think about the two pieces in the Notes: the word has to be one chapter 7 uses more than chapter 2 does. I will check your commit timestamps. Then:

```bash
uv run python micrograd/slope.py
```

*You should see* the vocabulary size, your word's two counts, the cosine, the cosine after a raise of 1 and after a raise of 100, and the slope two ways. On Homer, `WORD2 = jove`:

```text
vocabulary: 51 words   WORD2 = 'jove' at position 50
chapter 2 count of 'jove': 58   chapter 7: 105
cosine(chapter 2, chapter 7) = 0.953744
raise chapter 2's 'jove' by 1:    0.953750
raise it by 100:                     0.953641
slope at t = 0   by nudging  0.00000564   by rule  0.00000564
```

Nudging and the rule agree to every digit printed. The slope is small because the counts are in the thousands: one more `jove` in 28,000 words barely turns the vector. Look at the two raises: by 1 the cosine goes up, as the slope says, and by 100 it goes down. The slope is right where you are and nowhere else; the Extension measures how far "where you are" extends. Paste the six lines under `## Step 2`.

*If it broke* with `AssertionError: open linear/chapter_vectors.py and put your word in WORD2`, your T01 edit is missing on this branch; merge T01 first. *If it broke* with the two slopes disagreeing in the first digit, `slope_by_rule` has been edited; `git checkout micrograd/slope.py` restores it.

**Step 3. Every word at once: the gradient.**

The 00:14:12 segment nudged three inputs one at a time. Your cosine has 51 inputs, and nudging each one gives 51 slopes. A list of every input's slope is called the gradient; it is the object the rest of micrograd is built to compute.

```bash
uv run python micrograd/slope.py partials
```

*You should see* the three words that raise the cosine most, the three that lower it most, your word's rank, and the count of positive and negative slopes. On Homer:

```text
      word ch2 count ch7 count        slope
       the      1419      1935   0.00005714
        of       703      1022   0.00003639
       his       283       503   0.00002633
         i       539       184  -0.00004858
        to       796       634  -0.00002554
       you       452       277  -0.00002509
'jove': slope  0.00000564   rank 9 of 51 (1 = raises the cosine most)
positive slopes: 18   negative: 33
```

Read the counts next to the slopes. Every word in the top three has a chapter 7 count well above its chapter 2 count; every word in the bottom three is the other way round. `i` is the most negative on Homer because chapter 2 is Odysseus telling his own story in the first person and chapter 7 is the Iliad, where nobody says `i` for pages at a time. Whatever your corpus, the sign of a word's slope is set by which chapter uses it more, and the size by how far apart the two counts are. Paste the nine lines under `## Step 3` and check your second prediction.

**Step 4. The size of `h`.**

```bash
uv run python micrograd/slope.py h
```

*You should see* five rows, the nudge slope against the rule for `h` from `0.1` down to `1e-12`. On Homer:

```text
rule slope for 'jove':  0.0000056409
      h     nudge slope       error
  1e-01    0.0000056342    6.69e-09
  1e-03    0.0000056408    6.69e-11
  1e-06    0.0000056409    4.43e-11
  1e-09    0.0000056621    2.12e-08
  1e-12    0.0000000000    5.64e-06
```

The error falls, bottoms out around `1e-3` to `1e-6`, and climbs again. At `1e-12` the nudge slope is exactly zero: the cosine after the nudge is the same float as the cosine before it, so the subtraction gives zero and no division can bring it back. That bottom row is the biggest error in the table, from the "most careful" `h`. Paste the table under `## Step 4`.

**Extension — How far does the straight line stay right?** *(assigned)*

The slope says "going straight from here, the cosine rises by `0.0000056` per count." Before you run anything, write under `## Extension` in `evidence/T07.md` the raise `t` at which you expect the straight-line prediction to be off by more than the cosine's whole gain (that is, wrong about the direction), and commit the file. I will check your commit timestamps. Then:

```bash
uv run python micrograd/slope.py far
```

*You should see* the straight line against the real cosine for raises of 1, 10, 100, 1,000 and 10,000. On Homer:

```text
cosine now 0.953744   slope 0.00000564 per count of 'jove'
     t  straight line   real cosine       error
     1       0.953750      0.953750    0.000000
    10       0.953800      0.953794    0.000007
   100       0.954308      0.953641    0.000667
  1000       0.959385      0.899351    0.060034
 10000       1.010153      0.279175    0.730978
```

At `t = 1` the line is exact. At `100` the real cosine has already turned around and gone down while the line keeps climbing. At `10,000` the line predicts a cosine above 1, which no pair of vectors can have, while the real one has collapsed to `0.28`: chapter 2 is now mostly `jove`, and points almost nowhere near chapter 7. Write where it got worse under `## Extension`: the first `t` where the line and the real cosine disagree about the direction, and the error at `t = 1000`. That is the reason every optimization step in this course is small.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T07.md` (the questions are below) and run `uv run python scripts/check_evidence.py T07` until it says `all slots filled`. Then:

```bash
git add evidence/T07.md
git commit -m "T07: slope of the cosine by nudging and by rule, the gradient over 51 words, the size of h"
git push -u origin dev/slope
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the word whose slope surprised you.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/T07.md` with every slot filled. Nothing new in `micrograd/` yet; `slope.py` is the template's, and T08 is where you start typing the engine.

**Reflection Questions** (answer under `## Reflection` in `evidence/T07.md`)

1. Paste your Step 2 lines and the two pieces of the rule from the Notes, with your own numbers in them: `b[k] / (|A| |B|)` and `(A·B) · a[k] / (|A|³ |B|)`. Say which piece won for your `WORD2`, and what about the two chapters' counts made it win.

   *How to get it:* the counts are the second line of Step 2's output. `|A|` and `|B|` are the lengths of the two chapter vectors; `uv run python -c "import sys; sys.path.insert(0, 'micrograd'); import slope; import numpy as np; print(np.sqrt(slope.vA @ slope.vA), np.sqrt(slope.vB @ slope.vB), slope.vA @ slope.vB)"` prints Step 2's lines again and then `|A|`, `|B|` and `A·B`. On Homer those are `2671.0`, `2934.8` and `7476365`, so the first piece is `105 / (2671.0 × 2934.8) = 0.0000134` and the second is `7476365 × 58 / (2671.0³ × 2934.8) = 0.0000078`; the gain from chapter 7's 105 `jove`s outweighs the stretch of chapter 2's vector, and the slope is their difference, `0.0000056`, positive.

2. Paste your Step 3 table beside your second prediction. For the most negative word, say in one sentence what about your corpus makes chapter 2 use it more than chapter 7 does. Then say what the gradient is, in one sentence, using the word "list".

   *How to get it:* the counts in the table are the evidence; the sentence about your corpus is yours. The gradient is the list of every input's slope, one number per word, 51 of them here.

3. Paste the `h` table and the `far` table. Say what went wrong at `h = 1e-12`, in terms of two floats that could not be told apart, and what went wrong at `t = 10000`, in terms of a slope that was measured at one place and used at another. The two failures are opposite: name which one is "too close" and which is "too far".

   *How to get it:* `1e-12` is too close: `f(k, h)` and `f(k, 0)` are the same float, so their difference is zero. `t = 10000` is too far: the slope was right at `t = 0` and the cosine curved away from it long before `10,000`.
