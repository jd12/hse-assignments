# T15 · Twenty Names From the Counts, and a Number for How Good the Model Is

**Meetings:** D39 · **Points:** 15 pts

**Watch — 34 min**
[Karpathy, building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s) · 00:50:14 → 01:24:03, "loss function (the negative log likelihood of the data under our model)", "model smoothing with fake counts", "PART 2: the neural network approach: intro", "creating the bigram dataset for the neural net", "feeding integers into neural nets? one-hot encodings", "the neural net: one linear layer of neurons implemented with matrix multiplication" and "transforming neural net outputs into probabilities: the softmax" (34m)
 · `scratch/t13-video.py` open, `evidence/T15.md`, terminal in the repo root. The longest segment of the track; the walkthrough is short to match.

**During the video**

**Type everything he types, and run it.** The loss, in `scratch/t13-video.py`; then the start of the neural network, which he builds in the same notebook: the dataset of integer pairs, one-hot encoding, a 27×27 weight matrix, and the two lines of softmax.

**00:50:14 · compute before he does.** He prints the probability of each bigram in `emma` and takes the log of each. Before he prints the logs, pause and write under `## During the video` in `evidence/T15.md` whether a probability of `0.0377` has a log that is positive or negative, and whether a bigger probability gives a log closer to or further from zero. Then unpause and check.

**01:00:50 · compute before he does.** When he finds a name with a bigram the model has never seen and the loss comes out `inf`, pause and write what `log(0)` is and what adding one fake count to every cell does to it. Then unpause.

**01:18:46 → 01:24:03.** `logits.exp()` and the division by the row sums: that is `tanh` from micrograd replaced by `softmax`, a squash that makes a whole row add up to 1. Nothing to compute, but watch what the two lines do to a negative logit.

**Notes**

**The loss is the average of `-log(probability)` over every bigram of every real name.** A model that gives the real next character a high probability gets a small `-log`, near zero; a model that gives it `0.03` gets about `3.5`; a model that gives it `0` gets infinity. Averaging over all bigrams gives one number for the whole model, and lower is better. Everything from here to the GPT is trained to make this number go down, and the uniform model (every character equally likely, `1/27`) scores `log(27) = 3.30`, which is the number to beat.

**Smoothing adds fake counts.** `N + 1` gives every bigram at least one sighting, so no probability is zero and no name has infinite loss. It also flattens every row a little, so the loss on the real names goes up. The Extension measures both.

**Sampling is a random walk on the matrix.** Start at `.`, pick the next character by the `.` row's probabilities, move to that character's row, repeat until `.` comes up. The video's generator seed makes the walk repeatable: your twenty samples will be the same tomorrow, and T16's net will be sampled with the same seed so the two can be compared.

**Walkthrough — sample, then score**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T14 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/sample
```

*If T14 is not approved yet:* `git switch dev/counts && git switch -c dev/sample`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] makemore 50:14 → 1:24:03 watched, typed along, the two log questions answered before he did
- [ ] Step 2: predictions in evidence/T15.md and committed, then sample.py
- [ ] Extension: sample.py smooth, predicted first
- [ ] Fill every slot in evidence/T15.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Twenty names, and the loss.**

Open `names/sample.py` and read `sample_name` and `nll`: the random walk and the average `-log`, each a few lines. Before you run it, write two predictions under `## Step 2` in `evidence/T15.md` and commit the file: how many of twenty sampled names will be real names from your list, and what the average loss will be (between `0` and `3.30`; think about how sharp T14's rows were). I will check your commit timestamps. Then:

```bash
uv run python names/sample.py
```

*You should see* twenty samples, how many are real, the loss, the uniform model's loss, and one name's bigrams with their probabilities and `-log`s. On Homer:

```text
twenty names from the counts: cex macomphus icans an eronimithaneragaus man a s ax pus therygothimiclis cthus eptedoeuthempemes s eniavernioctlupidicidendorheus das cus neslepyns istha epidines
0 of the twenty are real names from your list
average negative log likelihood of your 508 names: 2.1902
a model that guessed every character uniformly would score 3.2958
'abantes', bigram by bigram:
   .a: probability 0.1555   -log 1.8610
   ab: probability 0.0054   -log 5.2176
   ba: probability 0.1429   -log 1.9459
   an: probability 0.2575   -log 1.3569
   nt: probability 0.0823   -log 2.4973
   te: probability 0.1737   -log 1.7505
   es: probability 0.1535   -log 1.8741
   s.: probability 0.6942   -log 0.3650
```

The samples are not names, and they are not random letters either: `macomphus`, `therygothimiclis` and `cthus` sound like something out of Homer because every pair in them is a pair Homer's names use. `a`, `s` and `an` are what a bigram model does when the start row picks a letter whose row then picks `.`; it cannot know that a one-letter name is wrong, because it never sees further back than one character. The loss, `2.19`, beats the uniform `3.30` and is a long way from `0`. In the bigram list, `ab` costs `5.2` because `b` rarely follows `a` in Homer's names, and `s.` costs `0.37` because it is the commonest pair there is. Paste everything under `## Step 2` and check your two predictions.

*If it broke* with the samples differing from run to run, the generator is not being passed to `multinomial`; `sample.py` is the template's, so `git checkout names/sample.py`. *If it broke* with `inf` for the loss, a name in your list has a bigram with zero count, which cannot happen when the counts come from the same list, so `names/names.txt` and the matrix are out of step: run `names/mine.py` again.

**Extension — Fake counts** *(assigned)*

`sample.py smooth` builds the matrix four times, adding `0`, `1`, `10` and `100` fake sightings to every cell, and prints how many bigrams have probability zero and the loss each time. Before you run it, write under `## Extension` in `evidence/T15.md` which direction the loss moves as fake counts are added and why, and how many of the 729 cells are zero in your unsmoothed matrix, and commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/sample.py smooth
```

*You should see* four rows. On Homer:

```text
fake counts   0   bigrams with probability 0: 419   loss 2.1902
fake counts   1   bigrams with probability 0:   0   loss 2.2554
fake counts  10   bigrams with probability 0:   0   loss 2.5447
fake counts 100   bigrams with probability 0:   0   loss 3.0432
```

With no smoothing, 419 of the 729 cells are zero: more than half of all possible pairs never occur in Homer's 508 names. One fake count removes every zero and costs `0.07` of loss. A hundred fake counts drown the real ones: a cell with `5` real sightings and `100` fake ones looks the same as a cell with none, and the loss climbs toward the uniform model's `3.30`. Write where it got worse under `## Extension`: the loss at `1` against `0`, the loss at `100`, and the one-sentence trade: what smoothing buys and what it costs. T17's network has a smoothing knob of its own, and the trade is the same.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T15.md` (the questions are below) and run `uv run python scripts/check_evidence.py T15` until it says `all slots filled`. Then:

```bash
git add scratch/t13-video.py evidence/T15.md
git commit -m "T15: twenty names sampled from the counts, the loss on my names, what fake counts cost"
git push -u origin dev/sample
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is the best of your twenty samples.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`scratch/t13-video.py` through 1:24:03 + `evidence/T15.md` with every slot filled, both prediction sets committed before their runs. Tomorrow is Thanksgiving break; T16 picks up on D40 with the network that learns this matrix.

**Reflection Questions** (answer under `## Reflection` in `evidence/T15.md`)

1. Paste your twenty samples beside your prediction. Pick the sample that sounds most like a name from your corpus and the one that sounds least like one, and say, using the bigram rows, why each came out as it did.

   *How to get it:* a good one is a chain of high-probability pairs; a bad one is a start row pick followed by an early `.`. `uv run python -c "import sys; sys.path.insert(0, 'names'); from counts import N, stoi, row_probabilities; P = row_probabilities(N); print(P[stoi['a'], 0].item())"` prints how often a name ends after `a` in your list.

2. Paste your loss line and `abantes`'s bigram list (or whichever name is first in your file). Say what the average loss means in words, using "per character", and why `-log` of a probability and not the probability itself is what gets averaged.

   *How to get it:* `2.19` per character is the average surprise the model feels at each real next character. Probabilities multiply along a name and logs add, so an average of logs is the same ordering as a product of probabilities, without the product underflowing to zero.

3. Paste your smoothing table beside your predictions. Say why the loss cannot go down when fake counts are added to a matrix built from the same names it is scored on, and name the one situation in which smoothing lowers the loss.

   *How to get it:* the unsmoothed matrix is the best possible bigram fit to those exact names, so any change moves it away from them. On new names it has never seen, a zero would cost infinity, and one fake count turns that into a finite loss. T18 splits the names so that situation exists.
