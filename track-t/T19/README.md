# T19 · Where the Net Put Each Letter, and Five Names That Could Pass

**Meetings:** D43 · **Points:** 15 pts

**Watch — none**
 · `names/embeddings.png` once it exists, `evidence/T19.md`, `names/SAMPLES.md` (new), terminal in the repo root.

No video. Thursday's segment ended with two things you typed and did not have time to read: the picture of the embedding table, and sampling from the trained MLP. Today is both of those on your names, and a judgment call: of twenty names the model makes up, which five could pass as names from your corpus, and why.

**Notes**

**The embedding table is 27 points on a plane.** `C` is 27×2, so each character is a point, and the network put the points where they help it predict. In his picture the vowels cluster and the rare letters drift to the edge. On 508 names the picture is noisier; on Homer four vowels sit together and `u` is off on its own, because in Homer's names `u` is mostly the `u` of `-us`, which predicts `s` the way a consonant would. Your picture is yours to read.

**Sampling from the MLP is T15's random walk with a longer memory.** Start from three dots, run the forward pass, pick the next character by the softmax probabilities, slide the window, repeat until `.`. The difference from T15 is that the probabilities depend on three characters, not one, so `.ul` can know `y` is coming.

**"Could pass" is a claim about your corpus, not about English.** A sample passes if a reader of your corpus would not blink at it among the real names. On Homer, `atus`, `leus` and `lyodus` pass; `fepesastilasgius` does not, and `mon` is a word. Say why for each of your five, using the bigram and trigram patterns your corpus's names have.

**Walkthrough — the picture, the samples, the five**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T18 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/embed
```

*If T18 is not approved yet:* `git switch dev/splits && git switch -c dev/embed`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: predictions in evidence/T19.md and committed, then embed_plot.py; the picture read
- [ ] Step 3: the twenty samples judged, five chosen, names/SAMPLES.md
- [ ] Extension: the same twenty, before and after training
- [ ] Fill every slot in evidence/T19.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The picture.**

Open `names/embed_plot.py` and read the top. Before you run it, write two predictions under `## Step 2` in `evidence/T19.md` and commit the file: whether your five vowels will sit closer together than letters in general, and which letter you expect to be farthest from all the others. I will check your commit timestamps. Then:

```bash
uv run python names/embed_plot.py
```

*You should see* `drawn: names/embeddings.png`, two distance lines, twenty samples, and how many are real. On Homer:

```text
drawn: names/embeddings.png
average distance between two vowels 2.122   between any two letters 1.246
the vowel farthest from the others: 'u', 3.829 from their middle; the other four average 0.344 apart
twenty names from the MLP: chrean atus leus imrous hilaspans neomacus fepesastilasgius lociatceriius slaciss lyons mechibus slinean adcius pelocianrysixippus iuretus depo atetus hys lyodus mon
0 of the twenty are real names from your list
```

Open `names/embeddings.png`. On Homer `a`, `o`, `e`, `i` and `y` sit in a tight group at the lower left, `0.34` apart on average, and `u` is at the top of the picture, `3.8` away from them, with `l` and `g` on the way. The first distance line says the vowels are *farther* apart than average, which is true only because `u` is counted as one of them; the second line is the picture as it really is: four vowels together and one outlier with a reason. Paste the five lines under `## Step 2`, check your two predictions, and write what your picture shows: which letters group, which one is farthest out, and one sentence on why that letter behaves differently in your corpus's names.

*If it broke* with `FileNotFoundError: names/mlp.pt`, T18's training did not finish or was not committed; run `uv run python names/train_mlp.py` first.

**Step 3. Twenty samples, five that pass.**

Read the twenty. For each one, decide: a name your corpus could contain, a word rather than a name, or noise. Then create `names/SAMPLES.md` with the twenty listed under those three headings, and under a fourth heading the five best, each with one line on what makes it pass: the ending, a familiar three-letter run, a real name it is one letter from. On Homer, `atus`, `leus`, `lyodus`, `iuretus` and `adcius` pass on their endings alone, `mon` and `depo` are words, and `fepesastilasgius` and `pelocianrysixippus` are three names run together. Paste the five with their reasons under `## Step 3`.

**Extension — The same twenty, before and after training** *(assigned)*

The untrained network, sampled with the same seed, gives twenty samples too, with one safety rail: a sample is cut off at 40 characters. Before you run it, write under `## Extension` in `evidence/T19.md` what you expect the untrained samples to look like (length, letters) and whether you expect the cut-off to matter, and commit the file. I will check your commit timestamps. Then:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'names')
import torch, torch.nn.functional as F
from mlp import make_parameters, forward, BLOCK, SEED
from counts import itos
parameters = make_parameters(torch.Generator().manual_seed(SEED))
g = torch.Generator().manual_seed(SEED + 10)
for _ in range(20):
    out, context = [], [0] * BLOCK
    while len(out) < 40:
        ix = int(torch.multinomial(F.softmax(forward(parameters, torch.tensor([context])), dim=1), 1, generator=g).item())
        if ix == 0: break
        out.append(itos[ix]); context = context[1:] + [ix]
    print(''.join(out), len(out))
"
```

*You should see* twenty strings and their lengths. On Homer every one of them is 40 long, cut off by the rail, and they read `jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj` and `ijjjjjjjjjjjjjjjiijjjjjjjjjjjiiijjjjjjjj`: the untrained net is a spike on one character per context, as T17's probability line showed, so once it says `j` after `jjj` it says `j` after `jjj` forever and never reaches `.`. Without the rail the first sample would never end. Write where it got worse under `## Extension`: how many of the twenty hit the rail, the letters they are made of, and the one sentence that explains the repetition from T17's probability line.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T19.md` (the questions are below) and run `uv run python scripts/check_evidence.py T19` until it says `all slots filled`. Then:

```bash
git add names/embeddings.png names/SAMPLES.md evidence/T19.md
git commit -m "T19: the embedding picture read, twenty samples judged, five that pass"
git push -u origin dev/embed
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is your best sample.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`names/embeddings.png` + `names/SAMPLES.md` (twenty sorted, five chosen with reasons) + `evidence/T19.md` with every slot filled, both prediction sets committed before their runs.

**Reflection Questions** (answer under `## Reflection` in `evidence/T19.md`)

1. Paste the two distance lines beside your predictions. Say what it means for the network that two letters sit close together in `C`, in terms of what the hidden layer sees, and name a pair of letters your corpus's names use interchangeably that the picture confirms or denies.

   *How to get it:* close points give the hidden layer nearly the same six numbers, so the network treats the two letters alike. On Homer `a` and `o` are nearly on top of each other, and Butler's names end in `-aus` and `-ous` both.

2. Paste your five with their reasons. For the best one, write the three-character contexts the network walked through to make it, and say which of those contexts appears in a real name of your corpus.

   *How to get it:* `lyodus` is `...` → `l`, `..l` → `y`, `.ly` → `o`, `lyo` → `d`, `yod` → `u`, `odu` → `s`, `dus` → `.`; `.ly` is `lycaon` and `dus` is `odysseus`'s tail without the `e`.

3. Paste the first three untrained samples beside your prediction. Say what training did to the network's probabilities, in one sentence that uses the word "spread", and what the bigram model of T15 would have been better at than the untrained MLP.

   *How to get it:* training spread each context's probability from a spike on one wrong character across the few right ones. The bigram counts, with no training at all, already knew which characters follow which; the untrained MLP knew nothing and was certain anyway.
