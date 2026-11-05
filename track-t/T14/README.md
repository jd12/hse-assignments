# T14 · The 27×27 Matrix of Your Names, Drawn, and the Row That Follows `u`

**Meetings:** D38 · **Points:** 15 pts

**Watch — 29 min**
[Karpathy, building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&t=1254s) · 00:20:54 → 00:50:14, "deleting spurious (S) and (E) tokens in favor of a single . token", "sampling from the model" and "efficiency! vectorized normalization of the rows, tensor broadcasting" (29m)
 · `scratch/t13-video.py` open (it keeps growing), `evidence/T14.md`, terminal in the repo root.

**During the video**

**Type everything he types, and run it.** The single `.` marker replaces `<S>` and `<E>`, the matrix becomes 27×27, each row is turned into probabilities, and a name is sampled by walking the matrix one character at a time with `torch.multinomial`.

**00:24:02 · compute before he does.** When he samples the first name from the untrained, uniform matrix and then from the real one, pause before each and write under `## During the video` in `evidence/T14.md` whether you expect anything pronounceable. Then unpause.

**00:36:17 → 00:50:14 · the broadcasting rule.** He divides the whole matrix by its row sums in one line and then, at about 00:48:00, shows what happens when `keepdim=True` is left out. **Before he explains it**, pause and write under `## During the video` what shape `N.sum(1)` has, what shape `N.sum(1, keepdim=True)` has, and which one lines up with the rows of a 27×27 matrix when the two are divided. Then unpause. The Extension does this to your matrix.

**Notes**

**One marker, not two.** `<S>` is never a second character and `<E>` is never a first, so each wasted a row or a column. With `.` for both, the matrix is 27×27 and `.` is row 0 and column 0. `N[0, 0]` is always zero: a name has at least one letter between the two markers.

**A row of the matrix is a model of "what comes next."** `N[stoi['u']]` is every count of what followed `u` in your names; divide by the row's sum and it is a probability for each next character. Sampling is reading one row, picking a character by those probabilities, and moving to that character's row. That is the whole bigram model and it is also, in miniature, what a language model does.

**Broadcasting is the rule for arithmetic between tensors of different shapes.** PyTorch lines the shapes up from the right and stretches any dimension of size 1 to match. `(27, 27) / (27, 1)` stretches the column of row sums across each row: right. `(27, 27) / (27,)` is read as `(27, 27) / (1, 27)`, a row of sums stretched down the columns: every entry is divided by the wrong row's sum, no error, and the rows no longer add up to 1. The Extension shows the symptom, which is that the columns add up to 1 instead.

**Walkthrough — the matrix of your names**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T13 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/counts
```

*If T13 is not approved yet:* `git switch dev/names && git switch -c dev/counts`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] makemore 20:54 → 50:14 watched, typed along, the two shapes written before he explained them
- [ ] Step 2: prediction in evidence/T14.md and committed, then counts.py; the picture opened
- [ ] Step 3: the row for my most common name's first letter, read
- [ ] Extension: counts.py bug, predicted first
- [ ] Fill every slot in evidence/T14.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Draw it.**

Open `names/counts.py` and read the top: the `.` marker at index 0, the 27×27 count matrix, and `row_probabilities`, which is his one-line normalization with `keepdim=True`. Before you run it, write under `## Step 2` in `evidence/T14.md` which character you expect to follow the first letter of your most common name most often (the name is the first line of `names/mined_counts.txt`), and commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/counts.py
```

*You should see* the vocabulary, the matrix's shape and sum, `drawn: names/counts.png`, then your most common name and the five likeliest characters after its first letter. On Homer:

```text
508 names   vocabulary 27: abcdefghijklmnopqrstuvwxyz and '.'
N: shape (27, 27)   sum 4125   N[0, 0] = 0 (a name cannot be empty, so . never follows .)
drawn: names/counts.png
most common name in your corpus: 'ulysses', first letter 'u', row 21
row 'u' sums to 1.0000; after 'u' the five likeliest next characters:
   's'  count  165   probability 0.757
   'r'  count   17   probability 0.078
   't'  count    8   probability 0.037
   'l'  count    6   probability 0.028
   'm'  count    5   probability 0.023
```

Open `names/counts.png`: his picture, on your names, every cell labeled with its pair and its count. Find the darkest cell. On Homer it is `s.`, 286, the Greek ending from T13, and then `us`, 165. Then find your most common name's row and read it across. On Homer `u` is followed by `s` three times out of four, which is `-us` in `ulysses`, `menelaus`, `telemachus` and a hundred others. Paste the output under `## Step 2` and check your prediction.

**Step 3. Read one row as a model.**

Under `## Step 3` in `evidence/T14.md`, write what the row you just printed predicts: if a name of your corpus has reached your most common name's first letter, what the model thinks comes next, with the top probability, and how often it would be wrong if it always guessed the top character. On Homer, `u` then `s` is right 76% of the time and wrong 24%, and when it is wrong the model has 25 other characters sharing the remaining quarter, none above 8%. Then write which row of your matrix you expect to be the most spread out (the one where the top character has the lowest probability) and check it with:

```bash
uv run python -c "import sys; sys.path.insert(0, 'names'); from counts import N, itos, row_probabilities; P = row_probabilities(N); tops = P.max(1).values; r = int(tops.argmin()); print('flattest row:', repr(itos[r]), 'top probability', round(float(tops[r]), 3), '  sharpest row:', repr(itos[int(tops.argmax())]), round(float(tops.max()), 3))"
```

*You should see* two rows named with their top probabilities. On Homer the flattest is `y`, with `c` on top at only `0.130`, and the start row `.` is close behind it with `a` at `0.156`: the first letter of a name is one of the hardest characters to guess. The sharpest is `q` at `1.000`, because the one `q` in Homer's names is followed by `u`. Paste the line under `## Step 3`. *If it broke* with the import failing, you are not in the repo root.

**Extension — The broadcasting bug, on your matrix** *(assigned)*

`counts.py bug` divides your matrix by `N.sum(1)` without `keepdim`, the way he did by accident at 48:00. Before you run it, write under `## Extension` in `evidence/T14.md` what row 1 and row 2 of the wrong matrix will add up to, and what column 1 will add up to, and commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/counts.py bug
```

*You should see* the right and wrong row sums, one column sum, and the two shapes. On Homer:

```text
right: row 1 sums to 1.0000   row 2 sums to 1.0000
wrong: row 1 sums to 2.9601   row 2 sums to 0.1541
wrong: column 1 sums to 1.0000   (the columns add to 1 instead of the rows: the giveaway)
shapes: N.sum(1, keepdim=True) is (27, 1)   N.sum(1) is (27,)
```

No error, and every row is wrong by a different amount: the entry in row `i`, column `j` was divided by row `j`'s sum instead of row `i`'s. The columns add up to 1 because in a chain of bigrams every letter is a left neighbour exactly as often as it is a right neighbour, so row `j`'s sum and column `j`'s sum are the same number. Check that on your picture: the `a` row and the `a` column have the same total. Sampling from the wrong matrix would still produce characters, just with the wrong probabilities, which is the dangerous kind of bug. Write where it got worse under `## Extension`: your row 1 and row 2 sums, why the columns add to 1 in one sentence, and the one word that fixes the line.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T14.md` (the questions are below) and run `uv run python scripts/check_evidence.py T14` until it says `all slots filled`. Then:

```bash
git add names/counts.png scratch/t13-video.py evidence/T14.md
git commit -m "T14: the 27x27 matrix of my names drawn, one row read as a model, the broadcasting bug measured"
git push -u origin dev/counts
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is the darkest cell of your picture and its count.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`names/counts.png` + `scratch/t13-video.py` through 50:14 + `evidence/T14.md` with every slot filled, both predictions committed before their runs.

**Reflection Questions** (answer under `## Reflection` in `evidence/T14.md`)

1. Paste your Step 2 row. Say what `row_probabilities` did to it in one sentence that uses the row's sum, and why `N[0, 0]` is zero in every student's matrix no matter the corpus.

   *How to get it:* each count was divided by the row's total so the row adds to 1; `.` then `.` would be a name with no letters, and `mine.py` keeps nothing shorter than three.

2. Paste the flattest-row line. Say why the start row is among the flattest on most corpora, and what a model that only knows bigrams cannot know about the third letter of a name that a model reading two characters back could.

   *How to get it:* the start row is the first-letter distribution of all your names at once, spread over most of the alphabet, because nothing comes before `.` to narrow it. A bigram model choosing the third letter of `ulysses` sees only `l`; a model reading two characters back sees `ul`, and after `ul` in Homer's names `y` is far likelier than after `l` alone. That is T17.

3. Paste your Extension lines beside your committed predictions. Say why `(27, 27) / (27,)` produced no error and the wrong answer, in terms of which side PyTorch lines shapes up from, and what single check (one line) would catch the bug in any matrix of probabilities.

   *How to get it:* shapes are lined up from the right, so `(27,)` sits under the columns and is stretched down the rows. `print(P.sum(1))` should be a column of ones; with the bug it is not, even though `P.sum(0)` is.
