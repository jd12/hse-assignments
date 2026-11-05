# T13 · Mine the Names Out of Your Corpus, and Count Their Bigrams

**Meetings:** D37 · **Points:** 15 pts

**Watch — 21 min**
[Karpathy, The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) · 00:00:00 → 00:20:54, "intro", "reading and exploring the dataset", "exploring the bigrams in the dataset", "counting bigrams in a python dictionary", "counting bigrams in a 2D torch tensor" and "visualizing the bigram tensor" (21m)
 · `scratch/t13-video.py` open, `evidence/T13.md`, terminal in the repo root. Step 2 downloads his dataset; do it before you press play.

**During the video**

**Type everything he types, and run it.** He is in a notebook with `names.txt`, 32,033 names. Step 2 puts that file at `scratch/names.txt`; your scratch file opens it with `open("scratch/names.txt").read().splitlines()` and runs with `uv run python scratch/t13-video.py` from the repo root.

**00:06:24 · compute before he does.** When he lists the bigrams of the first name, `emma`, with the start and end markers, pause and write all five under `## During the video` in `evidence/T13.md` before he prints them. Then unpause and check.

**00:12:45 → 00:18:19.** The same counts in a `torch.zeros((28, 28))` tensor, indexed by `stoi`, and the picture of it. Where he displays the picture inline, you write `plt.savefig("scratch/t13-counts.png")` after the `imshow` and open the file. The picture is the point of the segment: the whole bigram model is that one square of numbers.

**Notes**

**makemore is micrograd's lesson applied to text.** A model that, given one character, says how likely each next character is; trained on a list of names; sampled one character at a time until it says "stop". The first version is nothing but counts. The second (T16) is one layer of neurons that learns the same counts by gradient descent, with PyTorch doing what your `Value` class did. The third (T17 on) is a real network.

**Your names come from your corpus, not his file.** `names/mine.py` came with the template. It walks your corpus one word at a time and keeps every word that starts with a capital letter when it is not the first word of a sentence, has only lowercase letters after that, is at least three letters long, and is seen that way at least three times. On Homer that gives 508 names: `ulysses`, `achaeans`, `trojans`, `jove`, `achilles`. It also lets in words that are capitalized mid-sentence for other reasons, `king`, `trojan`, `book`; Step 3 is where you look at those and decide what to do. His file has 32,033 names and yours has a few hundred, and that difference is the subject of T18.

**No regular expressions.** The miner is `str.split`, `str.strip`, `isupper`, `islower`, `isalpha` and a flag that remembers whether the last token ended a sentence. Read it; every line is something you could have written.

**Walkthrough — your names, counted**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T21 milestone pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/names
```

*If T21 is not approved yet:* `git switch dev/milestone && git switch -c dev/names`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: his names.txt into scratch/
- [ ] makemore 0:00 → 20:54 watched, typed along, emma's five bigrams written before he printed them
- [ ] Step 3: prediction in evidence/T13.md and committed, then mine.py; the leaks named
- [ ] Step 4: bigrams.py, the ten most common pairs on my names
- [ ] Fill every slot in evidence/T13.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. His dataset.**

```bash
curl -L -o scratch/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
wc -l scratch/names.txt
```

*You should see* `32033 scratch/names.txt`. Then watch the segment and type along.

**Step 3. Mine your names.**

Open `names/mine.py` and read the top and the loop. Before you run it, write under `## Step 3` in `evidence/T13.md` how many names you expect it to find in your corpus, and which name you expect to be the most common, and commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/mine.py
head -5 names/names.txt
head -5 names/mined_counts.txt
```

*You should see* how many capitalized mid-sentence words there were and how many survived the three-times rule, the fifteen most common, the shortest, longest and average length, and the first ten alphabetically; then the first five lines of each file. On Homer:

```text
1535 capitalized mid-sentence words, 508 seen 3 or more times -> names/names.txt
most common: ['ulysses', 'achaeans', 'trojans', 'jove', 'achilles', 'hector', 'minerva', 'telemachus', 'agamemnon', 'menelaus', 'priam', 'patroclus', 'apollo', 'atreus', 'argives']
shortest 3   longest 14 (laestrygonians)   average 7.1
first ten alphabetically: ['abantes', 'abydos', 'acamas', 'achaean', 'achaeans', 'achilles', 'actor', 'adamas', 'admetus', 'adrestus']
```

`names/names.txt` is alphabetical, one name per line, the shape of his file; `names/mined_counts.txt` is the same names most common first, with their counts. Now read down `names.txt` and find the words that are not names. On Homer: `king` (112 times, as in `King Priam`), `trojan`, `odyssey`, `book`, `sir`, and `the` (47 times, from headings and dashes). Paste the four lines under `## Step 3`, check your two predictions, and list under them every non-name you can find in your file, with a guess at why it was capitalized mid-sentence. Leave them in: the model learns the shape of your corpus's capitalized words, and T19 asks you which of its samples could pass as names, which is the better place to judge.

*If it broke* with fewer than 100 names, your corpus has few capitalized words mid-sentence (a technical text, a transcript); tell me before T14, because the count matrix gets thin. *If it broke* with `FileNotFoundError: data/corpus.txt`, you are not in the repo root.

**Step 4. Their bigrams.**

```bash
uv run python names/bigrams.py
```

*You should see* the count and lengths, the first five, the number of distinct bigrams and the total, the ten most common pairs, and the 28×28 tensor's shape and sum. On Homer:

```text
508 names   shortest 3   longest 14
first five: ['abantes', 'abydos', 'acamas', 'achaean', 'achaeans']
310 distinct bigrams, 4125 in total
the ten most common:
     s <E> 286
     u s   165
     a n   98
   <S> a   79
   <S> p   75
     n <E> 74
     e s   68
     t h   67
     h e   58
     o n   58
N: shape (28, 28)   sum 4125   (the sum equals the dictionary's total)
row <S>: starts with the most common first letter 'a' (79 names)
```

On Homer the top pair by a wide margin is `s` then `<E>`: 286 of 508 names end in `s`, because Butler kept the Greek endings, and `u s` is next for the same reason. His names.txt's top pair is `n <E>`. Your top pairs say what your corpus's names sound like. Paste the output under `## Step 4`.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T13.md` (the questions are below) and run `uv run python scripts/check_evidence.py T13` until it says `all slots filled`. Then:

```bash
git add names/names.txt names/mined_counts.txt scratch/t13-video.py scratch/names.txt scratch/t13-counts.png evidence/T13.md
git commit -m "T13: names mined from the corpus, their bigrams counted in a dict and a tensor"
git push -u origin dev/names
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the non-name in your list that surprised you.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`names/names.txt` and `names/mined_counts.txt` + `scratch/t13-video.py` with `scratch/names.txt` and `scratch/t13-counts.png` + `evidence/T13.md` with every slot filled, the Step 3 predictions committed before `mine.py` ran.

**Reflection Questions** (answer under `## Reflection` in `evidence/T13.md`)

1. Paste your Step 3 lines beside your predictions. List the non-names you found and say which single change to `mine.py`'s rule would remove the most of them, and what real names that change would also throw away.

   *How to get it:* `king` goes if words that precede another capitalized word are skipped, but so does `agamemnon` in `Agamemnon King of men`. Raising `MIN_COUNT` to 10 drops `book` and `sir` and also every minor character named three to nine times. Every rule costs something; say what yours costs.

2. Paste your ten most common bigrams. Say what the top one tells you about your corpus's names, and compare it with his top pair, `n <E>`, in one sentence about what kind of names each list holds.

   *How to get it:* on Homer `s <E>` is Greek endings in an English translation. His is American first names, which end in `n` (`ethan`, `logan`, `aiden`) more than anything else.

3. Explain, in two sentences a classmate could follow, why a 28×28 tensor and a dictionary of pairs hold exactly the same information, and name one thing the tensor can do in one line that the dictionary cannot.

   *How to get it:* every pair of characters is one cell of the tensor and one key of the dictionary, and both store the same count. The tensor can give a whole row at once, `N[stoi['a']]`, which is everything that follows `a`, and T14 turns those rows into probabilities with one division.
