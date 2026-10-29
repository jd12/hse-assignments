# P03 · TF-IDF, Chapter 2

**Meetings:** D26 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 2 TF-IDF, first half, about 25 minutes.

**During the chapter**

**Copy the formula exactly.** When the chapter gives you the IDF formula, copy it into `scratch/p03-formula.md` in the rag repo character for character, including every `+ 1`, and under it the chapter's definition of term frequency (a raw count, or a count divided by the document's length). P04 has a line in `rag/tfidf.py` marked `# CHAPTER FORMULA`, and that is where it goes.

**Notes**

**Use the formula the chapter gives you, not one you remember.** There are at least four common IDF variants, differing in where the `+ 1`s go. The grader compares floats. A different smoothing choice is not close enough.

**An inverted index is the chapter's real subject.** TF-IDF is a number; the index is what makes it fast to find every document a term is in without reading every document. Reflection 2 asks what it maps from and to.

**Walkthrough — the chapter, and a prediction for your corpus**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P02 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/tfidf
```

*If P02 is not approved yet:* `git switch dev/preprocess && git switch -c dev/tfidf`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 2, first half, every lesson run locally before it is submitted
- [ ] scratch/p03-formula.md: the chapter's idf formula and its tf, character for character
- [ ] Step 4: two predictions for my corpus, in evidence/P03.md
- [ ] Fill every slot in evidence/P03.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The formula, character for character.**

The moment the chapter states it, before the next lesson. `scratch/p03-formula.md` holds the idf formula and the tf definition as the lesson shows them. Paste both under `## Step 2` in `evidence/P03.md`.

**Step 3. Chapter 2, first half.**

In `~/version_control/bootdev-rag/`, lesson by lesson, `bootdev run <id>` before `bootdev run <id> -s`, for about twenty-five minutes. Stop where the period leaves you; P04 finishes the chapter. Save the chapter page as `evidence/p03-bootdev.png`.

**Step 4. Two predictions.**

P04 builds the index over your corpus. Before it does, write under `## Step 4` in `evidence/P03.md` the word you expect to have the highest idf in your corpus and the one you expect to have the lowest, with one sentence each. Rare means few passages, not few occurrences: a word used two hundred times in one passage is rare by idf.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P03.md` (the questions are below) and run `uv run python scripts/check_evidence.py P03` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/
git commit -m "P03: chapter 2 started, the chapter's formula copied, two predictions"
git push -u origin dev/tfidf
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the lesson you stopped on. It stays open through P04.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P03.md` with every slot filled + `scratch/p03-formula.md` + `evidence/p03-bootdev.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P03.md`)

1. In your own words, with the chapter's formula in front of you: what makes a term's idf large, and what makes it near zero?

   *How to get it:* the formula divides a count of all documents by a count of the documents containing the term, then takes a log. Put a term that is in one document into it, then a term that is in every document.

2. The chapter builds an inverted index. Say what it maps from and to, and why a search cannot afford to scan every document for every query.

   *How to get it:* the lesson that introduces it shows the data structure; its keys are one thing and its values are lists of another. For the second half, count your passages from P02 and imagine a query of three terms.

3. Paste one tf-idf value the chapter's tests checked for its dataset, with the term, the count, the document length and the df that went into it, and show the arithmetic.

   *How to get it:* `bootdev run <id>` without `-s` prints the test's input and expected output for the tf-idf lesson; the numbers are in the input, and the formula from Step 2 turns them into the expected output.
