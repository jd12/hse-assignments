# A05b · Your Corpus for the Year, and the Tests It Has to Pass

**Meetings:** D13 · **Points:** 5 pts

**Watch** — none

No video today. This is thirty minutes of choosing one text file and proving it will hold up, and everything after it depends on the file being right. A06 starts from it this same period.

**Notes**

From today until May, every assignment in this course runs on one text file that is yours: `data/corpus.txt`. A06 searches it. A09 samples prompts from it. A10 asks a model questions whose answers are in it and catches the model making things up. Track A builds a retrieval system over it and measures recall against a golden set drawn from it. Track B computes its entropy, fits lines to it, and in the spring trains a character-level GPT on it and reads what comes out. Every number you report this year is a number about this file, which is why no two students in the room can hand in the same answer.

So the file has to be good, and "good" is specific. Big enough that a search over it has something to find and a model trained on it produces readable text: **one million characters is the floor**, and the checker fails below it. Plain UTF-8 text with paragraphs separated by blank lines, because that is what the chunker in A06 splits on. Free of the license header and footer a download comes wrapped in, because otherwise "Project Gutenberg" is the top hit for a third of your queries. Not mostly repeats. Something you can legally use and are willing to have excerpts of pasted into committed files all year, which rules out anything private and anything you scraped from behind a login. And re-fetchable by a script, because I re-run your numbers, and a corpus nobody can reproduce is a result nobody can check.

`Choosing a Corpus.md`, linked from the repo README, has sources for literature, history, sports, technology, science, games, law and food, each with the command that fetches it. The worked example throughout this course is Homer, because it is what I used: the Odyssey and the Iliad in Butler's translation, Project Gutenberg #1727 and #2199, glued into one file of about 1.5 million characters. The Odyssey alone is 700,000, under the floor, which is the first lesson of the day: one book is usually not enough, and the fix is a second book from the same source, fetched by the same script.

**The checker is a script and a test file, and they check the same things.** `scripts/check_corpus.py` prints a report with a fix under every failure. `tests/test_corpus.py` is the same set of facts as pytest tests, one per thing a later assignment assumes, and the last test in it is empty because it is yours to write.

One thing the checker will get wrong on purpose and you should know about. It warns if the text does not look like English prose, because A03 through A09 use English examples; a corpus of chess games or Node documentation is allowed and is a good choice, and the warning is there so you write down what you expect to be different, not so you change your mind. Size is not like that. Under a million characters is a FAIL, today, because Track B trains a model on this file in the spring and nothing under a million produces readable samples, and I would rather every one of you hit that wall now, with the fetch script open, than in February. Everything on today's list is required today: the fetch script, the checker at 0 FAIL, `SOURCE.md`, and your own test.

**Walkthrough — lock the corpus in**

**Step 1. Merge, branch, log.**

Open `hse-2026-2027-foundations-<your-username>` on GitHub and merge the A05 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/corpus
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-unit0
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Pick a corpus from the guide, or justify one that is not in it
- [ ] Write scripts/fetch_corpus.sh and run it
- [ ] Run scripts/check_corpus.py until it reports 0 FAIL
- [ ] Write data/SOURCE.md with the sha256
- [ ] Run pytest; write my own test
- [ ] Push, open the PR, sign off the log
```

**Step 2. Pull the checker into your repo.**

The template shipped without it. Pull it now:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git fetch template 2>/dev/null || git remote add template https://github.com/Sierra-Canyon/foundations-template.git
git fetch template
git checkout template/main -- scripts/check_corpus.py tests/test_corpus.py
uv add --dev pytest
ls scripts tests
```

*You should see* `check_corpus.py` under `scripts/` and `test_corpus.py` under `tests/`.

*If it broke* with `fatal: couldn't find remote ref main`, the template's default branch is `master` on your clone; use `template/master` in the checkout line.

**Step 3. Choose, then write the fetch script before you download anything.**

Open `Choosing a Corpus.md` and pick. Then create `scripts/fetch_corpus.sh` from this shape, with your own IDs or URLs and your own stripping rule:

```bash
#!/bin/bash
# Re-creates data/corpus.txt from scratch. Run from the repo root.
set -e
mkdir -p data
: > data/corpus.txt                      # start empty; every book below is appended
for ID in 1727 2199; do                  # 1727 = the Odyssey, 2199 = the Iliad (Butler). Change these.
  curl -fsSL --retry 4 --retry-delay 5 --retry-all-errors "https://www.gutenberg.org/cache/epub/$ID/pg$ID.txt" -o data/raw.txt
  # Keep only what is between the START and END markers, then drop the marker lines themselves.
  awk '/\*\*\* START OF/{flag=1; next} /\*\*\* END OF/{flag=0} flag' data/raw.txt >> data/corpus.txt
  printf "\n\n" >> data/corpus.txt
done
rm data/raw.txt
wc -c data/corpus.txt
```

Type it, do not paste it, and change the IDs. The `awk` line is the whole reason this is a script and not a download: it cuts the license wrapper the same way every time, so the file you get in January is byte-for-byte the file you got today. The `for` loop is the other reason: when one book comes up short, the fix is one more number on that line, not a second download you have to remember. A corpus from Wikipedia or a documentation repo needs a different rule, and the guide gives one for each source.

```bash
bash scripts/fetch_corpus.sh
```

*You should see* a byte count. On the Odyssey plus the Iliad this printed 1,607,209 (about 1.6 million bytes; the checker counts characters and says 1,594,224). If it printed something under 1,000, the URL returned an error page rather than the book; open `data/corpus.txt` and look. If it printed 700,000, you fetched one book; go back to the guide and pick its companion.

**Step 4. Run the checker. Read every line, not just the last one.**

```bash
uv run python scripts/check_corpus.py
```

*You should see* fifteen lines, each `ok`, `warn` or `FAIL`, and a verdict at the bottom. On two clean Gutenberg books, with `SOURCE.md` not yet written, you should see exactly one FAIL, `data/SOURCE.md`.

*If it broke* on `size`, you are under a million characters and the report says by how much. Add another book, season or category from the same source, in the fetch script, and run it again. Do not pad it with a different kind of text; a corpus that is half novel and half documentation gives you search results that make no sense all year.

*If it broke* on `boilerplate stripped`, your `awk` rule missed something. The checker names the phrase it found and which end of the file it found it at. The credits paragraph ("Produced by … Distributed Proofreading") sits *after* the START marker in older Gutenberg files, so the marker rule alone leaves it in; the checker reports it as a `warn` and quotes the line. Add a rule that drops that paragraph (`sed '1,/^$/d'` on the stripped text of that book, inside the loop) and re-run the fetch script from scratch to prove the rule works from raw.

*If it broke* on `chunks`, the paragraphs in your file are not separated by blank lines. The fix in the report converts single newlines; apply it in the fetch script, not by hand.

*If it broke* on `UTF-8 text` with "invalid continuation byte", you have a Latin-1 file. The `iconv` line in the report converts it; put that in the fetch script too.

**Step 5. Write `data/SOURCE.md` and make the hash match.**

```bash
shasum -a 256 data/corpus.txt     # sha256sum on Linux and Git Bash
```

Then `data/SOURCE.md`:

```markdown
# Corpus

Title: The Odyssey and The Iliad, Samuel Butler translations
URL: https://www.gutenberg.org/cache/epub/1727/pg1727.txt and https://www.gutenberg.org/cache/epub/2199/pg2199.txt
License: public domain (Project Gutenberg)
Fetched: 2026-09-28 by scripts/fetch_corpus.sh
sha256: <paste the whole hash>

What I expect to be different about this corpus: <one or two sentences, e.g.
"names are transliterated Greek, so the tokenizer will split most of them;
book headings repeat 48 times, 24 per poem">
```

`data/` is git-ignored except for this one file; `.gitignore` in the template already has the `!data/SOURCE.md` line. Re-run the checker.

*You should see* `0 FAIL` and the verdict "This corpus will carry you through the year."

**Step 6. Run the tests.**

```bash
uv run pytest tests/test_corpus.py -v
```

*You should see* nine passing and one skipped. The skipped one is yours, and that is the extension. If anything fails here that the checker passed, read the test's docstring: each one names the assignment that would break.

*If it broke* with `ModuleNotFoundError: No module named 'check_corpus'`, you ran `pytest` from inside `tests/`. Run it from the repo root; the test file finds `scripts/` relative to the root.

**Extension — one test that is true of your corpus and would fail on a random book**

Replace the body of `test_something_true_of_my_corpus` in `tests/test_corpus.py` with a test that a stranger could read and learn one specific thing about your file from. Three shapes that work:

A count that is a fact about the text: on the Odyssey plus the Iliad, `assert text.count("\nBOOK ") == 48`, and a count of 24 is how you would learn the second fetch silently failed. On the Federalist Papers, the number of `FEDERALIST No.` headings. On a season of box scores, the number of games.

A name that has to be there: `assert text.lower().count("telemachus") > 100`. Pick the name whose absence would prove you fetched the wrong file.

A structural property: every line under 80 characters; every chapter heading followed by a blank line; no line starting with a tab.

Run it. It has to pass, and it has to be the kind of thing that would *fail* if `fetch_corpus.sh` silently fetched the wrong book, which is the point: this test is the alarm that goes off in January if the URL rots. Then break it on purpose once, by changing the expected number, and read the failure message so you know what the alarm sounds like.

Commit, push, open the pull request:

```bash
git add scripts/fetch_corpus.sh scripts/check_corpus.py tests/test_corpus.py data/SOURCE.md pyproject.toml uv.lock
git status                      # data/corpus.txt must NOT be listed
git commit -m "A05b: lock in corpus, checker passes, one test of my own"
git push -u origin dev/corpus
```

**Open the pull request** for `dev/corpus`, **jd12** under **Reviewers**, **Create pull request**. In the PR body, paste the checker's verdict line and any `warn` lines, and say in one sentence what you did about each warning or why you left it.

Close the log:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
All of it, today: `scripts/fetch_corpus.sh` (re-creates a `data/corpus.txt` of at least 1,000,000 characters from the URL) · `data/SOURCE.md` (title, URL, license, date, sha256, what you expect to be different) · `tests/test_corpus.py` with your own test filled in · a checker run at `0 FAIL` pasted in the PR body. `data/corpus.txt` itself is never committed.

**Reflection Questions**
1. Paste the full checker report from your final run. For every `warn` line, say what you decided and why. If there were none, say which check you came closest to failing and how you know.
2. Paste your own test from `tests/test_corpus.py` and the pytest output for it, once passing and once after you broke the expected value on purpose. Say what about your corpus that test is guarding, and what a wrong-URL fetch would have to look like for the test to *miss* it.
3. Delete `data/corpus.txt`, run `bash scripts/fetch_corpus.sh`, and run `shasum -a 256 data/corpus.txt` again. Paste both hashes. If they match, say which line of the fetch script is doing the most work to make that true. If they do not, say what changed between runs and fix the script until they do, then paste the third hash.
