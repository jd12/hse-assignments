# A05b · Your Corpus for the Year, and the Tests It Has to Pass

**Meetings:** D13 · **Points:** 5 pts

**Watch** — none

No video today. This is thirty minutes of choosing one text file and proving it will hold up, and everything after it depends on the file being right. A06 starts from it this same period.

**Notes**

From today until May, every assignment in this course runs on one text file that is yours: `data/corpus.txt`. A06 searches it. A09 samples prompts from it. A10 asks a model questions whose answers are in it and catches the model making things up. Track A builds a retrieval system over it and measures recall against a golden set drawn from it. Track B computes its entropy, fits lines to it, and in the spring trains a character-level GPT on it and reads what comes out. Every number you report this year is a number about this file, which is why no two students in the room can hand in the same answer.

So the file has to be good, and "good" is specific. Big enough that a search over it has something to find and a model trained on it produces readable text: **one million characters is the floor**, and the checker fails below it. Plain UTF-8 text with paragraphs separated by blank lines, because that is what the chunker in A06 splits on. Free of the license header and footer a download comes wrapped in, because otherwise "Project Gutenberg" is the top hit for a third of your queries. Not mostly repeats. Something you can legally use and are willing to have excerpts of pasted into committed files all year, which rules out anything private and anything you scraped from behind a login. And re-fetchable by a script, because I re-run your numbers, and a corpus nobody can reproduce is a result nobody can check.

`Choosing a Corpus.md`, linked from the repo README, has sources for literature, history, sports, technology, science, games, law and food, each with the command that fetches it. The worked example throughout this course is Homer, because it is what I used: the Odyssey and the Iliad in Butler's translation, Project Gutenberg #1727 and #2199, glued into one file of about 1.5 million characters. The Odyssey alone is 700,000, under the floor, which is the first lesson of the day: one book is usually not enough, and the fix is a second book from the same source, fetched by the same script.

**The checker is a script and a test file, and they check the same things.** `scripts/check_corpus.py` prints a report with a fix under every failure. `tests/test_corpus.py` is the same set of facts as pytest tests, one per thing a later assignment assumes. The last test in it is a placeholder, and the extension replaces it with one of five sample tests, with your own string and your own number in it.

One thing the checker will get wrong on purpose and you should know about. It warns if the text does not look like English prose, because A03 through A09 use English examples; a corpus of chess games or Node documentation is allowed and is a good choice, and the warning is there so you write down what you expect to be different, not so you change your mind. Size is not like that. Under a million characters is a FAIL, today, because Track B trains a model on this file in the spring and nothing under a million produces readable samples, and I would rather every one of you hit that wall now, with the fetch script open, than in February. Everything on today's list is required today: the fetch script, the checker at 0 FAIL, `SOURCE.md`, and your test.

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
- [ ] Paste scripts/fetch_corpus.sh with my IDs in it, and run it
- [ ] Run scripts/check_corpus.py until it reports 0 FAIL
- [ ] Write data/SOURCE.md with the sha256
- [ ] Run pytest: 9 passed, 1 skipped
- [ ] Pick one sample test, find my number, paste it in: 10 passed
- [ ] Break it on purpose, read the failure, fix it back
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

Open `Choosing a Corpus.md` and pick. Then create `scripts/fetch_corpus.sh`: in VS Code, right-click `scripts`, **New File**, name it `fetch_corpus.sh`, and paste this in. **The one line to change is the `for ID in` line**: replace `1727 2199` with the IDs of your books, from the guide, separated by spaces.

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

The `awk` line is the whole reason this is a script and not a download: it cuts the license wrapper the same way every time, so the file you get in January is byte-for-byte the file you got today. The `for` loop is the other reason: when one book comes up short, the fix is one more number on that line, not a second download you have to remember. A corpus from Wikipedia or a documentation repo needs a different `curl` line and a different stripping rule; the guide gives the complete script for each source, so if yours is not Gutenberg, paste the guide's script instead of this one.

```bash
bash scripts/fetch_corpus.sh
```

*You should see* a byte count. On the Odyssey plus the Iliad this printed `1607209 data/corpus.txt` (about 1.6 million bytes; the checker counts characters and says 1,594,224). If it printed something under 1,000, the URL returned an error page rather than the book; open `data/corpus.txt` and look. If it printed about 700,000, you fetched one book; go back to the guide and pick its companion.

**Step 4. Run the checker. Read every line, not just the last one.**

```bash
uv run python scripts/check_corpus.py
```

*You should see* fifteen lines, each `ok`, `warn` or `FAIL`, and a verdict at the bottom. On two clean Gutenberg books, with `SOURCE.md` not yet written, you should see exactly one FAIL, `data/SOURCE.md`. On Homer the `chunks` line read `2079 chunks, median 616 chars, longest 3793` and the `vocabulary` line `0.0369 over 284,435 tokens, 10,482 distinct`; yours will differ, and both are numbers you will meet again in A06.

*If it broke* on `size`, you are under a million characters and the report says by how much. Add another book, season or category from the same source, on the `for ID in` line, and run the fetch script again. Do not pad it with a different kind of text; a corpus that is half novel and half documentation gives you search results that make no sense all year.

*If it broke* on `boilerplate stripped`, your `awk` rule missed something. The checker names the phrase it found and which end of the file it found it at. The credits line ("Produced by David Widger" or similar) sits *after* the START marker in older Gutenberg files, so the marker rule alone leaves it in; the checker reports it as a `warn` near the start and quotes the line. The fix is to cut each book to its own file first and drop everything through the credits line when there is one. Replace the `awk` line in the loop with these six lines, and change the last `rm` line to `rm data/raw.txt data/book.txt`:

```bash
  awk '/\*\*\* START OF/{flag=1; next} /\*\*\* END OF/{flag=0} flag' data/raw.txt > data/book.txt
  if grep -q "Produced by" data/book.txt; then
    sed '1,/Produced by/d' data/book.txt >> data/corpus.txt    # drop line 1 through the credits line
  else
    cat data/book.txt >> data/corpus.txt
  fi
```

`sed '1,/Produced by/d'` deletes from the first line through the first line containing `Produced by`; the `if` keeps it from running on a book that has no credits line, where it would delete the whole book. Re-run the fetch script from scratch to prove the rule works from raw, then the checker.

*If it broke* on `chunks`, the paragraphs in your file are not separated by blank lines. The fix in the report converts single newlines; apply it in the fetch script, not by hand.

*If it broke* on `UTF-8 text` with "invalid continuation byte", you have a Latin-1 file. The `iconv` line in the report converts it; put that in the fetch script too.

**Step 5. Write `data/SOURCE.md` and make the hash match.**

```bash
shasum -a 256 data/corpus.txt     # sha256sum on Linux and Git Bash
```

Then create `data/SOURCE.md` (right-click `data`, **New File**) and paste this in, with your own title, URLs, license, date and hash:

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

*You should see* `9 passed, 1 skipped`. The skipped one is `test_something_true_of_my_corpus`, and the extension fills it in. If anything fails here that the checker passed, read the test's docstring: each one names the assignment that would break.

*If it broke* with `file or directory not found: tests/test_corpus.py`, you ran `pytest` from inside `tests/`. Run it from the repo root, where the path `tests/test_corpus.py` exists.

**Extension — one test that is true of your corpus and would fail on a random book (ASSIGNED)**

The nine tests you just ran are true of every good corpus. The tenth has to be true of **yours** and false of almost any other file, because this test is the alarm that goes off in January if a URL rots and `fetch_corpus.sh` silently pulls down the wrong book. Below are five complete tests. Pick the one that fits your corpus, run the command that finds your number, change the string and the number in the test, paste it over the placeholder, and run pytest.

Open `tests/test_corpus.py`. The placeholder is the last thing in the file:

```python
def test_something_true_of_my_corpus(text):
    pytest.skip("write your own test here (A05b, Extension)")
```

Delete those two lines and paste one of the five in their place. Every one of them takes `text`, which is your whole corpus as one string, and the lines marked with a comment are the ones you change.

**Sample 1. A heading that repeats a known number of times.** For a Gutenberg book with `BOOK` or `CHAPTER` lines, the Federalist Papers (`FEDERALIST No.`), a legal code (`Section`), or a season of box scores where every game ends with the same word. Find your number:

```bash
grep -c "^BOOK " data/corpus.txt
```

`^` means the line starts with what follows, so this counts lines that begin with `BOOK` and a space. On Homer it printed `48`: 24 books in each poem. Then:

```python
def test_something_true_of_my_corpus(text):
    """A05b: the Odyssey and the Iliad have 24 books each, so 'BOOK ' starts a line 48 times.
    24 means the second fetch silently failed."""
    HEADING = "BOOK "          # the text a heading line starts with
    EXPECTED = 48              # how many lines start with it, from grep -c
    count = sum(1 for line in text.splitlines() if line.startswith(HEADING))
    assert count == EXPECTED, f"{count} lines start with {HEADING!r}, expected {EXPECTED}"
```

**Sample 2. A name that has to occur more than N times.** For any corpus with a main character, a team, a place or a product that appears on most pages: a novel's hero, a Wikipedia category's topic word, a team's name in a season of game reports. Pick the name whose absence would prove you fetched the wrong file, and count it:

```bash
grep -o -i "telemachus" data/corpus.txt | wc -l
```

`-o` prints each match on its own line and `-i` ignores case, so `wc -l` counts every occurrence, not every line. On Homer it printed `284`. Set the threshold to a round number safely under your count, so a different edition still passes and a different book does not:

```python
def test_something_true_of_my_corpus(text):
    """A05b: Telemachus is in the Odyssey hundreds of times and in no other book I could fetch by mistake."""
    NAME = "telemachus"        # lowercase; the count ignores case
    AT_LEAST = 200             # a round number safely under the grep -o | wc -l count
    count = text.lower().count(NAME)
    assert count > AT_LEAST, f"{NAME!r} occurs {count} times, expected more than {AT_LEAST}"
```

**Sample 3. No line is longer than a limit.** For a Gutenberg book (wrapped at about 72 characters) or a documentation repo (wrapped at 80 or 100). Not for Wikipedia, where one paragraph is one line. Find your longest line:

```bash
uv run python -c "print(max(len(line) for line in open('data/corpus.txt', encoding='utf-8').read().splitlines()))"
```

On Homer it printed `79`. The checker's `longest line` says 80 on the same file, because the file has Windows line endings and the checker counts the carriage return; use this command's number. (`awk '{print length}'` counts bytes, not characters, so it disagrees on every line with a curly quote in it.) Set the limit a little above your longest line:

```python
def test_something_true_of_my_corpus(text):
    """A05b: Gutenberg wraps prose at about 72 characters, so no line is longer than 80.
    A longer line means an unwrapped file (HTML, a different edition) got in."""
    MAX_LINE = 80              # the longest line this file is allowed to have
    longest = max(len(line) for line in text.splitlines())
    assert longest <= MAX_LINE, f"longest line is {longest} characters, limit {MAX_LINE}"
```

**Sample 4. A phrase that must be present.** For any corpus, and the best choice for Wikipedia and documentation: a sentence from the one article or page that defines the collection, a function name, a rule number. Pick a phrase from the part of the file you would miss most, and prove it is there:

```bash
grep -c "my name is Noman" data/corpus.txt
```

On Homer it printed `1`: the Cyclops scene, Book IX of the Odyssey. Copy the phrase from the file exactly, including capitals and punctuation; `grep -n` in place of `grep -c` shows you the line so you can check your spelling against it.

```python
def test_something_true_of_my_corpus(text):
    """A05b: the Cyclops scene is in Book IX of the Odyssey and nowhere else; if this phrase is gone, so is the book."""
    PHRASE = "my name is Noman"   # copied exactly from the file
    assert PHRASE in text, f"{PHRASE!r} is not in the corpus"
```

**Sample 5. A ratio: the share of lines that start with a digit.** For sports and logs, where most lines begin with a score, a date or a time, the test says the share must be high. For prose, it says the share must be tiny. Find your two numbers:

```bash
grep -c "^[0-9]" data/corpus.txt
wc -l < data/corpus.txt
```

`^[0-9]` means the first character of the line is a digit, 0 to 9. On Homer the two commands printed `12` and `26438`: twelve lines in 26,438 start with a digit, a share of 0.05%, eleven of them footnote numbers and one a date in the preface. For a prose corpus keep the test as written. For a box-score corpus, change `<=` to `>=` on the `assert` line and set the number to a share safely under yours (if 41% of your lines start with a digit, `0.30`); the name `AT_MOST` then reads wrong, so change it to `AT_LEAST` in all three places it appears, or leave it and say so in the docstring.

```python
def test_something_true_of_my_corpus(text):
    """A05b: box scores and logs start most lines with a number; prose starts almost none.
    This file is prose, so the share has to stay under 1%."""
    AT_MOST = 0.01             # share of lines allowed to start with a digit (0.01 = 1%)
    lines = text.splitlines()
    digit_lines = sum(1 for line in lines if line[:1].isdigit())
    share = digit_lines / len(lines)
    assert share <= AT_MOST, f"{digit_lines} of {len(lines)} lines start with a digit ({share:.1%}), limit {AT_MOST:.0%}"
```

Whichever one you pasted, change its docstring so it says what the test guards in your file, then run the whole file:

```bash
uv run pytest tests/test_corpus.py -v
```

*You should see* `10 passed`, and no `skipped`.

*If it broke* with `IndentationError`, the pasted function is indented; every line of it starts at the left margin except the body lines, which start four spaces in. If `test_something_true_of_my_corpus FAILED` on the first run, your number is wrong, not the file: read the message after `AssertionError:`, which prints the real count, and fix the number.

**Break it on purpose.** Change the expected value in your test to something wrong (in Sample 1, `EXPECTED = 24`; in Sample 2, `AT_LEAST = 100000`; in Sample 3, `MAX_LINE = 20`; in Sample 4, misspell the phrase; in Sample 5, `AT_MOST = 0.0`) and run pytest again:

```bash
uv run pytest tests/test_corpus.py -v --tb=short 2>&1 | tail -15
```

*You should see* `1 failed, 9 passed`, and above it the failing line marked with `>` and the message. On Homer, with `EXPECTED = 24`, the tail of the output read:

```text
>       assert count == EXPECTED, f"{count} lines start with {HEADING!r}, expected {EXPECTED}"
E       AssertionError: 48 lines start with 'BOOK ', expected 24
E       assert 48 == 24

tests/test_corpus.py:115: AssertionError
=========================== short test summary info ============================
FAILED tests/test_corpus.py::test_something_true_of_my_corpus - AssertionErro...
========================= 1 failed, 9 passed in 0.19s ==========================
```

That `AssertionError` line is what the alarm sounds like: it names the thing counted, the count it found, and the count it expected, which is enough to know whether the fetch pulled one book or two before you open the file. Paste that block into your log for Reflection Question 2. Then put the right value back and run pytest once more to see `10 passed`.

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
All of it, today: `scripts/fetch_corpus.sh` (re-creates a `data/corpus.txt` of at least 1,000,000 characters from the URL) · `data/SOURCE.md` (title, URL, license, date, sha256, what you expect to be different) · `tests/test_corpus.py` with one of the five sample tests filled in with your string, your number and your docstring · a checker run at `0 FAIL` pasted in the PR body. `data/corpus.txt` itself is never committed.

**Reflection Questions**

1. Paste the full checker report from your final run. For every `warn` line, say what you decided and why. If there were none, say which check you came closest to failing and how you know.

   *How to get it:* `uv run python scripts/check_corpus.py` from the repo root prints the report; copy all of it, from the first `=====` line to the last. The thresholds are listed at the top of `scripts/check_corpus.py` under "the thresholds, in one place", so "closest to failing" is the line whose number is nearest its threshold: on Homer that is `distinct characters`, 141 against a ceiling of 200, and you find yours by opening the script and comparing each `ok` line's number with the constant it is checked against.

2. Paste your test from `tests/test_corpus.py` and the pytest output for it, once passing and once after you broke the expected value on purpose. Say what about your corpus that test is guarding, and what a wrong-URL fetch would have to look like for the test to *miss* it.

   *How to get it:* the test is the last function in `tests/test_corpus.py`. The passing output is the `tests/test_corpus.py::test_something_true_of_my_corpus PASSED` line from `uv run pytest tests/test_corpus.py -v`; the broken output is the block you pasted into your log from the "Break it on purpose" step. For the last part, re-read the command that found your number: a wrong file that happens to produce the same count, the same name more than N times, or the same phrase, is the one your test cannot see, so name one such file if you can think of one (for Sample 1 on Homer, any two 24-book poems; for Sample 4, any other edition of the Odyssey).

3. Delete `data/corpus.txt`, run `bash scripts/fetch_corpus.sh`, and run `shasum -a 256 data/corpus.txt` again. Paste both hashes. If they match, say which line of the fetch script is doing the most work to make that true. If they do not, say what changed between runs and fix the script until they do, then paste the third hash.

   *How to get it:* from the repo root:

   ```bash
   rm data/corpus.txt
   bash scripts/fetch_corpus.sh
   shasum -a 256 data/corpus.txt     # sha256sum on Linux and Git Bash
   ```

   The first hash is the one in `data/SOURCE.md`; the second is what this prints. If they differ, `diff` on the two files is not possible, because the first is gone, so run the fetch twice more and compare those two hashes with each other: identical means the source changed between your first fetch and now (update `SOURCE.md` with the new hash and the date); different means something in the script is not repeatable, and the usual culprit is a line that does not clear `data/corpus.txt` before appending.
