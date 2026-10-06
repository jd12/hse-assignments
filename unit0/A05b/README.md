# A05b · Your Corpus for the Year, and the Tests It Has to Pass

**Meetings:** D13 · **Points:** 5 pts

**Watch — none**

**Notes**

From today until May, every assignment in this course runs on one text file that is yours: `data/corpus.txt`. A06 searches it, A09 samples prompts from it, A10 asks a model questions whose answers are in it, Track A measures recall over it, and Track B trains a character-level GPT on it in the spring. Every number you report this year is a number about this file, which is why no two students in the room can hand in the same answer. "Good" is specific. **One million characters is the floor**, because nothing under it trains into readable samples, and the checker fails below it. Plain UTF-8 with paragraphs separated by blank lines, because that is what A06's chunker splits on. No license header or footer, or "Project Gutenberg" is the top hit for a third of your queries. Something you can legally use and have excerpts of committed all year, which rules out anything private or scraped from behind a login. And re-fetchable by a script, because I re-run your numbers, and a corpus nobody can reproduce is a result nobody can check.

`Choosing a Corpus.md`, linked from the repo README, has sources for literature, history, sports, technology, science, games, law and food, each with the command that fetches it. The worked example all year is Homer: the Odyssey and the Iliad in Butler's translation, Project Gutenberg #1727 and #2199, about 1.5 million characters. The Odyssey alone is 700,000, under the floor: one book is usually not enough, and the fix is a second book from the same source. **The checker is a script and a test file, and they check the same things.** `scripts/check_corpus.py` prints a report with a fix under every failure. `tests/test_corpus.py` is the same facts as pytest tests, one per thing a later assignment assumes; its last test is a placeholder the extension fills with one of five sample tests. The checker warns if the text does not look like English prose; chess games or Node documentation are allowed, and the warning is there so you write down what you expect to be different, not so you change your mind.

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
- [ ] Pull the checker, the unit's helper scripts and the evidence files from the template
- [ ] Pick a corpus from the guide; paste scripts/fetch_corpus.sh with my IDs in it and run it
- [ ] Run scripts/check_corpus.py until it reports 0 FAIL; write data/SOURCE.md with the sha256
- [ ] Run pytest: 9 passed, 1 skipped
- [ ] Pick one sample test, find my number, paste it in: 10 passed; break it on purpose, fix it back
- [ ] Fill every slot in evidence/A05b.md (check_evidence.py: all slots filled); push, open the PR, sign off the log
```

**Step 2. Pull the checker, the helper scripts for the rest of the unit, and the evidence files into your repo.** The template shipped without them.

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git fetch template 2>/dev/null || git remote add template https://github.com/Sierra-Canyon/foundations-template.git
git fetch template
git checkout template/main -- scripts/check_corpus.py scripts/check_evidence.py tests/test_corpus.py scratch transformer failures evidence
uv add --dev pytest
ls scripts tests scratch evidence
```

*You should see* `check_corpus.py` and `check_evidence.py` under `scripts/`, `test_corpus.py` under `tests/`, the unit's helper scripts under `scratch/`, and `A05b.md` through `A10.md` under `evidence/`. **Each assignment from today on has one evidence file**, `evidence/A<nn>.md`, with a marked slot for every paste and every answer; you fill the slots as you reach them, and `uv run python scripts/check_evidence.py A05b` lists the ones still empty. *If it broke* with `fatal: couldn't find remote ref main`, the template's default branch is `master` on your clone; use `template/master` in the checkout line. <!-- JD: the checkout line pulls the directories the A06–A10 helpers and evidence templates land in (scratch/, transformer/, failures/, evidence/). If one of them is not in the template yet, drop it from the line: one missing pathspec fails the whole checkout. Students who accepted the repo before the helpers landed get them through the template-update PR instead. -->

**Step 3. Choose, then write the fetch script before you download anything.**

Open `Choosing a Corpus.md` and pick. Create `scripts/fetch_corpus.sh` (right-click `scripts`, **New File**) and paste this in. **The one line to change is the `for ID in` line**: replace `1727 2199` with the IDs of your books, from the guide, separated by spaces. If your source is not Gutenberg, paste the guide's script for that source instead. The `awk` line is why this is a script and not a download: it cuts the license wrapper the same way every time, so the file you get in January is byte-for-byte the file you got today. The `for` loop is the other reason: when one book comes up short, the fix is one more number on that line.

```bash
#!/bin/bash
# scripts/fetch_corpus.sh: re-creates data/corpus.txt from scratch, one Gutenberg book per ID.
# Run from the repo root:  bash scripts/fetch_corpus.sh      The one line to edit is the "for ID in" line.
set -e                                   # stop at the first command that fails, instead of carrying on with a broken file
mkdir -p data                            # make the folder if it is not there yet
: > data/corpus.txt                      # start empty; every book below is appended
for ID in 1727 2199; do                  # 1727 = the Odyssey, 2199 = the Iliad (Butler). Change these.
  curl -fsSL --retry 4 --retry-delay 5 --retry-all-errors "https://www.gutenberg.org/cache/epub/$ID/pg$ID.txt" -o data/raw.txt   # download; $ID is filled in; -o says where to save; retry up to 4 times
  # Keep only what is between the START and END markers, then drop the marker lines themselves.
  awk '/\*\*\* START OF/{flag=1; next} /\*\*\* END OF/{flag=0} flag' data/raw.txt >> data/corpus.txt   # flag turns on after START, off at END; lines print only while it is on; >> appends
  printf "\n\n" >> data/corpus.txt      # a blank line between books, so the chunker does not glue them
done
rm data/raw.txt                          # the download is no longer needed
wc -c data/corpus.txt                    # print the size in bytes
```

Run it with `bash scripts/fetch_corpus.sh`. *You should see* a byte count; paste that line under `## Step 3` in `evidence/A05b.md`. On the Odyssey plus the Iliad this printed `1607209 data/corpus.txt`. Under 1,000 means the URL returned an error page, not the book; open `data/corpus.txt` and look. About 700,000 means you fetched one book; go back to the guide and pick its companion.

**Step 4. Run the checker. Read every line, not just the last one.**

```bash
uv run python scripts/check_corpus.py
```

*You should see* fifteen lines, each `ok`, `warn` or `FAIL`, and a verdict at the bottom. On two clean Gutenberg books, with `SOURCE.md` not yet written, exactly one FAIL: `data/SOURCE.md`. On Homer the `chunks` line read `2079 chunks, median 616 chars, longest 3793` and the `vocabulary` line `0.0369 over 284,435 tokens, 10,482 distinct`; both are numbers you will meet again in A06.

*If it broke* on `size`, the report says by how much: add another book, season or category from the same source on the `for ID in` line and run the fetch script again; do not pad with a different kind of text, or your search results make no sense all year. *If it broke* on `chunks`, the paragraphs are not separated by blank lines; the fix in the report converts single newlines, and it goes in the fetch script, not by hand. *If it broke* on `UTF-8 text` with "invalid continuation byte", you have a Latin-1 file; the `iconv` line in the report converts it, and it goes in the fetch script too.

*If it broke* on `boilerplate stripped`, the checker names the phrase and which end of the file it found it at. The usual one is a credits line ("Produced by David Widger") that older Gutenberg files put *after* the START marker. Replace the `awk` line in the loop with these six lines, which cut each book to its own file and drop everything through the credits line only when there is one, change the last `rm` line to `rm data/raw.txt data/book.txt`, and re-run the fetch script from scratch, then the checker:

```bash
  # Cut this book to its own file, then drop its credits line if it has one.
  awk '/\*\*\* START OF/{flag=1; next} /\*\*\* END OF/{flag=0} flag' data/raw.txt > data/book.txt   # > writes a new file instead of appending
  if grep -q "Produced by" data/book.txt; then                 # -q: just say yes or no, print nothing
    sed '1,/Produced by/d' data/book.txt >> data/corpus.txt    # drop line 1 through the first line containing "Produced by"
  else
    cat data/book.txt >> data/corpus.txt                       # no credits line: append the whole book
  fi
```

**Step 5. Write `data/SOURCE.md` and make the hash match.**

Run `shasum -a 256 data/corpus.txt` (`sha256sum` on Linux and Git Bash) and paste the line it prints under `## Step 5` in `evidence/A05b.md`. Create `data/SOURCE.md` (right-click `data`, **New File**) and paste this in, with your own title, URLs, license, date and hash:

```markdown
# Corpus

Title: The Odyssey and The Iliad, Samuel Butler translations
URL: https://www.gutenberg.org/cache/epub/1727/pg1727.txt and https://www.gutenberg.org/cache/epub/2199/pg2199.txt
License: public domain (Project Gutenberg)
Fetched: 2026-09-28 by scripts/fetch_corpus.sh
sha256: <paste the whole hash>
What I expect to be different about this corpus: <one or two sentences, e.g. "names are transliterated Greek, so the tokenizer will split most of them; book headings repeat 48 times, 24 per poem">
```

`data/` is git-ignored except for this one file. Re-run the checker. *You should see* `0 FAIL` and the verdict "This corpus will carry you through the year."

**Step 6. Run the tests.**

```bash
uv run pytest tests/test_corpus.py -v
```

*You should see* `9 passed, 1 skipped`; paste the output under `## Step 6` in `evidence/A05b.md`. The skipped one is `test_something_true_of_my_corpus`, and the extension fills it in. If anything fails here that the checker passed, read the test's docstring: each one names the assignment that would break. *If it broke* with `file or directory not found: tests/test_corpus.py`, you ran `pytest` from inside `tests/`; run it from the repo root.

**Extension — one test that is true of your corpus and would fail on a random book (ASSIGNED)**

The nine tests you just ran are true of every good corpus. The tenth has to be true of **yours** and false of almost any other file, because it is the alarm that goes off in January if a URL rots and `fetch_corpus.sh` silently pulls down the wrong book. Below are five complete tests. Pick the one that fits your corpus, run the command that finds your number, change the string and the number, and paste it over the placeholder: the last two lines of `tests/test_corpus.py`, `def test_something_true_of_my_corpus(text):` and the `pytest.skip(...)` under it. `text` is your whole corpus as one string; the lines marked `(change this)` are the ones you change. Under `## Extension` in `evidence/A05b.md`, say which sample and why, and paste the command with its number and the test as you pasted it.

**Sample 1. A heading that repeats a known number of times.** For a Gutenberg book with `BOOK` or `CHAPTER` lines, the Federalist Papers (`FEDERALIST No.`), a legal code (`Section`), or a season of box scores where every game ends with the same word. Find your number with `grep -c "^BOOK " data/corpus.txt`; `^` means the line starts with what follows. On Homer it printed `48`, 24 books in each poem.

```python
# In: text, the whole corpus as one string (pytest passes it in). Out: nothing; the assert fails the test if the count is wrong.
def test_something_true_of_my_corpus(text):
    """A05b: 24 books in each poem, so 'BOOK ' starts a line 48 times; 24 means the second fetch failed."""
    HEADING = "BOOK "                    # what a heading line starts with (change this)
    EXPECTED = 48                        # the grep -c number (change this)
    count = 0
    for line in text.splitlines():       # splitlines() turns the text into a list of lines
        if line.startswith(HEADING):
            count += 1
    assert count == EXPECTED, f"{count} lines start with {HEADING!r}, expected {EXPECTED}"   # the message prints only when the test fails
```

**Sample 2. A name that has to occur more than N times.** For any corpus with a main character, a team, a place or a product on most pages. Count it with `grep -o -i "telemachus" data/corpus.txt | wc -l`; `-o` prints each match on its own line and `-i` ignores case, so `wc -l` counts every occurrence. On Homer it printed `284`. Set the threshold to a round number safely under your count, so a different edition still passes and a different book does not.

```python
# In: text, the whole corpus as one string. Out: nothing; the assert fails the test if the name is too rare.
def test_something_true_of_my_corpus(text):
    """A05b: Telemachus is in the Odyssey hundreds of times and in no other book I could fetch by mistake."""
    NAME = "telemachus"                  # lowercase, because the count below lowercases the text (change this)
    AT_LEAST = 200                       # a round number safely under the grep -o | wc -l count (change this)
    count = text.lower().count(NAME)     # how many times NAME appears, ignoring case
    assert count > AT_LEAST, f"{NAME!r} occurs {count} times, expected more than {AT_LEAST}"
```

**Sample 3. No line is longer than a limit.** For a Gutenberg book (wrapped at about 72 characters) or a documentation repo (80 or 100). Not for Wikipedia, where one paragraph is one line. Find your longest line with `uv run python -c "print(max(len(line) for line in open('data/corpus.txt', encoding='utf-8').read().splitlines()))"`; on Homer it printed `79`. The checker's `longest line` is one higher on the same file because it counts the carriage return, so use this number and set the limit a little above it.

```python
# In: text, the whole corpus as one string. Out: nothing; the assert fails the test if any line is too long.
def test_something_true_of_my_corpus(text):
    """A05b: Gutenberg wraps prose at about 72 characters; a longer line means an unwrapped file got in."""
    MAX_LINE = 80                        # a little above the longest line the command printed (change this)
    longest = max(len(line) for line in text.splitlines())   # same as: the length of every line, and the biggest of them
    assert longest <= MAX_LINE, f"longest line is {longest} characters, limit {MAX_LINE}"
```

**Sample 4. A phrase that must be present.** For any corpus, and the best choice for Wikipedia and documentation: a sentence from the one page that defines the collection, a function name, a rule number. Copy it from the file exactly, capitals and punctuation included, and prove it is there with `grep -c "my name is Noman" data/corpus.txt`; `grep -n` in place of `grep -c` shows the line so you can check your spelling against it. On Homer it printed `1`: the Cyclops scene, Book IX of the Odyssey.

```python
# In: text, the whole corpus as one string. Out: nothing; the assert fails the test if the phrase is missing.
def test_something_true_of_my_corpus(text):
    """A05b: the Cyclops scene is in Book IX of the Odyssey; if this phrase is gone, so is the book."""
    PHRASE = "my name is Noman"          # copied exactly from the file (change this)
    assert PHRASE in text, f"{PHRASE!r} is not in the corpus"   # "in" asks whether the phrase appears anywhere in the text
```

**Sample 5. A ratio: the share of lines that start with a digit.** High for sports and logs, tiny for prose. Find your two numbers with `grep -c "^[0-9]" data/corpus.txt` and `wc -l < data/corpus.txt`; `^[0-9]` means the first character of the line is a digit. On Homer they printed `12` and `26438`, a share of 0.05%: eleven footnote numbers and one date in the preface. For prose keep the test as written; for box scores, change `<=` to `>=` on the `assert` line, set the number safely under your share (if 41% of your lines start with a digit, `0.30`), and say so in the docstring.

```python
# In: text, the whole corpus as one string. Out: nothing; the assert fails the test if too many lines start with a digit.
def test_something_true_of_my_corpus(text):
    """A05b: box scores start most lines with a number; prose starts almost none. This file is prose."""
    AT_MOST = 0.01                       # share of lines allowed to start with a digit (0.01 = 1%) (change this)
    lines = text.splitlines()
    digit_lines = 0
    for line in lines:
        if line[:1].isdigit():           # line[:1] is the first character, or "" for an empty line
            digit_lines += 1
    share = digit_lines / len(lines)
    assert share <= AT_MOST, f"{share:.1%} of {len(lines)} lines start with a digit, limit {AT_MOST:.0%}"   # .1% prints a share as a percentage
```

Whichever one you pasted, change its docstring so it says what the test guards in your file, then run the whole file again: `uv run pytest tests/test_corpus.py -v`. *You should see* `10 passed`, and no `skipped`; paste that line in the evidence file. *If it broke* with `IndentationError`, the pasted function is indented; the `def` line starts at the left margin and the body lines start four spaces in. `test_something_true_of_my_corpus FAILED` on the first run means your number is wrong, not the file: the message after `AssertionError:` prints the real count. **Then break it on purpose.** Change the value in your test to something wrong (Sample 1, `EXPECTED = 24`; Sample 2, `AT_LEAST = 100000`; Sample 3, `MAX_LINE = 20`; Sample 4, misspell the phrase; Sample 5, `AT_MOST = 0.0`), write the wrong value in the evidence file's `<answer: ...>` slot, and run pytest again:

```bash
uv run pytest tests/test_corpus.py -v --tb=short 2>&1 | tail -15
```

*You should see* `1 failed, 9 passed`, and above it the failing `assert` line, a row of carets under it, and the message. On Homer, with Sample 1 and `EXPECTED = 24`:

```text
    assert count == EXPECTED, f"{count} lines start with {HEADING!r}, expected {EXPECTED}"   # the message prints only when the test fails
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   AssertionError: 48 lines start with 'BOOK ', expected 24
E   assert 48 == 24
```

That `AssertionError` line is what the alarm sounds like: it names the thing counted, the count it found and the count it expected, which is enough to know whether the fetch pulled one book or two before you open the file (Sample 4's failure also prints a long slice of the corpus on one line; the `AssertionError:` line above it is the one to read). Paste the block under `## Extension` in `evidence/A05b.md`, put the right value back, run pytest once more to see `10 passed`, and paste that line too. Fill `## Reflection` (the questions are below), run `uv run python scripts/check_evidence.py A05b` until it says `all slots filled`, then commit, push, and open the pull request:

```bash
git add scripts/fetch_corpus.sh scripts/check_corpus.py scripts/check_evidence.py tests/test_corpus.py data/SOURCE.md evidence/A05b.md pyproject.toml uv.lock
git status                      # data/corpus.txt must NOT be listed
git commit -m "A05b: lock in corpus, checker passes, one test of my own"
git push -u origin dev/corpus
```

**Open the pull request** for `dev/corpus`, **jd12** under **Reviewers**, **Create pull request**. In the PR body, paste the checker's verdict line and any `warn` lines, and say in one sentence what you did about each warning or why you left it. Then close the log:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
All of it, today: `scripts/fetch_corpus.sh` (re-creates a `data/corpus.txt` of at least 1,000,000 characters from the URL) · `data/SOURCE.md` (title, URL, license, date, sha256, what you expect to be different) · `tests/test_corpus.py` with one of the five sample tests filled in with your string, your number and your docstring · `evidence/A05b.md` with every slot filled (the fetch run, the hash, the first pytest run, your test with its command, the passing and broken runs, the checker report, the three reflection answers) · a checker run at `0 FAIL` pasted in the PR body. `data/corpus.txt` itself is never committed.

**Reflection Questions** (answer under `## Reflection` in `evidence/A05b.md`)

1. Paste the full checker report from your final run. For every `warn` line, say what you decided and why. If there were none, say which check you came closest to failing and how you know.

   *How to get it:* `uv run python scripts/check_corpus.py` from the repo root prints the report; copy all of it, from the first `=====` line to the last. The thresholds are at the top of `scripts/check_corpus.py` under "the thresholds, in one place", so "closest to failing" is the `ok` line whose number is nearest its threshold. On Homer that is `distinct characters`, 141 against a ceiling of 200.

2. Paste your test from `tests/test_corpus.py` and the pytest output for it, once passing and once after you broke the expected value on purpose. Say what about your corpus that test is guarding, and what a wrong-URL fetch would have to look like for the test to *miss* it.

   *How to get it:* the passing output is the `tests/test_corpus.py::test_something_true_of_my_corpus PASSED` line from `uv run pytest tests/test_corpus.py -v`; the broken output is the block you pasted under `## Extension` in `evidence/A05b.md`. For the last part, re-read the command that found your number: a wrong file that produces the same count, the same name more than N times, or the same phrase is the one your test cannot see, so name one if you can (for Sample 1 on Homer, any two 24-book poems; for Sample 4, any other edition of the Odyssey).

3. Delete `data/corpus.txt`, run `bash scripts/fetch_corpus.sh`, and run `shasum -a 256 data/corpus.txt` again. Paste both hashes. If they match, say which line of the fetch script is doing the most work to make that true. If they do not, say what changed between runs and fix the script until they do, then paste the third hash.

   *How to get it:* the first hash is the one in `data/SOURCE.md`; the second is what the command at the end of this paragraph prints. If they differ, run the fetch twice more and compare those two hashes with each other: identical means the source changed since your first fetch (update `SOURCE.md` with the new hash and date); different means something in the script is not repeatable, usually a line that does not clear `data/corpus.txt` before appending. The command, from the repo root: `rm data/corpus.txt && bash scripts/fetch_corpus.sh && shasum -a 256 data/corpus.txt`.
