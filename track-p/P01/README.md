# P01 · The Seat, the CLI, the Repo, and Chapter 1

**Meetings:** D24 · **Points:** 10 pts

**Watch — none**

Boot.dev has no video: short readings, a function to write, tests run on your machine by its CLI. Today you buy the seat, point the CLI at it, accept the track repo, bring your corpus over, and start chapter 1.

**Notes**

**Keep Boot.dev's project outside every course repo.** The course builds its own project on its own dataset, and its chapter 1 lesson 2 creates the folder for you: `uv init --no-package rag-search-engine` makes `rag-search-engine/` with its own git repo inside it. Run that from `~/version_control`, so the project lands at `~/version_control/rag-search-engine/`, a sibling of your course repos and never inside one. The CLI's checks run in that folder, so keep the course's name for it.

**Type every solution yourself.** Do not paste from the hint panel or the solution tab. Every chapter ends with you writing the same function a second time on your corpus, and the second time goes quickly only if the first time went through your fingers.

**Run before you submit.** Every lesson, `bootdev run <lesson-id>` first, without `-s`. That runs the tests locally and shows you what they check. Read the output, fix, and only then `bootdev run <lesson-id> -s`.

**Where the course calls a model, it uses free models through OpenRouter.** A `429` means you are throttled, not broken. Wait sixty seconds; hammering Run makes the cooldown longer.

**Walkthrough — set up the semester**

**Step 1. The seat.**

Sign up at [boot.dev](https://www.boot.dev) with your school email. I buy the month in class. Open [*Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) and leave it on chapter 1, lesson 1.

**Step 2. The CLI.**

The Boot.dev CLI is a Go program, so Go has to be installed for it to build:

```bash
brew install go
go install github.com/bootdotdev/bootdev@latest
bootdev login
```

On Windows, in Git Bash: `winget install GoLang.Go`, open a new Git Bash, then the same `go install` and `bootdev login` lines.

`bootdev login` prints a URL; open it, copy the token it shows, paste it back. *You should see* `Logged in successfully!`. *If it broke* with `bootdev: command not found` after the install, `$(go env GOPATH)/bin` is not on your PATH: add `export PATH="$PATH:$(go env GOPATH)/bin"` to `~/.bash_profile` (`~/.bashrc` on Windows) and open a new terminal.

**Step 3. Accept the rag repo.**

Everything in the applied track from today to June lives in one repo. Accept it once:

**[RAG](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/rag/accept)**

```bash
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-rag-<your-username>.git
cd hse-2026-2027-rag-<your-username>
./setup.sh
git switch -c dev/setup
```

Hooks do not travel with a clone, so `setup.sh` again. *You should see* it end by asking you to try committing a fake key; do it and watch it get refused. Its environment check now has two lines it did not have in the foundations repo, `Go toolchain` and `Boot.dev CLI`, and both have to show `[  ok  ]`. Paste those two lines under `## Step 2: the CLI` in `evidence/P01.md`. The slot is named for the CLI, not for the step you are on.

Then close the Unit 0 log and open the track log. On GitHub, merge your student log's `<your-username>-unit0` pull request and delete the branch, then:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git switch main && git pull
git switch -c <your-username>-track
bash scripts/start-entry.sh
```

That branch stays open for every applied assignment until Wed Dec 9. Under the timestamp, write today's checklist:

```markdown
- [ ] Seat bought, bootdev login done, check_env says ok twice
- [ ] rag repo accepted, setup.sh run, corpus fetched and the sha256 matches
- [ ] Chapter 1, first half, every lesson run locally before it is submitted
- [ ] Fill every slot in evidence/P01.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 4. Bring your corpus.**

The corpus is the one you locked at A05b, and the same file has to land here byte for byte:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
cp ../hse-2026-2027-foundations-<your-username>/scripts/fetch_corpus.sh scripts/
cp ../hse-2026-2027-foundations-<your-username>/data/SOURCE.md data/
bash scripts/fetch_corpus.sh
shasum -a 256 data/corpus.txt
grep -i sha256 data/SOURCE.md
```

*You should see* the same 64 hex characters twice. Paste both lines under `## Step 4`, and put the `Title:` line from `data/SOURCE.md` on the `Corpus:` line at the top of `evidence/P01.md`. *If it broke* with a `curl` error or a `data/corpus.txt` of 0 bytes, the download was blocked; copy `data/corpus.txt` from the foundations repo instead and run the hash check again. *If it broke* with different hashes, the source changed since September; do not edit the hash to match. Run the script in the foundations repo too, `diff` the two files, and tell me which one moved. `git status` shows `scripts/fetch_corpus.sh` and `data/SOURCE.md` as new and nothing else under `data/`.

**Step 5. Chapter 1, first half.**

Lesson 2 sets up the project. Its six steps, as commands, run from `~/version_control` so the project lands beside your course repos:

```bash
cd ~/version_control
uv init --no-package rag-search-engine
cd rag-search-engine
uv venv
source .venv/bin/activate
mkdir data
curl -L -o data/movies.json https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json
echo "/data/" >> .gitignore
bootdev run -s
```

*You should see* `(rag-search-engine)` at the front of your prompt after the `source` line, a `data/movies.json` of a few megabytes after the `curl`, and the lesson's three checks pass on `bootdev run -s`. *If it broke* with `curl: command not found` on Windows, download the file in the browser from that link and move it to `data/movies.json`. Every later lesson runs inside `~/version_control/rag-search-engine/` with that environment active. (That project uses an activated venv the way Boot.dev likes; the course repos use `uv run`. Both are fine, they are different folders.) For the rest of the period: read, write the function, `bootdev run <id>`, fix, `bootdev run <id> -s`. Today's setup eats most of the period, so one or two lessons is a normal day. Stop when the period ends, wherever you are; P02 finishes the chapter. Before you stop, save a screenshot of the chapter page showing which lessons are green as `evidence/p01-bootdev.png` in the rag repo.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P01.md` (the questions are below) and run `uv run python scripts/check_evidence.py P01` until it says `all slots filled`. Then:

```bash
git add scripts/ data/SOURCE.md evidence/ scratch/
git status                      # data/corpus.txt must NOT be listed
git commit -m "P01: seat, CLI, corpus, chapter 1 started"
git push -u origin dev/setup
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the lesson you stopped on.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push -u origin <your-username>-track
```

**Deliverable**
`evidence/P01.md` with every slot filled + `evidence/p01-bootdev.png` + `scripts/fetch_corpus.sh` and `data/SOURCE.md` committed + a Boot.dev account with the seat on it and the CLI logged in. `data/corpus.txt` is never committed.

**Reflection Questions** (answer under `## Reflection` in `evidence/P01.md`)

1. Chapter 1's first lessons normalize text before anything else sees it. Name the first two operations the chapter applies, in its order, and say what each one is for.

   *How to get it:* the lesson titles in chapter 1's sidebar are the operations, in order. Write them down in `scratch/p02-order.md` now; P02 needs the whole list.

2. `bootdev run <id>` without `-s` runs the tests on your machine. Paste one test's output from a lesson that failed before it passed, and say what the test was checking.

   *How to get it:* the CLI prints each test's input and expected output above the pass or fail line. Copy the lines for the one that failed, not the whole run. If every lesson passed first time, break one on purpose (return the wrong thing), run `bootdev run <id>` once without `-s`, paste that, and put it back.

3. The course grades on its own dataset. Name one thing about your corpus that the chapter's preprocessing would handle differently from the course's text, and why.

   *How to get it:* `head -c 600 data/corpus.txt` shows the opening of your file. Look for what the chapter's examples do not have: curly quotes, non-English letters, chapter headings, numbers, a license line.
