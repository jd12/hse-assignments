# Honors Software Engineering — Assignments

Every assignment in the course, one folder each, as the students see it (teacher callouts stripped). Built from the 2026–27 run; calendar dates are removed from the headers and the meeting-day numbers (`D13–D14`) are kept, so a future year re-dates without rewriting.

| Unit | What it is |
|---|---|
| `unit0/` | A01–A12 + A05b + A11b: everyone, Sep–Oct. Ends with the track election. |
| `track-a/` | A13–A25: applied AI engineering (retrieval, agents, MCP, evals). |
| `track-b/` | B13–B25: LLM internals (linear algebra, autograd, entropy, a GPT from scratch). |
| `capstone/` | C01–C09: April–May, everyone. |
| `resources/` | `Choosing a Corpus.md`, the corpus-sourcing guide linked from A05b. |

Assignments from A05b on follow the follow-along shape: **Watch** (per meeting, runtime in the header) → **During the video** (what to type along) → **Notes** → **Walkthrough** (code given, *You should see*, *If it broke*) → **Extension** (CHOOSE or ASSIGNED, with a number and a "where it got worse") → **Deliverable** → three **Reflection Questions** that each require a paste, a number, or a file:line.

Every assignment from A05b on runs on the student's own corpus, chosen and locked in A05b (`data/corpus.txt`, ≥ 1,000,000 characters). The worked example throughout is the Odyssey + the Iliad, Butler's translations, Project Gutenberg #1727 + #2199.

## Index

| ID | Title | Meetings | Points | Folder |
|---|---|---|---|---|
| [A01](unit0/A01/README.md) | Cold Start: Environment & Repo Bring-Up | D01–D02 | 15 pts | `unit0` |
| [A03](unit0/A03/README.md) | Transformer LLMs 01–03 + Karpathy Tokenizer Pt. 1 | D03–D05 | 15 pts | `unit0` |
| [A04](unit0/A04/README.md) | Transformer LLMs 04–06 + Karpathy Tokenizer Pt. 2 (BPE from Scratch) | ? | 15 pts | `unit0` |
| [A05](unit0/A05/README.md) | Neural Nets Ch. 1 + Embedding Space Probe | D09 | 15 pts | `unit0` |
| [A05b](unit0/A05b/README.md) | Your Corpus for the Year, and the Tests It Has to Pass | D13 | 5 pts | `unit0` |
| [A06](unit0/A06/README.md) | Semantic Search from Scratch (NumPy Only) | D13–D14 | 15 pts | `unit0` |
| [A07](unit0/A07/README.md) | 3B1B Transformers (Ch. 5) + Transformer LLMs 07–08 + Build the Block | D15–D16 | 15 pts | `unit0` |
| [A08](unit0/A08/README.md) | 3B1B Attention (Ch. 6) + Transformer LLMs 09 + One Head on Your Corpus | D17–D18 | 15 pts | `unit0` |
| [A09](unit0/A09/README.md) | Transformer LLMs 10 + Sampling by Hand | D19 | 15 pts | `unit0` |
| [A10](unit0/A10/README.md) | Failure Catalog: Why Models Lie | D20–D21 | 15 pts | `unit0` |
| [A11](unit0/A11/README.md) | Evaluating AI Agents 02–06: Build and Trace Your Own Agent | D22–D23 | 15 pts | `unit0` |
| [A11b](unit0/A11b/README.md) | Q1 Self-Assessment | D22–D23 | Ungraded | `unit0` |
| [A12](unit0/A12/README.md) | Evaluating AI Agents 07–14 + Unit 0 Capstone & Track Election | D24–D26 | 15 pts | `unit0` |
| [A13](track-a/A13/README.md) | Text In, Numbers Out: Preprocessing and TF-IDF | D27–D28 | 15 pts | `track-a` |
| [A14](track-a/A14/README.md) | Keyword Search and Its Ceiling | D29 | 15 pts | `track-a` |
| [A15](track-a/A15/README.md) | Semantic Search and the Embedding Space | D30 | 15 pts | `track-a` |
| [A16](track-a/A16/README.md) | Chunking and Hybrid Search | D31 | 15 pts | `track-a` |
| [A17](track-a/A17/README.md) | Query Expansion and Reranking | D32–D33 | 15 pts | `track-a` |
| [A18](track-a/A18/README.md) | End-to-End RAG with Citations and a Refusal Policy | D34–D35 | 15 pts | `track-a` |
| [A19](track-a/A19/README.md) | Measuring Retrieval: Recall, Precision, and the Golden Set | D36–D37 | 15 pts | `track-a` |
| [A20](track-a/A20/README.md) | ★ MILESTONE A1: Grounded Agent with a Measured Delta | D38–D39 | 15 pts | `track-a` |
| [A21](track-a/A21/README.md) | Three Kinds of Memory | D40–D42 | 15 pts | `track-a` |
| [A22](track-a/A22/README.md) | Agentic Retrieval and a Retention Policy | D43 | 15 pts | `track-a` |
| [A25](track-a/A25/README.md) | FLOAT: Catch-Up, Repair, and Retrieval Postmortem | D44–D45 | 15 pts | `track-a` |
| [B13](track-b/B13/README.md) | Vectors, Span, and What a System of Equations Actually Asks | D27 | 15 pts | `track-b` |
| [B14](track-b/B14/README.md) | Dot Product and Matrix Multiplication, Built From Nothing | D28 | 15 pts | `track-b` |
| [B15](track-b/B15/README.md) | Matrices Are Functions | D29 | 15 pts | `track-b` |
| [B16](track-b/B16/README.md) | Composition, Determinants, and the Visualizer | D30 | 15 pts | `track-b` |
| [B17](track-b/B17/README.md) | From One Variable to Many: Derivatives and Optimization | D31 | 15 pts | `track-b` |
| [B18](track-b/B18/README.md) | Gradients and Gradient Descent | D32–D33 | 15 pts | `track-b` |
| [B19](track-b/B19/README.md) | micrograd I: The Value Class and the Forward Pass | D34–D35 | 15 pts | `track-b` |
| [B20](track-b/B20/README.md) | micrograd II: Topological Sort | D36–D37 | 15 pts | `track-b` |
| [B21](track-b/B21/README.md) | micrograd III: The Backward Pass | D38–D39 | 15 pts | `track-b` |
| [B22](track-b/B22/README.md) | ★ MILESTONE B1: A Working Autograd Engine and a Trained Net | D41–D43 | 15 pts | `track-b` |
| [B23](track-b/B23/README.md) | Probability, Surprise, and Information | D44–D45 | 15 pts | `track-b` |
| [B25](track-b/B25/README.md) | Float Week (Math Clinic and Resubmission) | D40 | 15 pts | `track-b` |
| [C01](capstone/C01/README.md) | Capstone Declaration | D85–D86 | 15 pts | `capstone` |
| [C02](capstone/C02/README.md) | Proposal & Eval Plan (GATE) | D87–D88 | 15 pts | `capstone` |
| [C03](capstone/C03/README.md) | Sprint 1 | D89–D91 | 15 pts | `capstone` |
| [C04](capstone/C04/README.md) | Sprint 2 + Peer Code Review | D92–D94 | 15 pts | `capstone` |
| [C05](capstone/C05/README.md) | Sprint 3 (SKELETON GATE) | D95–D97 | 15 pts | `capstone` |
| [C06](capstone/C06/README.md) | Sprint 4 (Last Feature Week) | D98–D100 | 15 pts | `capstone` |
| [C07](capstone/C07/README.md) | CODE FREEZE | D101–D103 | 15 pts | `capstone` |
| [C08](capstone/C08/README.md) | Rehearsal & Documentation | D104–D105 | 15 pts | `capstone` |
| [C09](capstone/C09/README.md) | Public Symposium (FINAL) | D106 | 15 pts | `capstone` |

Not in this repo: A02, A24, A27, A29 (cut from the calendar) and the Optional / Stretch appendix; the Blackbaud import tooling; the student log repo scripts.
