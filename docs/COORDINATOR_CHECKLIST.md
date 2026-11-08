# Coordinator checklist (Gagandeep only)

Teammates follow [TEAMMATE_GUIDE.md](TEAMMATE_GUIDE.md). This page is your own order of work and the two
messages to send. Every Kaggle run below is started the same way: **new notebook → File → Import
Notebook → `run_study.ipynb` → rename → Session options (GPU T4 x2 unless stated, Internet On) → add
inputs → edit the first code cell → Save Version → Save & Run All (Commit) → stop the editor session**.

Links you will need:

- Guide for teammates: <https://github.com/Gagan-1718/afml-transformer-vs-state-space-models/blob/upgrade/full-study/docs/TEAMMATE_GUIDE.md>
- Your notebooks: `https://www.kaggle.com/code/gagandeep26032006/<notebook-name>`

## Day 1 (today)

1. **Data.** Wait for `afml-data` Version 2 to succeed. Check its Output has
   `afml/data_fineweb_16k/meta.json`. If it failed, check the log.
2. **Make `afml-data` public:** notebook page → Share → Public → Save.
3. **Send Message 1** (below) to Krishna, Dikshith and Mallikarjuna.
4. **Smoke test** (about 1 hour). Only after the kernel fix is pushed.
   - Notebook name `afml-smoke`. Input: `afml-data`.
   - First cell: `STAGES = {'smoke'}`. Leave `ONLY_RUNS = None`.
   - 10-minute check in Logs: `GPUs:` lists two Tesla T4, and `mamba-ssm fused kernel available: True`.
   - When it succeeds, download `afml/results/smoke/smoke_report.md` from the Output tab and copy the
     last 40 lines of the log.
5. **Wait** until the training-token budget is set from the smoke numbers and pushed.
6. **Learning-rate sweep** (about 3 to 7 hours; fine overnight).
   - Notebook name `afml-sweep`. Input: `afml-data`.
   - First cell: `STAGES = {'sweep'}`. Leave `ONLY_RUNS = None`.
   - When it succeeds, the log ends with the best learning rates, and the Output has
     `afml/results/S/best_lr.json`. Check the last 40 lines of the log.
   - Make `afml-sweep` public.

## Day 2

7. When the sweep is confirmed fine, **send Message 2** (GO) to the three teammates.
8. **Start your own main run** the same way they do (guide, Part 2):
   - Notebook name `afml-main-gagandeep`. Inputs: `afml-data` and `afml-sweep`.
   - First cell: `STAGES = {'S'}` and `ONLY_RUNS = {'S_transformer_s0', 'S_mamba_s0'}`.
9. Collect each teammate's 10-minute-check screenshot. If anyone reports a problem, check their log.

## Day 3

10. Collect "done, public" and the notebook link from all three teammates. Check your own run also
    shows both `done` lines.
11. **Final session** (about 4 to 6 hours).
    - Notebook name `afml-final`. Inputs, all six: `afml-data`, `afml-sweep`, `afml-main-gagandeep`,
      `afml-main-krishna`, `afml-main-dikshith`, `afml-main-mallikarjuna`.
    - First cell: `STAGES = {'synthetic', 'eval', 'figures'}`. Leave `ONLY_RUNS = None`.
    - When it succeeds, the log's "Runs in this output" list shows 8 `done` S runs. Download
      `afml/results.zip` from the Output tab and put it in the project folder.

## Days 4 to 5

12. Fill in the report from `results/`. You and the team review it. Day 5 is a buffer for any rerun.

## GPU hours (free quota is about 30 per account per week)

| Account | Uses about |
| --- | --- |
| Gagandeep | smoke 1 h + sweep 3 to 7 h + main run 11 h + final 4 to 6 h = 19 to 25 h |
| Each teammate | 11 h |

---

## Message 1 (send today)

```text
Hi! For our AFML project I need your Kaggle account for ONE training run. Your total effort is
about 30 minutes spread over two days, and your laptop can be off while it runs (it's all on
Kaggle's free GPUs).

Guide (you can paste it into ChatGPT and ask it to walk you through it step by step):
https://github.com/Gagan-1718/afml-transformer-vs-state-space-models/blob/upgrade/full-study/docs/TEAMMATE_GUIDE.md

TODAY: do only Part 1 of the guide (Kaggle account, phone verification, download the notebook
file), then send me your Kaggle username.

Your row in the guide's table: Krishna = afml-main-krishna, Dikshith = afml-main-dikshith,
Mallikarjuna = afml-main-mallikarjuna.

Do NOT start any run until I send "GO".
```

## Message 2 (send on day 2, when the sweep is confirmed)

```text
GO! Please do Part 2 of the guide now:
https://github.com/Gagan-1718/afml-transformer-vs-state-space-models/blob/upgrade/full-study/docs/TEAMMATE_GUIDE.md

The two inputs to add (search for them in "Add Input"):
- afml-data:  https://www.kaggle.com/code/gagandeep26032006/afml-data
- afml-sweep: https://www.kaggle.com/code/gagandeep26032006/afml-sweep

Use YOUR OWN ONLY_RUNS line from the table. After about 10 minutes, do "The 10-minute check" and
send me a screenshot of the Logs. When it finishes (about 11 hours), do Part 3 and send me your
notebook link.
```
