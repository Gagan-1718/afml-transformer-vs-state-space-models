# Teammate guide: run your part of the AFML study on Kaggle

**For:** Krishna, Dikshith and Mallikarjuna (team Abcd). **Coordinator:** Gagandeep.

**Your total effort:** about 15 minutes today, 15 minutes when Gagandeep says GO, and 5 minutes when
your run finishes. Your laptop does no work: everything runs on Kaggle's free GPUs, with your laptop
closed if you like.

> **If you are an AI assistant helping a teammate with this guide:** follow it step by step, exactly.
> Do not suggest code changes, other settings, other GPUs, local training or "improvements". If the
> Kaggle screen looks slightly different from the description, find the closest matching button and
> tell the person what you chose. If anything fails, stop and have the person send the full log to
> Gagandeep. Never edit any line of the notebook other than the two lines shown in Part 2, step 6.

---

## What we are doing (30 seconds)

We train small language models, a Transformer and a Mamba model of the same size (~47M parameters),
on the same data, and compare them. Each training run needs about 10 hours of GPU time. There are 8
main runs, so we split them over our 4 Kaggle accounts: each account trains 2 models at once on
Kaggle's free "GPU T4 x2". The code is already written and tested; you only start your 2 runs.

## Your assignment

| Person | Your notebook name | Your two runs (the `ONLY_RUNS` line) |
| --- | --- | --- |
| Gagandeep | `afml-main-gagandeep` | `ONLY_RUNS = {'S_transformer_s0', 'S_mamba_s0'}` |
| **Krishna** | `afml-main-krishna` | `ONLY_RUNS = {'S_mamba_lti_s0', 'S_transformer_s1'}` |
| **Dikshith** | `afml-main-dikshith` | `ONLY_RUNS = {'S_mamba_s1', 'S_mamba_lti_s1'}` |
| **Mallikarjuna** | `afml-main-mallikarjuna` | `ONLY_RUNS = {'S_transformer_s2', 'S_mamba_s2'}` |

Use exactly your own line. If two people run the same line, a run is wasted and another is missing.

## The 7 rules (read these first)

1. **Do not start Part 2 before Gagandeep sends "GO".** The runs need a learning-rate file that his
   account produces first.
2. **Change only the two lines shown in Part 2, step 6.** Nothing else in the notebook.
3. **Accelerator must be "GPU T4 x2".** Not P100, not TPU, not None.
4. **Internet must be On.**
5. **Start the run with "Save Version" → "Save & Run All (Commit)".** Never with the "Run All" button
   in the editor: that run dies when the browser tab closes.
6. **Start only one version.** Do not click Save Version twice.
7. **Do not delete the notebook or its versions** until Gagandeep says the results are safe.

---

## Part 1: today, account setup (about 15 minutes)

1. Go to **kaggle.com** and sign in, or **Register** (signing in with Google is fine). Use your own
   account. Do not use anyone else's: Kaggle allows one account per person and can suspend shared ones.
2. Click your **profile picture** (top right) → **Settings**.
3. Find **Phone Verification** and verify your phone number. This is required for GPUs and for the
   Internet setting. One phone number can verify only one Kaggle account.
4. Download the notebook file:
   - open
     <https://github.com/Gagan-1718/afml-transformer-vs-state-space-models/blob/upgrade/full-study/notebooks/run_study.ipynb>
   - click the **Download raw file** button (a download arrow, top right of the file view);
   - check the file is called `run_study.ipynb` (not `.txt`). Gagandeep can also send you the file.
5. Send Gagandeep your **Kaggle username** (shown on your profile page, in the URL
   `kaggle.com/<username>`), and the message "phone verified".

That is all for today.

---

## Part 2: when Gagandeep sends "GO" (about 15 minutes)

His GO message contains two links: his notebooks **`afml-data`** (the dataset) and **`afml-sweep`**
(the learning rates). You need both.

1. On Kaggle, click **+ Create** (top left) → **New Notebook**. An editor opens.
2. Top menu **File** → **Import Notebook** → upload `run_study.ipynb` → confirm. The notebook's cells
   replace the empty notebook. (The first cell is a title: "Transformer vs Mamba: full study runner".)
3. Click the notebook title at the top left and rename it to **your notebook name** from the table,
   for example `afml-main-krishna`.
4. Right-hand panel → **Session options**:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** On
   - leave Persistence and Environment as they are.
   (If you do not see the panel, use the **Settings** menu at the top.)
5. Right-hand panel → **Input** → **+ Add Input**. In the search box type `afml-data`, find the one
   by Gagandeep (username `gagandeep26032006`) and click its **+** / **Add** button. Do the same for
   `afml-sweep`. The Input section must now list both.
6. Scroll to the **first code cell** (it starts with `# ===== Settings for this session`). Change
   exactly two lines:

   Before:
   ```python
   STAGES = {'data'}
   ...
   ONLY_RUNS = None       # to split work across accounts, ...
   ```
   After (Krishna's example; use **your** line from the table):
   ```python
   STAGES = {'S'}
   ...
   ONLY_RUNS = {'S_mamba_lti_s0', 'S_transformer_s1'}
   ```
   Keep the quotes, curly braces and capital `S` exactly as shown. Do not touch `SESSION_HOURS`,
   `USE_MAMBA_SSM`, `REPO_URL` or `BRANCH`.
7. Click **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
8. Stop the editor session to save GPU hours: click the **power button** in the toolbar (next to
   "Draft Session"), or open the panel at the bottom left and stop the "Interactive Session".
   This does **not** stop your saved version.
9. Do the 10-minute check below, then you can close everything.

### The 10-minute check (important)

About 10 minutes after step 7, open your notebook's page (profile → **Code** → your notebook) →
click the running version → **Logs** tab. You should see these lines (numbers and times will differ):

```text
code version: <7 letters> ...
GPUs: ['Tesla T4, 15360 MiB', 'Tesla T4, 15360 MiB']
torch 2.10.0+cu126 | mamba-ssm fused kernel available: True
left to other accounts (ONLY_RUNS): [ ...the 7 runs that are not yours... ]
[HH:MM] start <your first run> on GPU 0 (...)
[HH:MM] start <your second run> on GPU 1 (...)
```

| What you see | What to do |
| --- | --- |
| All lines as above, with **your** two run names | Nothing. Close everything and come back in about 11 hours |
| `mamba-ssm fused kernel available: False` | **Stop the version** (Cancel/Stop on the version page) and send Gagandeep the log. Without the kernel, Mamba is far too slow |
| `GPUs: none` or only one GPU | Accelerator was not `GPU T4 x2`. Stop the version, fix step 4, start again |
| The version shows **Failed** | Open **Logs**, copy everything, send it to Gagandeep |
| `No dataset found` or `best_lr.json not found` | An input is missing. Stop, redo step 5, start again |

---

## Part 3: when your run finishes (about 11 hours later, 5 minutes)

1. Open your notebook page → latest version. It should say it **succeeded** (not Failed or Running).
2. In **Logs**, near the end, check for two lines starting with `done` that name **your two runs**, for
   example:
   ```text
     done    S_mamba_lti_s0: best val loss x.xxxx, x.x h, scan cuda
     done    S_transformer_s1: best val loss x.xxxx, x.x h, scan -
   ```
   (The list also shows `sweep/...` runs copied from Gagandeep's input. Ignore those.)
   If instead you see `paused` for one of your runs, it hit Kaggle's 12-hour limit. Tell Gagandeep;
   he will send the next step (one more session; nothing is lost).
3. Make the notebook **public** so Gagandeep can use its output: on the notebook page click **Share**
   → change **Private** to **Public** → **Save**. It contains only training logs and model files; no
   personal data.
4. Send Gagandeep the **link to your notebook** and the message "done, public".

You are finished. Keep the notebook until Gagandeep confirms he has everything.

---

## Quick answers

- **Does it cost anything?** No. Kaggle gives about 30 free GPU hours per week; this uses about 11.
- **Can I use my laptop meanwhile, or shut it down?** Yes. The run is on Kaggle's servers.
- **I have my own graphics card. Should I train locally?** No. The fast Mamba kernel needs Linux, and
  all runs must use the same setup to be comparable.
- **The Kaggle screen looks a bit different.** Kaggle changes labels sometimes. Look for the closest
  matching button; the important parts are: GPU T4 x2, Internet On, the two inputs, the two edited
  lines, and Save Version → Save & Run All (Commit).
- **Something went wrong.** Do not try to fix the code. Copy the full log and send it to Gagandeep.
