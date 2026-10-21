# CV Selection Bootcamp: learning from noisy labels (CIFAR-10N)

Everything runs on Google Colab with an A100 GPU. Seed 42 for every run; cuDNN autotuning is on, so reruns can differ in the last decimals.

## Setup

1. Put a folder `cv_bootcamp` in Google Drive (`MyDrive/cv_bootcamp`) containing `data/` (the provided files), `model.py`, `check_format.py`, `results_template.json` and `code/cvlib.py`.
2. Open each notebook in Colab, choose an A100 runtime and use Runtime > Run all, in the order below. Each notebook mounts Drive and reads and writes inside `MyDrive/cv_bootcamp` (`work/` for intermediate files, `predictions/` for the submission files). Finished training runs are saved in `work/` and are loaded instead of retrained on a rerun.

## Notebooks

| Notebook | Task | Writes | Time (A100) |
|---|---|---|---|
| `01_task1_noise_and_clean_ref_final.ipynb` | Task 1: noise rate, confusion matrix, clean-label reference | `t1_clean_ref.csv` | 5 min |
| `02_task1_noisy_ce_task2_dynamics_final.ipynb` | Task 1 noisy CE run, Task 2 dynamics and early stopping | `t1_noisy_ce.csv`, `t2_dynamics.csv`, `t2_early_stop.csv` | 6 min |
| `03_task3_noise_scores_final.ipynb` | Task 3: 5-fold cross-validated noise scores and the clean-label check | `t3_noise_scores.csv` | 25 min |
| `04_task4_robust_training_final.ipynb` | Task 4: relabel, SCE, co-teaching | `t4_relabel.csv`, `t4_sce.csv`, `t4_coteaching.csv` | 20 min |
| `05_task5_synthetic_noise_final.ipynb` | Task 5: CE and co-teaching on synthetic labels, comparison | `t5_baseline_synth.csv`, `t5_best_synth.csv` | 45 min |
| `06_package_submission_final.ipynb` | results.json, report.md, this README, the zip | `<rollno>_cv_bootcamp.zip` | 1 min |

`cvlib.py` holds the shared code: data loading, GPU augmentation (random crop with 4-pixel padding, horizontal flip), the training loop (SGD with Nesterov momentum 0.9, lr 0.1 with per-step cosine decay, weight decay 5e-4, batch 128, 60 epochs, bf16 autocast), symmetric cross-entropy, co-teaching and the prediction writers. `model.py` is the provided ResNet-18, trained from scratch everywhere.

## Rules followed

* Clean labels are used only to train the Task 1 reference and to measure (confusion matrix, Task 2 curves, Task 3 check, clean validation accuracy printed next to results). Every early-stopping epoch, averaging window, method choice and checkpoint comes from noisy validation accuracy (synthetic validation accuracy in Task 5).
* Budget: 60 epochs per network; co-teaching uses two networks; Task 3 uses 5 runs. In Task 5 the co-teaching noise-rate estimate repeats the Task 3 cross-validation on the synthetic labels, and relabel and SCE are also run on synthetic labels only for the ranking in Q5 (not submitted).

Notebook code was written with help from an AI assistant (Claude) and checked and run by me.
