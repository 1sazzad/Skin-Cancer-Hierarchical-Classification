# Phase 00 Completion Checklist

## Purpose

This checklist confirms that the project foundation, scope, architecture, governance, and reproducibility rules are complete before dataset acquisition, auditing, preprocessing, or model development begins.

Phase 01 must not begin until every mandatory Phase 00 item is completed or explicitly deferred with justification.

---

# 1. Project Directory and Repository

* [x] Permanent local project directory created
* [x] Project folder structure created
* [x] Git repository initialised
* [x] Git branch configured
* [x] Remote repository connected
* [x] `.gitignore` configured
* [x] Raw data directories excluded from Git
* [x] Model checkpoints excluded from Git
* [x] Experiment run outputs excluded from Git
* [x] Empty architecture folders preserved using `.gitkeep`
* [x] Project root kept clean
* [x] Local directory confirmed as permanent source of truth

Permanent project directory:

```text
F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification
```

---

# 2. Project Architecture

* [x] `configs/` created
* [x] `data/` created
* [x] `docs/` created
* [x] `experiments/` created
* [x] `models/` created
* [x] `notebooks/` created
* [x] `reports/` created
* [x] `scripts/` created
* [x] `src/` created
* [x] `tests/` created
* [x] Raw and processed data directories separated
* [x] External datasets separated from development datasets
* [x] Reusable code separated from notebooks
* [x] Generated artifacts separated from source code
* [x] Meaningful naming rules established
* [x] Working-directory cleanliness rule established

---

# 3. Project Charter

File:

```text
docs/00_project_charter.md
```

* [x] Working title defined
* [x] Research problem defined
* [x] Proposed solution defined
* [x] Primary research contribution defined
* [x] Dataset responsibilities summarised
* [x] Research boundaries defined
* [x] Clinical-claim limitations defined
* [x] Source-of-truth rule defined
* [x] Scope-control rule defined
* [x] Current project phase recorded

---

# 4. Scope Lock

File:

```text
docs/01_scope_lock.md
```

* [x] Stage 1 task defined
* [x] Stage 1 classes defined
* [x] Stage 1 routing rule defined
* [x] Stage 2 task defined
* [x] Stage 2 classes defined
* [x] Oracle-gate evaluation required
* [x] Predicted-gate evaluation required
* [x] Stage 3 target identified
* [x] Stage 3 feasibility condition defined
* [x] Stage 3 fallback rule defined
* [x] Dataset roles locked
* [x] Data modality locked
* [x] Flat baseline required
* [x] Fair-comparison rule defined
* [x] Evaluation priorities locked
* [x] Model-selection rules locked
* [x] Leakage-control rules locked
* [x] Preprocessing rules locked
* [x] Reproducibility rules locked
* [x] Naming convention locked
* [x] Non-goals locked
* [x] Scope-change procedure defined

---

# 5. Research Questions and Hypotheses

File:

```text
docs/02_research_questions.md
```

* [x] Flat versus hierarchical research question defined
* [x] Error-propagation research question defined
* [x] Imbalance-aware learning research question defined
* [x] External-generalisation research question defined
* [x] Stage 3 feasibility research question defined
* [x] Model-efficiency research question defined
* [x] Calibration research question defined
* [x] Explainability research question defined
* [x] Primary metrics defined
* [x] Secondary metrics defined
* [x] Statistical-reporting expectations defined
* [x] Research success criteria defined
* [x] Negative or mixed results accepted as valid outcomes

---

# 6. Dataset Roles and Governance

File:

```text
docs/03_dataset_roles.md
```

* [x] ISIC 2019 assigned as primary development dataset
* [x] EMB assigned to Stage 3 feasibility and modelling
* [x] HIBA assigned as mandatory external evaluation
* [x] MRA-MIDAS assigned as optional external evaluation
* [x] ISIC 2019 permitted uses defined
* [x] EMB feasibility requirements defined
* [x] HIBA prohibited uses defined
* [x] Zero-shot HIBA evaluation rule defined
* [x] MRA-MIDAS activation conditions defined
* [x] Raw and derived data separation defined
* [x] Dataset manifest requirements defined
* [x] Dataset registry requirements defined
* [x] Label-mapping governance defined
* [x] Exclusion governance defined
* [x] Dataset checksum rules defined
* [x] Leakage-prevention requirements defined
* [x] External-evaluation contamination rule defined
* [x] Dataset naming rules defined

---

# 7. Reproducibility Protocol

File:

```text
docs/04_reproducibility_protocol.md
```

* [x] Local and Azure responsibilities defined
* [x] Azure artifact-return rule defined
* [x] Reproducible experiment workflow defined
* [x] Run identifier format defined
* [x] Experiment configuration naming defined
* [x] Standard run-directory structure defined
* [x] Required run metadata defined
* [x] Random-seed policy defined
* [x] Initial seed set defined
* [x] Determinism policy defined
* [x] Environment-recording requirements defined
* [x] Git reproducibility rules defined
* [x] Dataset-manifest hash requirement defined
* [x] Split-manifest hash requirement defined
* [x] Checkpoint naming defined
* [x] Prediction-file requirements defined
* [x] Metric-storage requirements defined
* [x] Experiment registry requirements defined
* [x] Run-status values defined
* [x] Failure-documentation rules defined
* [x] Model-selection protocol defined
* [x] Calibration protocol defined
* [x] Statistical-comparison protocol defined
* [x] Figure and table naming defined
* [x] Notebook governance defined
* [x] Working-directory cleanliness defined
* [x] Session-handover process defined
* [x] Backup policy defined
* [x] Minimum reproducible experiment requirements defined

---

# 8. Risk Register

File:

```text
docs/05_risk_register.md
```

* [x] Risk likelihood scale defined
* [x] Risk impact scale defined
* [x] Risk status values defined
* [x] Stage 3 feasibility risk recorded
* [x] Class-imbalance risk recorded
* [x] Patient and lesion leakage risk recorded
* [x] External label incompatibility risk recorded
* [x] Hierarchical error-propagation risk recorded
* [x] Misleading overall accuracy risk recorded
* [x] Explainability-validity risk recorded
* [x] Duplicate-image risk recorded
* [x] External contamination risk recorded
* [x] Scope-creep risk recorded
* [x] GPU availability and cost risk recorded
* [x] Azure artifact-loss risk recorded
* [x] Naming and directory-clutter risks recorded
* [x] Single-seed risk recorded
* [x] Unfair comparison risk recorded
* [x] Test-feedback leakage risk recorded
* [x] External domain-shift risk recorded
* [x] Model-overconfidence risk recorded
* [x] Time-limit risk recorded
* [x] Dataset licence risk recorded
* [x] Clinical-label interpretation risk recorded
* [x] Corrupted-file risk recorded
* [x] Efficiency-comparison risk recorded
* [x] Clinical-overclaiming risk recorded
* [x] Risk-review procedure defined
* [x] Triggered-risk template defined

---

# 9. Decision Log

File:

```text
docs/06_decision_log.md
```

* [x] Decision identifier format defined
* [x] Decision status values defined
* [x] Decision template defined
* [x] Image-only scope decision recorded
* [x] Conditional hierarchy decision recorded
* [x] Dataset-role decision recorded
* [x] HIBA zero-shot decision recorded
* [x] Stage 3 feasibility decision recorded
* [x] Local source-of-truth decision recorded
* [x] Meaningful naming decision recorded
* [x] Clean architecture decision recorded
* [x] Validation-only model-selection decision recorded
* [x] Multi-metric evaluation decision recorded
* [x] Next decision identifier reserved

Next decision identifier:

```text
D-011
```

---

# 10. Configuration Files

The following files must be completed before Phase 00 closes:

```text
configs/project.yaml
configs/dataset_registry.yaml
```

## Project Configuration

* [ ] Project identifier defined
* [ ] Current phase recorded
* [ ] Local source-of-truth path recorded
* [ ] Pipeline stages recorded
* [ ] Stage classes recorded
* [ ] Dataset roles recorded
* [ ] Evaluation metrics recorded
* [ ] Seed policy recorded
* [ ] Model-selection restrictions recorded
* [ ] Naming conventions recorded

## Dataset Registry

* [ ] ISIC 2019 registry entry created
* [ ] EMB registry entry created
* [ ] HIBA registry entry created
* [ ] MRA-MIDAS registry entry created
* [ ] Dataset project identifiers standardised
* [ ] Dataset local paths recorded
* [ ] Dataset manifest paths recorded
* [ ] Dataset checksum paths recorded
* [ ] Dataset roles recorded
* [ ] Dataset audit statuses recorded
* [ ] Source URLs left pending until verified
* [ ] Licence information left pending until verified

---

# 11. Dataset Manifest Schema

File:

```text
data/manifests/manifest_schema.csv
```

* [ ] Manifest schema created
* [ ] Dataset name field included
* [ ] Image identifier field included
* [ ] Image path field included
* [ ] Patient identifier field included
* [ ] Lesion identifier field included
* [ ] Original label field included
* [ ] Stage 1 mapped label field included
* [ ] Stage 2 mapped label field included
* [ ] Stage 3 mapped label field included
* [ ] Split field included
* [ ] Inclusion field included
* [ ] Exclusion-reason field included
* [ ] Image checksum field included
* [ ] Source-reference field included

---

# 12. Experiment Registry

File:

```text
experiments/experiment_registry.csv
```

* [ ] Experiment registry created
* [ ] Run identifier column included
* [ ] Phase column included
* [ ] Research-stage column included
* [ ] Dataset column included
* [ ] Model column included
* [ ] Variant column included
* [ ] Seed column included
* [ ] Status column included
* [ ] Git commit column included
* [ ] Configuration path column included
* [ ] Dataset-manifest column included
* [ ] Dataset-manifest hash column included
* [ ] Split-manifest column included
* [ ] Split-manifest hash column included
* [ ] Checkpoint-selection metric column included
* [ ] Checkpoint path column included
* [ ] Primary metric column included
* [ ] Primary value column included
* [ ] Run directory column included
* [ ] Notes column included

---

# 13. Session Handover Template

File:

```text
reports/session_handover_template.md
```

* [ ] Session date field included
* [ ] Phase and task field included
* [ ] Completed-work section included
* [ ] Modified-files section included
* [ ] Decision section included
* [ ] Evidence and results section included
* [ ] Problems and risks section included
* [ ] Exact next-task section included
* [ ] Required-command section included

---

# 14. README Review

File:

```text
README.md
```

* [ ] Project title recorded
* [ ] Project summary recorded
* [ ] Current phase recorded
* [ ] Pipeline stages summarised
* [ ] Dataset roles summarised
* [ ] Project architecture described
* [ ] Local setup instructions included
* [ ] Environment setup placeholder included
* [ ] Dataset setup placeholder included
* [ ] Experiment execution placeholder included
* [ ] Reproducibility rules referenced
* [ ] Clinical-use disclaimer included
* [ ] Licence section included or marked pending
* [ ] Citation section included or marked pending

---

# 15. Git Verification

Before closing Phase 00, run:

```powershell
git status
git log --oneline --decorate -10
git remote -v
```

Expected conditions:

* [ ] Working tree is clean
* [ ] Current branch is `main`
* [ ] Remote repository is configured
* [ ] All Phase 00 documents are committed
* [ ] All Phase 00 commits are pushed
* [ ] No dataset files are tracked
* [ ] No checkpoints are tracked
* [ ] No generated experiment outputs are tracked
* [ ] No secrets are tracked
* [ ] No temporary files exist in the project root

---

# 16. Phase 00 Exit Criteria

Phase 00 is complete only when all of the following are true:

* [ ] Project architecture is complete
* [ ] Git repository is clean
* [ ] Scope is frozen
* [ ] Research questions are frozen
* [ ] Dataset responsibilities are frozen
* [ ] Reproducibility protocol is active
* [ ] Risk register is active
* [ ] Decision log is active
* [ ] Project configuration is complete
* [ ] Dataset registry is complete
* [ ] Manifest schema is complete
* [ ] Experiment registry is complete
* [ ] Session handover template is complete
* [ ] README is updated
* [ ] Final Phase 00 verification is committed and pushed

---

# 17. Current Phase 00 Status

## Completed

* Project directory
* Clean architecture
* Git initialisation
* Remote repository connection
* `.gitignore`
* Project charter
* Scope lock
* Research questions
* Dataset roles
* Reproducibility protocol
* Risk register
* Decision log

## Remaining

1. `configs/project.yaml`
2. `configs/dataset_registry.yaml`
3. `data/manifests/manifest_schema.csv`
4. `experiments/experiment_registry.csv`
5. `reports/session_handover_template.md`
6. Final `README.md` update
7. Final Git verification
8. Phase 00 completion commit

## Current Status

**Phase 00 is in progress.**

The next task is to complete:

```text
configs/project.yaml
```
