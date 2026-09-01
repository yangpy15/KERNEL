# KERNEL: A Knowledge-Enhanced Retrieval and Networked Evidence-Linking Framework for Laboratory Test Recommendation

KERNEL is a knowledge-guided clinical decision-support framework for patient-specific laboratory test recommendation in emergency departments. Inspired by emergency physicians’ diagnostic reasoning, KERNEL links diagnostic hypotheses to laboratory selection through knowledge graph–enhanced evidence retrieval. It integrates institutional ordering patterns and similar patient cases to generate evidence-grounded recommendations and clinician-facing explanations.

![KERNEL framework](image/figure1.png)

## Requirements

KERNEL requires Python 3.10 or later. The main dependencies are:

- PyTorch
- Transformers (Hugging Face)
- Accelerate
- Pandas
- NumPy
- scikit-learn
- openpyxl
- tqdm

Install the core dependencies with:

```bash
pip install torch transformers accelerate pandas numpy scikit-learn openpyxl tqdm
```

## Repository Structure

```text
KERNEL/
├── src/
│   ├── run_kernel.py       # Main KERNEL inference pipeline
│   ├── prompts.py          # Prompt templates for every LLM stage
│   ├── inference_utils.py  # EOP, retrieval, and Transformers utilities
│   ├── common.py           # Shared text and progress utilities
│   ├── train_diagnosis.py  # Stage 1 cross-encoder training entry point
│   ├── predict_diagnosis.py # Stage 1 ICD-block prediction entry point
│   ├── diagnosis_model.py  # Stage 1 training and prediction functions
│   ├── retrieval.py        # Frozen text encoder for embedding retrieval
│   └── evaluate.py         # Recommendation evaluation
└── data/
    ├── icd10_titles.csv           # Mapped Stage 1 label vocabulary
    ├── icd2test_tag.csv           # ICD block-to-test-tag mapping
    ├── test_tag_list.xlsx         # laboratory-category vocabulary
    ├── MC-MED_test_tag.xlsx       # Reference MC-MED laboratory-name mapping
    ├── MIMIC-IV_test_tag.xlsx     # Reference MIMIC-IV laboratory-name mapping
    ├── MedlinePlus_test_tag.xlsx  # Reference MedlinePlus laboratory-name mapping
    └── MedlinePlus_kg.json        # MedlinePlus rationale evidence by tag
```

The `src/` directory contains the reusable functions and complete pipeline, while `src/run_kernel.py` is the executable mainline entry point. Clinical input and generated-output paths are supplied by the user at runtime; they are not embedded in the implementation.

The published method settings are centralized in `KERNEL_SETTINGS` near the top of `src/run_kernel.py`.

The exact system, first-pass recommendation, reassessment, final-rationale, and response-repair prompts are centralized in `src/prompts.py`. The inference code imports and renders these templates directly.

### How to Use the Python Files

| File                         | Usage                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `src/run_kernel.py`        | Main executable for building EOP and retrieval evidence, running KERNEL inference, and saving recommendations.       |
| `src/train_diagnosis.py`   | Train the Stage 1 cross-encoder from prepared patient contexts and ICD-block labels.                                |
| `src/predict_diagnosis.py` | Generate ranked Stage 1 ICD-block predictions.                                                                        |
| `src/diagnosis_model.py`   | Shared Stage 1 training and prediction implementation.                                                                |
| `src/evaluate.py`          | Run after inference to calculate recommendation metrics.                                                             |
| `src/prompts.py`           | View or edit the exact prompts used at each LLM stage. This file is imported automatically and is not run directly. |
| `src/inference_utils.py`   | Shared EOP, retrieval, and Transformers inference functions imported by`run_kernel.py`.                            |
| `src/retrieval.py`         | Text-encoder and retrieval functions imported by the main program.                                                   |
| `src/common.py`            | Shared text-processing and progress utilities.                                                                        |

## Step 1: Prepare the Data

Users must complete de-identification, `patient_full_context` construction, ICD-block mapping, and laboratory-category mapping for their own data before running the released programs.

| Input                       | Required columns                                                        |
| --------------------------- | ----------------------------------------------------------------------- |
| Stage 1 training/validation | `visit_row_id`, `patient_full_context`, `gold_icd_codes`          |
| Mainline training           | All Stage 1 columns plus`CC`, `gold_icd_titles`, `gold_test_tags` |
| Evaluation visits           | `visit_row_id`, `patient_full_context`, `CC`                      |
| Evaluation gold             | `visit_row_id`, `gold_test_tags`                                    |

Multiple ICD blocks or laboratory categories must be separated by semicolons, and `gold_icd_codes` must align one-to-one with `gold_icd_titles`.

`data/icd10_titles.csv` file is pre-provided and contains the definitions of the 273 labels used in Stage 1. The laboratory mapping workbooks in `data/` are reference files only and are not loaded automatically.

## Step 2: Training

Stage 1 can be run in either of two ways:

1. Use one of our pretrained Stage 1 cross-encoder checkpoints for MC-MED and MIMIC-IV: [download](https://drive.google.com/drive/folders/1ILUX9G8oWLEvJg2Yyjo7u8haLXWEoNFq?usp=drive_link).
2. Train a new checkpoint from your own prepared training and validation partitions.

The download folder contains `stage1_mcmed_cross_encoder.pt` and `stage1_MIMIC-IV_cross_encoder.pt`.

Choose the pretrained checkpoint whose training domain matches the experiment.
For example, generate Top-15 predictions with the MC-MED-trained checkpoint as follows:

```bash
python src/predict_diagnosis.py \
  --input-path PATH_TO_EVALUATION_VISITS.csv \
  --checkpoint-path checkpoints/stage1_mcmed_cross_encoder.pt \
  --top-k 15 \
  --max-length 512 \
  --output-path PATH_TO_ICD_PREDICTIONS.csv
```

To use the MIMIC-IV-trained model, change `--checkpoint-path` to `checkpoints/stage1_MIMIC-IV_cross_encoder.pt`.

Alternatively, train a new Stage 1 checkpoint:

```bash
python src/train_diagnosis.py \
  --train-path PATH_TO_TRAIN.csv \
  --validation-path PATH_TO_VALIDATION.csv \
  --icd-label-path data/icd10_titles.csv \
  --checkpoint-path PATH_TO_SAVE_CHECKPOINT.pt \
  --max-length 512 \
  --epochs 6

python src/predict_diagnosis.py \
  --input-path PATH_TO_EVALUATION_VISITS.csv \
  --checkpoint-path PATH_TO_CHECKPOINT.pt \
  --top-k 15 \
  --max-length 512 \
  --output-path PATH_TO_ICD_PREDICTIONS.csv
```

Training uses `patient_full_context` as patient text and the already-mapped `gold_icd_codes` column as Stage 1 supervision. `--icd-label-path` defaults to the bundled 273-block vocabulary in `data/icd10_titles.csv`.

If compatible Stage 1 predictions are already available, training and prediction can be skipped by passing their path through `--review-path`.

`src/run_kernel.py` reads the prepared training partition supplied through `--train-gold-path` and builds EOP and the retrieval indexes automatically before inference begins.

## Step 3: Inference

Run `src/run_kernel.py` with the Stage 1 predictions, evaluation visits, and authorized training partition. The program builds the required evidence, loads the prompts from `src/prompts.py`, runs recommendation and reassessment, and writes the final recommendation and rationale.

### Run KERNEL Inference

```bash
python src/run_kernel.py \
  --review-path PATH_TO_ICD_PREDICTIONS.csv \
  --eval-path PATH_TO_EVALUATION_VISITS.csv \
  --train-gold-path PATH_TO_GOLD_TRAIN.csv \
  --output-path PATH_TO_KERNEL_PREDICTIONS.csv
```

An external diagnosis knowledge graph is optional. To enable diagnosis-concept expansion and diagnosis-aware similar-case retrieval, provide both related files:

```bash
python src/run_kernel.py \
  --review-path PATH_TO_ICD_PREDICTIONS.csv \
  --eval-path PATH_TO_EVALUATION_VISITS.csv \
  --train-gold-path PATH_TO_GOLD_TRAIN.csv \
  --diagnosis-kg-path PATH_TO_DIAGNOSIS_KG.csv \
  --concept-metadata-path PATH_TO_CONCEPT_METADATA.csv \
  --output-path PATH_TO_KERNEL_PREDICTIONS.csv
```

The two diagnosis-KG arguments must be supplied together. When they are omitted, KERNEL skips diagnosis-KG expansion and diagnosis-aware retrieval while retaining Stage 1 predictions, ICD block-to-test-tag knowledge, EOP, patient-context retrieval, reassessment, evidence completion, and final-rationale generation.

### Automatic Evidence Preparation During Inference

Before inference, `src/run_kernel.py` builds EOP and retrieval indexes in memory from the prepared training partition supplied through `--train-gold-path`. It reads `gold_icd_codes`, `gold_icd_titles`, and `gold_test_tags` directly and does not perform raw diagnosis or laboratory-name mapping. No separate EOP or retrieval-training command is required. The training partition must remain separate from validation, test, and external evaluation cohorts.

### Generate the Final Rationale

Final-rationale generation is part of `src/run_kernel.py` and does not require a separate script. After `recommended_test_tags` has been finalized, KERNEL runs one additional LLM generation pass and writes the resulting clinician-facing rationale to `final_explanation`. The rationale is grounded in the patient context, final test categories, and available evidence, but it cannot change `recommended_test_tags`.

Rationale generation is enabled in `KERNEL_SETTINGS`. Only the resulting clinician-facing rationale is retained in the final output.

### Output Format

KERNEL writes only the identifier, final recommendation, and final rationale to the path supplied through `--output-path`.

| Output field              | Description                                | Example                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `visit_row_id`          | Visit identifier.                          | `visit_0001`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `recommended_test_tags` | Final laboratory-category recommendations. | Cell Count;Electrolytes;Renal Function;Liver Function;Coagulation;Microbiology/Infection;Infection Markers;Blood Gas&Acid-Base;Cardiovascular                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `final_explanation`     | Final clinician-facing rationale.          | The patient's acute neurological problem and high blood pressure warrant a comprehensive evaluation of renal function, electrolyte balance, and liver function. Given the patient's history of end-stage renal disease and type 2 diabetes, it is essential to assess renal function and electrolyte balance to rule out complications such as hyperkalemia or hypokalemia. Additionally, liver function tests are necessary to evaluate potential liver damage or dysfunction. Coagulation studies are also recommended to assess the risk of bleeding or clotting disorders. Microbiology and infection markers are necessary to evaluate the risk of infection, particularly in the context of altered mental status. Blood gas and acid-base analysis will help assess the patient's acid-base balance and respiratory status. Cardiovascular and metabolic tests are added to evaluate the patient's cardiovascular and metabolic status, given the patient's history of hypertension and diabetes. |

### MedlinePlus Final-Rationale Evidence

The mainline automatically loads the bundled structured MedlinePlus laboratory resource from `data/MedlinePlus_kg.json`; no MedlinePlus enable or decision-mode argument is required. After `recommended_test_tags` has been finalized, the program selects MedlinePlus evidence only for those final tags and supplies it to the final clinician-facing rationale call, which is enabled by default.

The `data/MedlinePlus_test_tag.xlsx` is a reference laboratory-name mapping and is not used at runtime; it is distinct from the structured MedlinePlus JSON used for final-rationale evidence.

## Evaluation

Use `evaluate.py` to calculate patient-level laboratory-category metrics on lab-positive visits and overall patient-level label-set metrics including `no_lab_recommended`. The gold file must already contain `gold_test_tags`; evaluation does not map raw laboratory names. An empty `recommended_test_tags` set is represented as `no_lab_recommended` in the overall label-set evaluation.

```bash
python src/evaluate.py \
  --prediction-path PATH_TO_KERNEL_PREDICTIONS.csv \
  --gold-path PATH_TO_GOLD_TEST.csv \
  --output-summary-path PATH_TO_METRICS.csv \
  --output-per-visit-path PATH_TO_PER_VISIT_METRICS.csv
```

## Data Availability and Intended Use

The clinical datasets (MC-MED and MIMIC-IV) used by KERNEL are restricted-access resources and are not redistributed here.
KERNEL is research software for clinical decision support. Its recommendations must not be interpreted as autonomous laboratory orders and do not replace qualified clinical judgment.

## Citation

This is the code repository for our KERNEL paper. Please cite us if you use this repository:

```bibtex
@article{kernel,
  title={KERNEL: A Knowledge-Enhanced Retrieval and Networked Evidence-Linking Framework for Laboratory Test Recommendation},
  author={Pei-Ying Yang, Chien Chin Chen, Ting-Yun Huang, and Yung-Chun Chang},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}
```

The complete citation will be added after formal publication.
