# Poisson Neural Decoder for Visual Cortex Spike Data

This project uses population spike counts from the Allen Institute OpenScope
Psycode dataset to predict which natural image a mouse viewed. It compares a
baseline Poisson Naive Bayes decoder with an L2-regularized, probability-
calibrated version.

The analysis asks:

> Can visual-cortex population activity predict natural-image identity, and
> how reliable are the decoder's probability estimates?

## Repository contents

| File | Purpose |
| --- | --- |
| `poisson_image_identity_baseline.py` | Fits the original Poisson Naive Bayes decoder and reports accuracy, confidence, and a confusion matrix. |
| `poisson_image_identity_calibrated.py` | Adds L2-regularized Poisson rate estimation, a separate calibration set, power calibration, and detailed probability metrics. |

Keep both Python files in the same directory. The calibrated program imports
the shared NWB-processing functions from the baseline file.

## Dataset

The programs analyze an NWB file from
[DANDI:001417](https://dandiarchive.org/dandiset/001417), the OpenScope Psycode
study. The data contain Neuropixels spike times, brain-region information,
natural-image presentation labels, and an injection timestamp.

The preprocessing follows the official
[OpenScope Psycode documentation](https://alleninstitute.github.io/openscope_databook/psycode/).

The default analysis:

- Selects quality-controlled single units from `VISp` and `VISam`.
- Keeps repeated, non-omitted natural-image presentations.
- Separates trials before or after the injection timestamp.
- Counts spikes from 50 to 250 ms after image onset.
- Uses one spike count per selected neuron as the representation of each trial.

NWB files are not included in this repository because each recording is many
gigabytes. Download the required file separately from DANDI.

## How the decoder works

For each neuron \(i\) and image class \(c\), the baseline learns an expected
spike count \(\lambda_{ic}\) from the training trials:

\[
K_i \mid c \sim \operatorname{Poisson}(\lambda_{ic})
\]

It assumes neurons are conditionally independent given the image and combines
their evidence:

\[
P(c \mid K) \propto P(c)
\prod_i P(K_i \mid c)
\]

The predicted image is the class with the largest posterior probability.

The calibrated version adds two safeguards:

1. **L2 regularization** shrinks extreme image-specific firing-rate estimates,
   reducing sensitivity to noise in the training trials.
2. **Power calibration** learns one probability-softening parameter from a
   calibration set that is separate from both training and final testing.

The final test set remains untouched until the model and calibration parameter
have been fitted.

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment,
then install the required packages:

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn pynwb pandas
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn pynwb pandas
```

## Usage

### Baseline decoder

```bash
python poisson_image_identity_baseline.py \
    --nwb-path "/path/to/session_ecephys.nwb" \
    --segment pre
```

### Calibrated decoder

```bash
python poisson_image_identity_calibrated.py \
    --nwb-path "/path/to/session_ecephys.nwb" \
    --segment pre
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python poisson_image_identity_calibrated.py --nwb-path "D:\path\to\session_ecephys.nwb" --segment pre
```

Use `--segment post` to analyze post-injection trials. Display every available
option with:

```bash
python poisson_image_identity_calibrated.py --help
```

Important optional arguments include:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--window-start` | `0.050` | Start of the response window in seconds after image onset. |
| `--window-end` | `0.250` | End of the response window in seconds after image onset. |
| `--bin-size` | `0.005` | Spike-count bin width in seconds. |
| `--test-size` | `0.20` | Fraction reserved for final testing. |
| `--calibration-size` | `0.20` | Fraction reserved for calibration in the calibrated version. |
| `--l2-strength` | `0.01` | Regularization strength in the calibrated version. |
| `--confidence-bins` | `10` | Number of bins used for calibration diagnostics. |

## Outputs

The baseline reports:

- Training accuracy
- Test accuracy and balanced accuracy
- Mean predicted confidence
- Confidence minus accuracy
- Confusion matrix

The calibrated version additionally reports:

- Raw and calibrated probability results
- Log loss
- Multiclass Brier score
- Expected calibration error (ECE)
- Confidence-bin accuracy
- Top-1, top-3, and top-5 accuracy
- Credible-set coverage and average set size

## Example single-session result

For one pre-injection session from subject `760322`, 49 selected visual-cortex
units and 1,563 repeated-image trials produced the following results:

| Model | Top-1 accuracy | Balanced accuracy | Mean confidence |
| --- | ---: | ---: | ---: |
| Baseline Poisson Naive Bayes | 0.7732 | 0.7757 | 0.9143 |
| Regularized and calibrated decoder | 0.7508 | 0.7525 | 0.7149 |

The baseline classified images accurately but was overconfident: its mean
confidence exceeded accuracy by about 0.14. Calibration slightly reduced
top-1 accuracy while bringing confidence much closer to observed accuracy.
The calibrated model also achieved top-3 accuracy of 0.9010 and top-5 accuracy
of 0.9808.

These values describe one session and should not be treated as population-level
estimates. A complete study should repeat the analysis across animals, sessions,
and pre/post conditions.

## Interpretation

Above-chance decoding supports the conclusion that the recorded population
contains information that distinguishes the images. It does **not** demonstrate
that:

- The brain uses a Poisson Naive Bayes computation.
- These neurons cause image perception.
- The decoder reconstructs the original pixels.
- A result from one recording generalizes across mice or drug conditions.

The model predicts an image label from a neural population response; it does
not recreate the visual image itself.

## Reproducibility notes

- All data splits are stratified by image identity.
- Random seeds are fixed but can be changed through command-line arguments.
- Feature filtering is fitted inside the training pipeline.
- The calibrated version uses distinct training, calibration, and test sets.
- Run pre- and post-injection analyses separately before comparing performance.

## Citation and data use

If this project is used in research, cite the DANDI dataset, the OpenScope
Psycode study, and the relevant neural-decoding methodology. Review the dataset
page for its current citation and reuse requirements.
