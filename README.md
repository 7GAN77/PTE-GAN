# PTE-GAN

**PTE-GAN: A Progressive Thickness Expansion Generative Adversarial Network for Three-Dimensional Reconstruction of Multi-Category Geological Structures**

This repository contains the source code used to implement the progressive thickness-expansion workflow. A 2-D categorical conditioning profile is expanded stage by stage (1→3→5→…→25 slices). The implementation uses a 2-D U-Net generator, a PatchGAN discriminator, categorical one-hot conditioning, a known/unknown mask, relative thickness position and expansion-direction encoding. Training combines known/unknown cross-entropy, generalized Dice, feature matching and hinge adversarial terms; stochastic realizations are supported through bottleneck Gaussian perturbation, MC dropout and probabilistic sampling during cascade generation.

## Repository contents

```text
PTE-GAN/
├── README.md
├── requirements.txt
├── quick_test.py
├── make_example_data.py
├── example/
│   └── example_data.npy
├── prepare_condition_volume.py
├── run_train_all_stages_1to25.py
├── generate_cascade_228_7class.py
├── train_pairs_1to3.py
├── train_pairs_3to5.py
├── ...
├── train_pairs_23to25.py
├── dataset/
│   └── README.md
└── Ti/
    └── README.md
```

The repository contains individual source files rather than a compressed code archive.

## Environment

The manuscript experiments were developed with Python 3.9.23 and PyTorch 2.5.1 on an NVIDIA RTX 6000 24-GB GPU. The quick test is intentionally small and runs on CPU.

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

`tvtk`/Mayavi is optional and is needed only for VTK export. NumPy outputs do not require it.

## Quick test (recommended first)

The included example volume is generated specifically for software testing and is **not** one of the paper's experimental datasets. The test imports the actual `train_pairs_1to3.py` generator/discriminator and performs one small 1→3 adversarial optimization step using the same conditioning representation and loss family as the main code.

Run:

```bash
python quick_test.py --device cpu
```

Expected final message:

```text
[PTE-GAN QUICK TEST PASSED]
```

Outputs are written to:

```text
quick_test_output/predicted_1to3_slab.npy
quick_test_output/metrics.json
```

To regenerate the small example data:

```bash
python make_example_data.py --out ./example/example_data.npy
```

The quick test verifies code execution only; it is not intended to reproduce the paper-scale quantitative results.

## Seven-class example: formal training workflow

Place the preprocessed categorical reference model at:

```text
dataset/diceng_228_228_228_zyx_change_xiangsu.npy
```

The array order is `(Z, Y, X)`, and categorical values are `1..7`; `0` is reserved for unknown conditioning cells.

Prepare the 16 sparse conditioning profiles used by the current seven-class example:

```bash
python prepare_condition_volume.py
```

Check the 12-stage training sequence without starting training:

```bash
python run_train_all_stages_1to25.py --dry-run
```

Train all stages:

```bash
python run_train_all_stages_1to25.py
```

The stages are trained independently and then used sequentially during inference. Checkpoints are written to the stage-specific `checkpoints_*_cgan/` directories.

## Cascade generation

After all 12 stage checkpoints are available, generate stochastic realizations with:

```bash
python generate_cascade_228_7class.py --device cuda:0 --num_realizations 100
```

For a single realization per test profile:

```bash
python generate_cascade_228_7class.py --device cuda:0 --num_realizations 1
```

The current seven-class generation script expects the conditioning volume and split file under `./Ti/` using the default names documented in the script. Use `--help` to override paths and sampling parameters.

## Data note

The small `example/example_data.npy` file is distributed only to make the repository immediately testable. It is procedurally generated and does not replace the datasets used in the manuscript. Availability of the manuscript datasets should follow the paper's Data Availability statement and the permissions associated with the original data source.

## Reproducibility notes

- Volumes are stored in `(Z, Y, X)` order.
- Known categorical labels are `1..K`; `0` denotes unknown conditioning cells.
- Each thickness-expansion stage adds one slice on each side of the current slab.
- The full training workflow is GPU-oriented; the quick test is CPU-friendly.
- Random seeds are exposed where relevant, but stochastic sampling is intentionally used to produce multiple realizations.

## License

No license is imposed by this template. Add the open-source license approved by the authors/institution before public release if desired.
