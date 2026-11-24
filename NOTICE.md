# Third-party notices

The code in this repository is licensed under **AGPL-3.0** (see `LICENSE`).
It also contains one evaluation module from the same project
(`src/evaluate_operating_points.py`), distributed under the same terms.

## Runtime dependency: Ultralytics

The code imports `ultralytics` (tested with version 8.4.198), which is
distributed under **AGPL-3.0**. Publishing code that imports it, or redistributing
modified Ultralytics code or derived weights, may carry AGPL obligations. Review
`https://github.com/ultralytics/ultralytics` and your institution's policy before
making the repository public or shipping weights.

The experiments were run with a project-built Ultralytics wheel. This repository
does **not** redistribute that wheel; `requirements.txt` pins the version string
only. The code path here uses standard Ultralytics APIs.

## Other dependencies

| package | license |
|---|---|
| PyTorch / torchvision | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache-2.0 |
| pycocotools | BSD-2-Clause |
| Matplotlib | PSF-based |
| Pillow | HPND |
| psutil | BSD-3-Clause |

Check the installed distributions for the authoritative license texts.
