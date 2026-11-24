# Push checklist

The tree is code-only: datasets, weights, runs, predictions, logs and generated
reports are excluded by `.gitignore`.

## 1. Before the first commit

- [x] `LICENSE` added: AGPL-3.0 (see `NOTICE.md` for third-party terms).
- [ ] Add author/contact or a `CITATION.cff` if the paper is public.
- [ ] Confirm `git status` contains no `data/`, `weights/`, `runs/`, `reports/`,
      `state/` or `*.pt` files.
- [ ] Confirm no machine-specific path remains:
      `grep -RIn "D:/" src scripts tests docs configs README.md`
      (expected: nothing).
- [ ] Run the CPU tests: `python -m unittest -v tests/test_relative_context.py`
      (expected: 8 tests, OK).
- [ ] Byte-compile everything: `python -m compileall -q src scripts tests`.

## 2. Create and push

```bash
git init -b main
git add .
git commit -m "RCBR-YOLO reference implementation"
# create the remote (choose one)
gh repo create <org>/RCBR-YOLO --public --source=. --remote=origin --push
# or
git remote add origin git@github.com:<org>/RCBR-YOLO.git
git push -u origin main
```

## 3. After pushing

- [ ] Add topics/keywords (object detection, YOLO26, defect detection).
- [ ] Add the paper/preprint link to `README.md` when available.
- [ ] Tag the snapshot: `git tag -a v0.1.0 -m "code snapshot" && git push --tags`.
- [ ] Optional CI: Python 3.12 + CPU-only `torch` install, run the unit tests and
      `compileall` (no GPU, no data needed).
- [ ] If weights are ever published: verify the AGPL implications and that the
      pickled checkpoints can resolve `cr_integration.CtxWrap` (the module names
      are intentionally kept).
- [ ] Re-check that generated `reports/` and `runs/` stay untracked after a local
      run.

## 4. What deliberately stays out

| item | reason |
|---|---|
| `data/`, annotations, images | not redistributable / too large |
| `weights/*.pt`, `runs/**/weights/*.pt` | size, license, and no need for code review |
| `runs/`, `reports/`, `state/` | local evidence and generated results |
| `preds/*.json` | large prediction dumps |

The code reproduces the pipeline; the numerical results are reported in the
paper, not in this repository.
