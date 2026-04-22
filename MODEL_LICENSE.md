# Model and Artifact Publishing Notes

This repository is public. To keep the public tree safe to share, trained model weights and raw dataset archives should not be committed into git history.

## Public-safe default

- Publish source code, configs, metrics, and reproducible training/evaluation commands.
- Publish model weights only as release assets or external artifacts after checking dataset redistribution terms.
- Do not commit raw dataset archives, extracted datasets, SQLite runtime databases, or generated logs.

## Dataset caution

This project uses multiple data sources with different redistribution expectations.

- PKLot-style data may be easier to cite and reproduce, but you still need to confirm the exact export/license used for your training set.
- CNRPark-EXT is commonly treated as research-use data. Public redistribution of derived checkpoints may be restricted or ambiguous.

Because of that, the safest default for this public repo is:

- keep training and evaluation code public
- keep reproducibility instructions public
- keep trained weights out of git
- publish weights only after verifying the underlying data license permits redistribution of trained artifacts

## Recommended release pattern

If you decide to publish a model:

1. Put the checkpoint in a GitHub Release or external model registry, not in the repo tree.
2. Document exactly which datasets were used to train it.
3. State the intended use, such as `research/demo only`, if redistribution terms are limited.
4. Prefer publishing the smallest set of final weights needed for reproducibility or demo use.

## Current repo policy

- No model weights are tracked in git.
- No dataset archives should remain tracked in git.
- Runtime artifacts such as `parking.db`, `logs/`, `runs/`, and exported model bundles should stay untracked.
