# Environment

This project uses the existing conda environment:

```powershell
conda activate qlib
```

Do not change package versions without explicit approval.

In particular:

- Do not run `conda install`.
- Do not run `conda update`.
- Do not run `pip install`.
- Do not export and re-create the environment unless requested.

If a missing dependency blocks implementation or testing, document the missing
package first and ask before changing the environment.
