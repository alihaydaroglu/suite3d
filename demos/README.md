# Suite3D demos

| | demo | recording |
|---|---|---|
| **01** | [`01-v1-tc030/`](01-v1-tc030/) | V1, standard 2P, 7 planes |
| **02** | [`02-lbm-ss004/`](02-lbm-ss004/) | LBM, 22 planes |
| **03** | [`03-hippocampus/`](03-hippocampus/) | CA1, standard 2P, 4 planes |
| **04** | [`04-sweep/`](04-sweep/) | parameter sweep, reuses demo 03 |

Each dataset demo has a `run_pipeline.py` (start to finish) and a
`walkthrough.ipynb` (stage by stage). Shared code is in [`common/`](common/).

## Data

Download the datasets from figshare and point `--data-root` at the unpacked
folder:

```
<data-root>/
    v1/raw/*.tif
    hippocampus/raw/*.tif
    lbm/raw/*.tif
    manifest.json
```

`--data-root` also accepts a single dataset folder (`.../v1`) or its `raw/`
directory.

## Install

```bash
pip install git+https://github.com/alihaydaroglu/suite3d.git
pip install 'cupy-cuda12x>=13.0,<14.0'    # GPU, for registration
pip install 'suite3d[viz]'                # only for --viewer napari
```

## Run

```bash
cd 01-v1-tc030
python run_pipeline.py --data-root /path/to/figshare --out-dir ./results
```

Writes `./results/s3d-demo-v1/`, then exports results and opens a viewer.

### Flags

```
--data-root PATH       figshare download root
--out-dir PATH         where the job directory is written
--job-id NAME          run name (default: demo-<dataset>)
--viewer {html,napari,none}   default: html
--n-frames N           clip the correlation map to the first N volumes
--extract-batch N      volumes held in RAM at once during extraction
--extract-batch-gb G   target extraction batch size in GiB (default: 4.0)
--t-batch-size N       volumes per correlation-map batch
--n-init-files N       tifs read by the init pass
--skip-{init,register,corrmap,segment,extract}   reuse what is on disk
--overwrite            recreate the job directory from scratch
```

## Requirements

| demo | download | peak RAM | wall time (GPU) |
|---|---|---|---|
| 01 v1 | 21 GB | 14 GiB | ~5 min |
| 02 lbm | 56 GB | 114 GiB | ~40 min |
| 03 hippocampus | 21 GB | 8 GiB | ~5 min |

Registration needs a GPU (`gpu_reg=True`); it has a CPU fallback but is much
slower. Plan for disk equal to `2 × raw` (the registered movie is written beside
the raw data).

## AWS

See [`aws/README.md`](aws/README.md) for demo 01.
