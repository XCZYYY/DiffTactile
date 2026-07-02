# DiffTactile Headless Video Notes

This checkout is patched for headless Linux GPU servers. In headless mode the task scripts avoid `ti.GUI` and `ti.ui.Window`, set Matplotlib to `Agg`, set `PYOPENGL_PLATFORM=egl` for pyrender/mesh sampling, and save canonical outputs under:

`/data1/determined/users/thomas/Dataset/Difftactile/output`

## Install

```bash
conda create -n difftactile python=3.9.16 -y
cd /data1/determined/users/thomas/Dataset/Difftactile/DiffTactile
iconv -f UTF-16 -t UTF-8 requirements.txt > /data1/determined/users/thomas/Dataset/Difftactile/output/notes/requirements_utf8.txt
conda run -n difftactile python -m pip install -U pip setuptools wheel
conda run -n difftactile python -m pip install -r /data1/determined/users/thomas/Dataset/Difftactile/output/notes/requirements_utf8.txt -e .
```

## Smoke Tests

Run only `box_open`:

```bash
bash scripts/run_headless_smoke_tests.sh
```

Run all patched tasks:

```bash
bash scripts/run_headless_smoke_tests.sh --all
```

The scripts choose the GPU with the most free memory unless `CUDA_VISIBLE_DEVICES` is already set.

## Direct Task Command

```bash
cd /data1/determined/users/thomas/Dataset/Difftactile/DiffTactile/difftactile/tasks
DIFFTACTILE_HEADLESS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
conda run -n difftactile python box_open.py \
  --use_state --use_tactile --headless --record_video --smoke \
  --output_root /data1/determined/users/thomas/Dataset/Difftactile/output
```

Medium `box_open` run:

```bash
conda run -n difftactile python box_open.py \
  --use_state --use_tactile --headless --record_video \
  --num_sub_steps 20 --num_total_steps 100 --num_opt_steps 3 --record_stride 5 \
  --run_name box_open_medium \
  --output_root /data1/determined/users/thomas/Dataset/Difftactile/output
```

## Outputs

Each run writes to `output/runs/<task>/<run_id>/` with:

- `videos/*.mp4` or `*.avi`
- `plots/*.png`
- `trajectories/*.npy`
- `metadata.json`

Videos are also mirrored to `output/videos/<run_id>_<task>.mp4`.

## Known Limitations

- Headless videos are state visualizations drawn from task arrays, not pixel-perfect captures of the original GUI windows.
- `mp4v` is tried first; if OpenCV cannot open it, the recorder falls back to AVI/XVID.
- Taichi may print precision warnings during smoke tests. Those are preserved in logs and were present during successful smoke runs.
