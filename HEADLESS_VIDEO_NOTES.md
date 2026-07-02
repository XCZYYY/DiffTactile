# DiffTactile Headless And Platform Recording Notes

This checkout supports two server modes:

- Headless compute mode: task scripts avoid `ti.GUI` and `ti.ui.Window`, use Matplotlib `Agg`, and save canonical outputs.
- Platform recording mode: task scripts run their real Taichi GUI/Window surfaces on a virtual display while `ffmpeg x11grab` records the displayed platform output.

Canonical outputs are saved under:

`/data1/determined/users/thomas/Dataset/Difftactile/output`

## Install

```bash
conda create -n difftactile python=3.9.16 -y
cd /data1/determined/users/thomas/Dataset/Difftactile/DiffTactile
iconv -f UTF-16 -t UTF-8 requirements.txt > /data1/determined/users/thomas/Dataset/Difftactile/output/notes/requirements_utf8.txt
conda run -n difftactile python -m pip install -U pip setuptools wheel
conda run -n difftactile python -m pip install -r /data1/determined/users/thomas/Dataset/Difftactile/output/notes/requirements_utf8.txt -e .
conda install -n difftactile -c conda-forge -y xorg-x11-server-xvfb-conda-x86_64 pyvirtualdisplay
conda create -p /data1/determined/users/thomas/Dataset/Difftactile/runtime/openssl10 -c conda-forge -y openssl=1.0.2u
ln -sfn libcrypto.so.1.0.0 /data1/determined/users/thomas/Dataset/Difftactile/runtime/openssl10/lib/libcrypto.so.10
ln -sfn libssl.so.1.0.0 /data1/determined/users/thomas/Dataset/Difftactile/runtime/openssl10/lib/libssl.so.10
```

## Headless Smoke Tests

Run only `box_open`:

```bash
bash scripts/run_headless_smoke_tests.sh
```

Run all patched tasks:

```bash
bash scripts/run_headless_smoke_tests.sh --all
```

The scripts choose the GPU with the most free memory unless `CUDA_VISIBLE_DEVICES` is already set. These smoke tests do not create videos; platform recording is the video deliverable path.

## Platform Screen Recording

Record the real `box_open` platform display with the long profile:

```bash
conda run -n difftactile python scripts/record_platform_run.py \
  --task box_open \
  --profile long \
  --run_name box_open_platform_long \
  --output_root /data1/determined/users/thomas/Dataset/Difftactile/output \
  -- --use_state --use_tactile
```

The long profile starts with `num_sub_steps=20`, `num_total_steps=300`, `num_opt_steps=10`, records the complete run with no frame cap, and automatically retries once with `20/600/20` if the resulting video is shorter than 10 minutes.

## Outputs

Each run writes to `output/runs/<task>/<run_id>/` with:

- `screen_recordings/*_platform.mp4` for real platform screen captures
- `plots/*.png`
- `trajectories/*.npy`
- `metadata.json`
- `platform_recording_metadata.json` for platform recording runs

Platform recordings are also mirrored to `output/videos/<run_id>_<task>_platform.mp4`.

Validate platform recordings with:

```bash
conda run -n difftactile python scripts/validate_outputs.py \
  --output-root /data1/determined/users/thomas/Dataset/Difftactile/output \
  --require-platform-video
```

## Notes

- Inline `--record_video` is intentionally disabled because it produced synthetic state visualizations instead of platform screen captures.
- Platform recording uses Conda-provided Xvfb plus system `ffmpeg`; it does not use `sudo`.
- The Conda Xvfb binary expects `libcrypto.so.10`; the isolated `runtime/openssl10` prefix provides that legacy ABI only to the Xvfb process.
- Taichi may print precision warnings during smoke tests. Those are preserved in logs and were present during successful smoke runs.
