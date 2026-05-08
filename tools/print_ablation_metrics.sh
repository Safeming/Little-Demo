#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/anim/bin/python}"

"$PYTHON_BIN" - <<'PY'
import glob
import os
import numpy as np

candidates = [
    'exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-baseline_15k_main',
    'exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-expbind_v3_15k-0311-0717',
    'exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main',
    'exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nobodycloth_15k-0312-1125',
]
candidates.extend(sorted(glob.glob('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_notemporal_15k-*')))
candidates.extend(sorted(glob.glob('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nosemantic_15k-*')))

print('=' * 124)
print(f"{'experiment':78s} {'psnr':>10s} {'ssim':>10s} {'lpips':>10s}")
print('=' * 124)
for exp_dir in candidates:
    path = os.path.join(exp_dir, 'test-view', 'results.npz')
    if not os.path.exists(path):
        print(f"{exp_dir:78s} {'MISSING':>10s} {'':>10s} {'':>10s}")
        continue
    data = np.load(path)
    psnr = float(data['psnr']) if 'psnr' in data else float('nan')
    ssim = float(data['ssim']) if 'ssim' in data else float('nan')
    lpips = float(data['lpips']) if 'lpips' in data else float('nan')
    print(f"{exp_dir:78s} {psnr:10.4f} {ssim:10.4f} {lpips:10.4f}")
print('=' * 124)
PY
