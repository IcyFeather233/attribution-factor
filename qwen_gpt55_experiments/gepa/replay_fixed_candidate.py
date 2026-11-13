"""Re-evaluate an already selected candidate to measure execution-regime confounding."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .protocol import require_completed_evolution
from ..common.execution_protocol import record_execution_backend
from ..supervise_parallel import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=['group2_1_causing_shifted','group2_2_inevitable_shifted'], required=True)
    args = p.parse_args()
    base = Path(__file__).resolve().parent / args.group
    source = base / 'multi_proposal_rerun_20260917'
    target = base / 'fixed_candidate_replay_20261002'
    evolution = json.loads((source / 'evolution_summary.json').read_text())
    require_completed_evolution(evolution)
    if evolution.get('best_idx') != 0:
        raise RuntimeError('Expected audited retained-original candidate')
    record_execution_backend(target)
    copied = {}
    for name in ['best_candidate_overlay.yaml','best_scaffold.yaml','evolution_summary.json']:
        original = source / name
        output = target / name
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Candidate snapshot differs on resume')
        shutil.copy2(original, output)
        copied[name] = digest
    atomic_json(target / 'replay_protocol.json', {'source':str(source),'candidate_hashes':copied,
                'rerun_evolution':False,'reason':'Historical 92/148 or 85/148 TimeExceeded outcomes under same scaffold',
                'analysis':'Compare execution-regime sensitivity; do not reinterpret replay as new optimized candidate',
                'wall_time_seconds':3600,'workers':8})
    subprocess.run([sys.executable,'-B','-m','qwen_gpt55_experiments.gepa.resume_final','--ablation-group',args.group,
                    '--run-name',target.name,'--workers','8'],check=True)


if __name__ == '__main__':
    main()
