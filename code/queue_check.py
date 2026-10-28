"""Provenance checks used by run_queue.sh (exit code 0 = yes).

  queue_check.py stamp "<train args + flags>"            -> prints a stamp (args + train/common code hashes)
  queue_check.py train_done <run_dir> <stamp>            -> complete run trained with exactly this stamp
  queue_check.py eval_done <eval_dir> <ckpt> <attacks> <pgd_restarts>
                                                          -> eval.json matches this checkpoint, code and attack set
"""
import hashlib, json, sys
from pathlib import Path

CODE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


cmd = sys.argv[1]
if cmd == 'stamp':
    h = hashlib.sha256()
    for part in (' '.join(sys.argv[2].split()), sha(CODE / 'train_sprat_v2.py'), sha(CODE / 'sprat_common.py')):
        h.update(part.encode())
    print(h.hexdigest()[:16])
elif cmd == 'train_done':
    d, stamp = Path(sys.argv[2]), sys.argv[3]
    ok = ((d / 'final.pth').exists() and (d / 'train_summary.json').exists() and (d / '.stamp').exists()
          and (d / '.stamp').read_text().strip() == stamp)
    sys.exit(0 if ok else 1)
elif cmd == 'eval_done':
    e, ckpt, attacks, restarts = Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
    try:
        js = json.loads((e / 'eval.json').read_text())
        ok = (js['ckpt_sha256'] == sha(ckpt) and js['eval_script_sha256'] == sha(CODE / 'eval_sprat_v2.py')
              and js['common_sha256'] == sha(CODE / 'sprat_common.py') and js['args']['attacks'] == attacks
              and js['args']['pgd_restarts'] == restarts and not js['args'].get('max_n')
              and (e / 'per_sample_preds.npz').exists())
    except Exception:
        ok = False
    sys.exit(0 if ok else 1)
else:
    sys.exit(f'unknown command {cmd}')
