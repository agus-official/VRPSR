"""Static checks by default; --runtime also checks imports and actual codec operations."""
import argparse
import ast
import builtins
import importlib
from pathlib import Path
import subprocess
import symtable
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime', action='store_true')
args = p.parse_args()
sys.path.insert(0, str(ROOT / 'src'))
files = [ROOT / 'src/train_simulator.py', ROOT / 'src/train_sr.py']
files += sorted((ROOT / 'src/vrpsr_training').glob('*.py'))
for path in files:
    text = path.read_text()
    compile(text, str(path), 'exec')
    table = symtable.symtable(text, str(path), 'exec')
    known = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    known |= set(vars(builtins)) | {'__name__'}
    def visit(scope):
        for sym in scope.get_symbols():
            if sym.is_global() and sym.is_referenced() and sym.get_name() not in known:
                raise AssertionError(f'{path}: unresolved name {sym.get_name()}')
        for child in scope.get_children():
            visit(child)
    visit(table)
for name in ['train_simulator.py', 'train_sr.py']:
    subprocess.run([sys.executable, str(ROOT / 'src' / name), '--help'], check=True, stdout=subprocess.DEVNULL)
from vrpsr_training.options import parse_sim_args, parse_sr_args
common = ['--train_data','train.txt','--val_data','val.txt','--de_net_path','de.pth','--output_dir','out']
sim = parse_sim_args(common)
sr = parse_sr_args(common + ['--pretrained_sr_path','sr.pkl','--pretrained_sim_path','sim.pkl'])
assert (sim.train_batch_size, sim.gradient_accumulation_steps) == (4, 2)
assert (sr.train_batch_size, sr.gradient_accumulation_steps) == (2, 4)
assert sim.fixed_qp is sr.fixed_qp is None
assert sim.fixed_qp_val == sr.fixed_qp_val == 38
assert sim.fixed_codec_type_val == sr.fixed_codec_type_val == 'libx264'
for script in (ROOT / 'scripts').glob('*.sh'):
    subprocess.run(['bash','-n',str(script)], check=True)
print(f'PASS: syntax, unresolved names, CLI defaults and shell syntax ({len(files)} Python files).')
if args.runtime:
    required = ['torch','torchvision','accelerate','diffusers','transformers','peft','lpips','pyiqa','vision_aided_loss','s3diff_meta_embed','de_net','vrpsr_training.data']
    for module in required:
        importlib.import_module(module)
        print('Import OK:', module)
    import torch
    import numpy as np
    from PIL import Image
    from vrpsr_training.codec import proc, build_temb
    sample = Image.fromarray(np.random.default_rng(0).integers(0,256,(512,512,3),dtype=np.uint8))
    rates = []
    for codec in ['libx264','libx265','libx266']:
        _, rec, rate = proc(sample, codec, 38)
        assert rec.size == (512,512) and np.isfinite(rate) and rate > 0
        rates.append(rate)
        print(f'Codec OK: {codec}; legacy rate={rate}')
    meta = build_temb(['libx264','libx265','libx266'], rates, device='cpu')
    assert meta.shape == (3,4)
    assert torch.equal(meta[:,:3], torch.eye(3))
    print('CUDA available:', torch.cuda.is_available())
    print('PASS: runtime imports, all three codecs, condition vector. No model weights or training were tested.')
else:
    print('NOT RUN: runtime imports, codec binaries, weights, forward/backward, CUDA or DDP.')
