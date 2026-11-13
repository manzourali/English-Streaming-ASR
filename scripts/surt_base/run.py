"""Launch the actual recipe entry point once and retain the exact argv."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from common import dump


def main():
    base = Path(os.environ['BASE'])
    inputs = base / 'assets/inputs.json'
    items = json.loads(inputs.read_text())
    assert len(items) == 3 and sum(i['duration'] for i in items) <= 36.5
    output = base / 'results'
    output.mkdir(exist_ok=True)
    result = output / 'result.json'
    if result.exists():
        raise RuntimeError('Existing result: preserve it and use a fresh run directory/session')
    model = base / 'assets/model'
    argv = [sys.executable, str(Path(os.environ['RECIPE']) / 'dprnn_zipformer/pretrained.py'),
            '--checkpoint', str(model / 'exp/surt_base/pretrained.pt'),
            '--tokens', str(model / 'data/lang_bpe_500/tokens.txt'),
            '--sample-rate', '16000', '--decoding-method', 'greedy_search',
            '--method', 'greedy_search', '--max-sym-per-frame', '1',
            '--context-size', '2', '--num-mask-encoder-layers', '4',
            '--mask-encoder-dim', '256', '--mask-encoder-segment-size', '32',
            '--num-encoder-layers', '2,2,2,2,2',
            '--feedforward-dims', '768,768,768,768,768', '--nhead', '8,8,8,8,8',
            '--encoder-dims', '256,256,256,256,256',
            '--attention-dims', '192,192,192,192,192',
            '--encoder-unmasked-dims', '192,192,192,192,192',
            '--zipformer-downsampling-factors', '1,2,4,8,2',
            '--cnn-module-kernels', '31,31,31,31,31',
            '--use-joint-encoder-layer', 'lstm', '--decoder-dim', '512',
            '--joiner-dim', '512', '--decode-chunk-len', '32',
            '--num-left-chunks', '4', *[i['audio'] for i in items]]
    env = dict(os.environ, SURT_BASE_INPUTS=str(inputs), SURT_BASE_RESULT=str(result))
    dump(output / 'command.json', argv)
    (output / 'command.sh').write_text(shlex.join(argv) + '\n')
    begin = time.perf_counter()
    status = subprocess.run(argv, env=env, check=False).returncode
    dump(output / 'process.json', dict(exit_code=status,
         wall_seconds=time.perf_counter() - begin,
         includes='process startup, imports, load, audio IO, features, warmup, decode, scoring'))
    if status:
        raise SystemExit(status)
    data = json.loads(result.read_text())
    assert data['status'] == 'real_decode_completed'
    assert len(data['utterances']) == 3


if __name__ == '__main__':
    main()
