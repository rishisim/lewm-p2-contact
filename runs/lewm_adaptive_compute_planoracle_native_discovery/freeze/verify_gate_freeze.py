#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parents[3];s=json.loads((Path(__file__).parent/'gate_freeze_seal.json').read_text())
for p,e in s['files'].items():
 if hashlib.sha256((r/p).read_bytes()).hexdigest()!=e:raise SystemExit('FAIL '+p)
print('gate_freeze_verified')
