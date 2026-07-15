#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parents[3]; s=json.loads((Path(__file__).parent/'pre_generation_seal.json').read_text())
for p,e in s['files'].items():
 h=hashlib.sha256((r/p).read_bytes()).hexdigest()
 if h!=e: raise SystemExit(f"FAIL {p} {h} != {e}")
print("pre_generation_seal_verified")
