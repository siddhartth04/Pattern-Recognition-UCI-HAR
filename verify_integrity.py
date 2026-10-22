"""Verify the distributed package against its SHA-256 manifest."""
from pathlib import Path
import json, hashlib
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST_SHA256.json').read_text())
bad=[]
for relative,expected in manifest.items():
    path=root/relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        bad.append(relative)
if bad:
    raise SystemExit('Missing or changed files:\n'+'\n'.join(bad))
print(f'All {len(manifest)} distributed files match their SHA-256 hashes.')
