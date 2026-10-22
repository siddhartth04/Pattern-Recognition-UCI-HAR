"""Download, verify, and load the official subject-separated UCI HAR data."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile
import numpy as np

URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
ARCHIVE_SHA256 = 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'
ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def download(data_dir=None):
    data_dir = Path(data_dir or ROOT / 'data')
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / 'uci_har_official.zip'
    if not target.exists():
        print('Downloading public UCI HAR archive...', flush=True)
        temporary = target.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        temporary.replace(target)
    if sha256(target) != ARCHIVE_SHA256:
        raise ValueError('Archive checksum differs from the verified experiment snapshot')
    destination = data_dir / 'UCI HAR Dataset'
    # Re-extract verified members to repair any modified local cache.
    with zipfile.ZipFile(target) as outer:
        nested = [n for n in outer.namelist() if n.endswith('UCI HAR Dataset.zip')]
        payload = io.BytesIO(outer.read(nested[0])) if nested else target
        with zipfile.ZipFile(payload) as archive:
            # Extract only known data members; never trust arbitrary archive paths.
            wanted = ['features.txt', 'features_info.txt', 'activity_labels.txt', 'README.txt']
            wanted += [f'{split}/{name}_{split}.txt' for split in ['train','test']
                       for name in ['X','y','subject']]
            for relative in wanted:
                member = 'UCI HAR Dataset/' + relative
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.read(member))
    manifest = {'url': URL, 'archive_sha256': sha256(target),
                'files': {p.relative_to(data_dir).as_posix(): sha256(p)
                          for p in sorted(destination.rglob('*.txt'))}}
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return destination

def load_split(split, data_dir=None):
    folder = Path(data_dir or ROOT / 'data') / 'UCI HAR Dataset'
    X = np.loadtxt(folder / split / f'X_{split}.txt')
    y = np.loadtxt(folder / split / f'y_{split}.txt', dtype=int)
    subjects = np.loadtxt(folder / split / f'subject_{split}.txt', dtype=int)
    expected = 7352 if split == 'train' else 2947
    if X.shape != (expected, 561) or len(y) != expected or len(subjects) != expected:
        raise ValueError('Unexpected official dataset dimensions')
    if not np.isfinite(X).all() or set(y) != set(range(1,7)):
        raise ValueError('Invalid data values or class labels')
    return X, y, subjects

if __name__ == '__main__':
    print(download())
