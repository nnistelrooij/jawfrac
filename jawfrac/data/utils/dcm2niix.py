import os
from pathlib import Path
import subprocess


if __name__ == '__main__':
    root = Path('/mnt/diag/fractures/CTs')
    dirs = set(p.parent for p in root.glob('**/*.dcm'))
    for path in dirs:
        subprocess.run([
            'dcm2niix',
            '-f', 'main',
            '-z', 'y',
            str(path),
        ])
