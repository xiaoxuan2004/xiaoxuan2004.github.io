"""Download the author's fixed public input files for local reproduction."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import shutil

ROOT=Path(__file__).resolve().parent
COMMIT='54cc540a8bed3e2cbc6d56dfa3cf0b05ce6b5c81'
DEST=ROOT/'reference/PIC_mortality_upstream'
for name in ['icu_first24hours.csv','tutorial/data.xlsx']:
    target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        urlretrieve(f'https://raw.githubusercontent.com/hsd1503/PIC_mortality/{COMMIT}/{name}',target)
    print('Available:',target)
source=DEST/'tutorial/data.xlsx'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='77bebefce09b2127a79a09c4c0fbf8346a60ab3d870ad25f8dff3a2d6025444b'
(ROOT/'data').mkdir(exist_ok=True)
target=ROOT/'data/data.xlsx'
if target.exists():
    assert target.read_bytes()==source.read_bytes(),'Existing data/data.xlsx differs; preserved without overwrite.'
else:shutil.copy2(source,target)
