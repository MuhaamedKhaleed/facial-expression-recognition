"""Invent a tiny2304-pixel CSV. Not faces or a FER quality benchmark."""
import csv
from pathlib import Path

path=Path('synthetic-demo')
if path.exists():raise ValueError('synthetic-demo exists, no overwrite.')
path.mkdir()
with (path/'fer.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.writer(stream);writer.writerow(['emotion','pixels','Usage'])
    for i,usage in enumerate(['Training','Training','PublicTest','PrivateTest']):
        writer.writerow([i,' '.join([str(i*20)]*2304),usage])
print(path/'fer.csv')
