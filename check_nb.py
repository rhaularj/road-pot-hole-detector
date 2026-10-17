import json
with open('notebooks/01_eda.ipynb') as f:
    data = json.load(f)
for i, c in enumerate(data['cells']):
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    print(f"Cell {i} ({c['cell_type']}): len={len(src)}")
    print(src[:300])
    print('---')
