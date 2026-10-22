import glob
import pandas as pd

files = sorted(glob.glob('csv/*.csv'))

raw_names_by_line = {'A': set(), 'B': set(), 'C': set()}

for f in files:
    print(f"Reading {f}...")
    for chunk in pd.read_csv(f, chunksize=1000000, encoding='latin1', low_memory=False):
        chunk.columns = [c.upper().strip().replace('"', '') for c in chunk.columns]
        l_col = [c for c in chunk.columns if 'LINEA' in c or 'LINE' in c][0]
        s_col = [c for c in chunk.columns if 'ESTACION' in c and 'ID' not in c][0]
        
        chunk['clean_line'] = chunk[l_col].astype(str).str.upper().str.replace('LINEA', '', regex=False).str.replace('_', '', regex=False).str.strip()
        filtered = chunk[chunk['clean_line'].isin(['A', 'B', 'C'])]
        for l in ['A', 'B', 'C']:
            sub = filtered[filtered['clean_line'] == l]
            if not sub.empty:
                raw_names_by_line[l].update(sub[s_col].dropna().unique())

for l in ['A', 'B', 'C']:
    print(f"\nAll unique station values for Line {l}:")
    for s in sorted(list(raw_names_by_line[l])):
        print(f"  {repr(s)}")
