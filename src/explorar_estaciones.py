import glob
import pandas as pd

files = sorted(glob.glob('csv/*.csv'))

stations_by_line = {'A': set(), 'B': set(), 'C': set()}

for f in files:
    print(f"Reading sample from {f}...")
    # read in chunks
    for chunk in pd.read_csv(f, chunksize=250000, low_memory=False):
        chunk.columns = [c.upper().strip().replace('"', '') for c in chunk.columns]
        line_col = [c for c in chunk.columns if 'LINEA' in c or 'LINE' in c][0]
        st_col = [c for c in chunk.columns if 'ESTACION' in c and 'ID' not in c][0]
        
        # normalize line
        chunk['clean_line'] = chunk[line_col].astype(str).str.upper().str.replace('LINEA', '', regex=False).str.replace('_', '', regex=False).str.strip()
        
        filtered = chunk[chunk['clean_line'].isin(['A', 'B', 'C'])]
        for l in ['A', 'B', 'C']:
            sub = filtered[filtered['clean_line'] == l]
            if not sub.empty:
                stations_by_line[l].update(sub[st_col].dropna().unique())
        # Break after 2 chunks per file just to get a good preview
        break

for l, sts in stations_by_line.items():
    print(f"\nLine {l} ({len(sts)} stations found):")
    print(sorted(list(sts)))
