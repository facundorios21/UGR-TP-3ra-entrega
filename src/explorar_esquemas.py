import glob
import pandas as pd

files = sorted(glob.glob('csv/*.csv'))

for f in files:
    print(f"=== {f} ===")
    df_sample = pd.read_csv(f, nrows=50000, low_memory=False)
    cols = [c.upper().strip().replace('"', '') for c in df_sample.columns]
    df_sample.columns = cols
    print("Columns:", list(df_sample.columns))
    
    # Check linea col
    line_col = [c for c in df_sample.columns if 'LINEA' in c or 'LINE' in c]
    if line_col:
        print("Unique Lineas (sample):", df_sample[line_col[0]].dropna().unique()[:10])
    
    station_col = [c for c in df_sample.columns if 'ESTACION' in c and 'ID' not in c]
    if station_col:
        print("Sample Stations:", df_sample[station_col[0]].dropna().unique()[:5])
    
    date_col = [c for c in df_sample.columns if 'FECHA' in c]
    if date_col:
        print("Sample Dates:", df_sample[date_col[0]].dropna().unique()[:5])
    print()
