import sys
import polars as pl
from collections import defaultdict
import json
import os

def convert_parquet_files(fn: str):
    fn_no_ext, _ = os.path.splitext(fn)
    df = pl.read_parquet(fn)
    column_names = df.columns
    all_counts = {}
    for name in column_names:
        vals = df[name].to_numpy()
        vals.sort()
        counts = {int(val): 0 for val in vals}
        for val in df[name]:
            counts[val] += 1
        all_counts[name] = counts
        
    json_fn = f'{fn_no_ext}.json'
    with open(json_fn, 'w') as f:
        json.dump(all_counts, f, indent=4)
    # return json.dumps(all_counts, indent=4)
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Please provide exactly one argument: <parquet_file>")
    fn = sys.argv[1]
    convert_parquet_files(fn)