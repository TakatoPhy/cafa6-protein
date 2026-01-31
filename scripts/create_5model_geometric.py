import pandas as pd
import numpy as np

files = [
    ('sota', '/home/taka/cafa6-protein/notebooks/external/sota_27jan_output/submission.tsv'),
    ('esm2', '/home/taka/cafa6-protein/notebooks/external/esm2_386_new/submission.tsv'),
    ('ens378', '/home/taka/cafa6-protein/notebooks/external/cafa_ensemble_378/submission.tsv'),
    ('goa', '/home/taka/cafa6-protein/notebooks/external/goa_propagation/submission.tsv'),
    ('tuning', '/home/taka/cafa6-protein/notebooks/external/cafa_tuning/submission.tsv'),
]

print('Loading sota...')
merged = pd.read_csv(files[0][1], sep='\t', names=['protein', 'go', 'score_sota'])
print(f'  {len(merged):,} rows')

for name, path in files[1:]:
    print(f'Loading {name}...')
    df = pd.read_csv(path, sep='\t', names=['protein', 'go', f'score_{name}'])
    print(f'  {len(df):,} rows')
    merged = merged.merge(df, on=['protein', 'go'], how='inner')
    print(f'  After merge: {len(merged):,} rows')
    del df

print('Computing geometric mean of 5 models...')
score_cols = [c for c in merged.columns if c.startswith('score_')]
# 幾何平均: (s1 * s2 * s3 * s4 * s5)^(1/5)
merged['score'] = np.power(merged[score_cols].prod(axis=1), 1/5)
result = merged[['protein', 'go', 'score']]
print(f'Result: {len(result):,} rows')

output_path = '/home/taka/cafa6-protein/submissions/5model_geometric.tsv'
result.to_csv(output_path, sep='\t', index=False, header=False)
print(f'Saved to {output_path}')
