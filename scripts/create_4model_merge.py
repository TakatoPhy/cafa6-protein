import pandas as pd
import numpy as np

files = {
    'sota': '/home/taka/cafa6-protein/notebooks/external/sota_27jan_output/submission.tsv',
    'esm2_new': '/home/taka/cafa6-protein/notebooks/external/esm2_386_new/submission.tsv',
    'ensemble378': '/home/taka/cafa6-protein/notebooks/external/cafa_ensemble_378/submission.tsv',
    'goa': '/home/taka/cafa6-protein/notebooks/external/goa_propagation/submission.tsv',
}

dfs = {}
for name, path in files.items():
    print(f'Loading {name}...')
    df = pd.read_csv(path, sep='\t', names=['protein', 'go', 'score'])
    print(f'  {len(df):,} rows')
    dfs[name] = df

print('Merging 4 models...')
# Start with sota
merged = dfs['sota'].copy()
merged = merged.rename(columns={'score': 'score_sota'})

for name in ['esm2_new', 'ensemble378', 'goa']:
    merged = merged.merge(dfs[name], on=['protein', 'go'], how='outer', suffixes=('', f'_{name}'))
    merged = merged.rename(columns={'score': f'score_{name}'})

# Fill NaN with 0 and compute mean
score_cols = [c for c in merged.columns if c.startswith('score_')]
print(f'Score columns: {score_cols}')

for col in score_cols:
    merged[col] = merged[col].fillna(0)

merged['score'] = merged[score_cols].mean(axis=1)
result = merged[['protein', 'go', 'score']]
print(f'Result: {len(result):,} rows')

output_path = '/home/taka/cafa6-protein/submissions/4model_merge.tsv'
result.to_csv(output_path, sep='\t', index=False, header=False)
print(f'Saved to {output_path}')
