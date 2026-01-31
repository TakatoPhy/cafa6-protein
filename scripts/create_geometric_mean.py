import pandas as pd
import numpy as np

print('Loading SOTA...')
sota = pd.read_csv('/home/taka/cafa6-protein/notebooks/external/sota_27jan_output/submission.tsv', 
                   sep='\t', names=['protein', 'go', 'score'])
print(f'SOTA: {len(sota):,} rows')

print('Loading ESM2 new...')
esm2 = pd.read_csv('/home/taka/cafa6-protein/notebooks/external/esm2_386_new/submission.tsv',
                   sep='\t', names=['protein', 'go', 'score'])
print(f'ESM2: {len(esm2):,} rows')

print('Merging with geometric mean...')
merged = sota.merge(esm2, on=['protein', 'go'], suffixes=('_sota', '_esm2'))
merged['score'] = np.sqrt(merged['score_sota'] * merged['score_esm2'])
result = merged[['protein', 'go', 'score']]
print(f'Result: {len(result):,} rows')

output_path = '/home/taka/cafa6-protein/submissions/geometric_mean_new.tsv'
result.to_csv(output_path, sep='\t', index=False, header=False)
print(f'Saved to {output_path}')
