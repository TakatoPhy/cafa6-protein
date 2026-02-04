"""Merge 4 model submissions using memory-efficient inner join."""
import pandas as pd
import numpy as np

# メモリ効率のため、共通のprot-goペアのみで結合（inner join）
files = {
    'sota': '/home/taka/cafa6-protein/notebooks/external/sota_27jan_output/submission.tsv',
    'esm2_new': '/home/taka/cafa6-protein/notebooks/external/esm2_386_new/submission.tsv',
    'ensemble378': '/home/taka/cafa6-protein/notebooks/external/cafa_ensemble_378/submission.tsv',
    'goa': '/home/taka/cafa6-protein/notebooks/external/goa_propagation/submission.tsv',
}

print('Loading sota...')
merged = pd.read_csv(files['sota'], sep='\t', names=['protein', 'go', 'score_sota'])
print(f'  {len(merged):,} rows')

print('Loading esm2_new...')
esm2 = pd.read_csv(files['esm2_new'], sep='\t', names=['protein', 'go', 'score_esm2'])
print(f'  {len(esm2):,} rows')
merged = merged.merge(esm2, on=['protein', 'go'], how='inner')
print(f'  After merge: {len(merged):,} rows')
del esm2

print('Loading ensemble378...')
ens = pd.read_csv(files['ensemble378'], sep='\t', names=['protein', 'go', 'score_ens'])
print(f'  {len(ens):,} rows')
merged = merged.merge(ens, on=['protein', 'go'], how='inner')
print(f'  After merge: {len(merged):,} rows')
del ens

print('Loading goa...')
goa = pd.read_csv(files['goa'], sep='\t', names=['protein', 'go', 'score_goa'])
print(f'  {len(goa):,} rows')
merged = merged.merge(goa, on=['protein', 'go'], how='inner')
print(f'  After merge: {len(merged):,} rows')
del goa

print('Computing mean...')
merged['score'] = (merged['score_sota'] + merged['score_esm2'] + merged['score_ens'] + merged['score_goa']) / 4
result = merged[['protein', 'go', 'score']]
print(f'Result: {len(result):,} rows')

output_path = '/home/taka/cafa6-protein/submissions/4model_merge.tsv'
result.to_csv(output_path, sep='\t', index=False, header=False)
print(f'Saved to {output_path}')
