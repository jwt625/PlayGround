#!/usr/bin/env python3
import pandas as pd

# Load both training runs
df1 = pd.read_csv('nanochat/training_20251221_070319.csv')  # depth=16, 336M params
df2 = pd.read_csv('nanochat/training_20251221_183935.csv')  # depth=24, 881M params

print('Training Run Comparison')
print('=' * 70)
print()
print('Run 1: depth=16, 336M parameters, 12,800 steps')
print('-' * 70)
print(f'Initial loss:  {df1.iloc[0]["loss"]:.6f}')
print(f'Final loss:    {df1.iloc[-1]["loss"]:.6f}')
print(f'Loss reduction: {df1.iloc[0]["loss"] - df1.iloc[-1]["loss"]:.6f}')
print(f'Training time: {df1.iloc[-1]["total_time_min"]:.2f} minutes ({df1.iloc[-1]["total_time_min"]/60:.2f} hours)')
print()

print('Run 2: depth=24, 881M parameters, 33,600 steps (incomplete)')
print('-' * 70)
print(f'Initial loss:  {df2.iloc[0]["loss"]:.6f}')
print(f'Final loss:    {df2.iloc[-1]["loss"]:.6f}')
print(f'Loss reduction: {df2.iloc[0]["loss"] - df2.iloc[-1]["loss"]:.6f}')
print(f'Steps completed: {len(df2)} / 33,600 ({100*len(df2)/33600:.1f}%)')
print(f'Training time: {df2.iloc[-1]["total_time_min"]:.2f} minutes ({df2.iloc[-1]["total_time_min"]/60:.2f} hours)')
print()

print('Comparison at equivalent steps')
print('-' * 70)
# Compare at step 12,000 (both runs have this)
step_compare = 12000
loss1_at_step = df1[df1['step'] == step_compare]['loss'].values[0]
loss2_at_step = df2[df2['step'] == step_compare]['loss'].values[0]
print(f'At step {step_compare}:')
print(f'  depth=16 (336M): {loss1_at_step:.6f}')
print(f'  depth=24 (881M): {loss2_at_step:.6f}')
print(f'  Difference: {loss1_at_step - loss2_at_step:.6f} ({"better" if loss2_at_step < loss1_at_step else "worse"} for larger model)')
print()

# Compare at various milestones
print('Loss progression comparison')
print('-' * 70)
milestones = [0, 1000, 5000, 10000, 12000]
print(f'{"Step":>6} | {"depth=16 (336M)":>16} | {"depth=24 (881M)":>16} | {"Difference":>12}')
print('-' * 70)
for step in milestones:
    if step < len(df1) and step < len(df2):
        l1 = df1.iloc[step]['loss']
        l2 = df2.iloc[step]['loss']
        diff = l1 - l2
        print(f'{step:6d} | {l1:16.6f} | {l2:16.6f} | {diff:+12.6f}')

