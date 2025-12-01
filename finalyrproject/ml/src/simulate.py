import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)
users = ['Asna', 'Ann Mary', 'Elvina','Pavithra','Masoom','Harshala','Priya']
rows = []

for u in users:

    for i in range(10000):
        base = rng.normal(1.0 + 0.2 * users.index(u), 0.05)
        row = {
            'ks_count': rng.integers(150, 300),
            'ks_rate': base * 6 + rng.normal(0, 0.5),
            'dwell_mean': base * 90 + rng.normal(0, 6),
            'dwell_std': base * 12 + rng.normal(0, 2),
            'flight_mean': base * 110 + rng.normal(0, 7),
            'flight_std': base * 20 + rng.normal(0, 3),
            'backspace_rate': abs(rng.normal(0.05 + 0.01 * users.index(u), 0.01)),
            'digraph_mean': base * 115 + rng.normal(0, 8),
            'digraph_std': base * 18 + rng.normal(0, 3),
            'user_id': u
        }
        rows.append(row)

df = pd.DataFrame(rows)
os.makedirs('ml/data_csv', exist_ok=True)
df.to_csv('ml/data_csv/typing_dataset.csv', index=False)
print(f"Wrote ml/data_csv/typing_dataset.csv with {len(df)} rows")
# Add impostor samples (wrong user_id)
for i in range(10):
    row = rows[i].copy()
    row['user_id'] = 'impostor'
    rows.append(row)
