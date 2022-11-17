import numpy as np
from pathlib import Path

path = Path('/home/ou/workspace/codebase/detection3d/OpenPCDet/data')
num_samples = 7481
num_rate = 4

val_id = np.random.choice(7481, int(num_samples / 5), replace=False)
val_id.sort()
train_id = []
for i in range(0, num_samples):
    if i not in val_id:
        train_id.append(i)

train_id = np.array(train_id)
print(val_id)
print(train_id)

train_file = path / 'train.txt'
val_file = path / 'val.txt'

with open(train_file, 'w') as f:
    for i in train_id[:-1]:
        f.write('%06d\n' % i)
    f.write('%06d' % train_id[-1])

with open(val_file, 'w') as f:
    for i in val_id[:-1]:
        f.write('%06d\n' % i)
    f.write('%06d' % val_id[-1])
