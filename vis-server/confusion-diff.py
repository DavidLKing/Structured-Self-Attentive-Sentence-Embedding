import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
import numpy as np

def get_confusions(file_handle):
    confs = {}
    for line in file_handle:
        _ , pred, lbl, _ = line.split('\t')
        pred, lbl = int(pred), int(lbl)
        if lbl in confs:
            if pred in confs[lbl]:
                confs[lbl][pred] += 1
            else:
                confs[lbl][pred] = 1
        else:
            confs[lbl] = {}
            confs[lbl][pred] = 1
    return confs

def get_freqmap(file_handle):
    class_counts = {}
    for line in file_handle:
        _ , _, lbl, _ = line.split('\t')
        lbl = int(lbl)
        if lbl in class_counts:
            class_counts[lbl] += 1
        else:
            class_counts[lbl] = 1
    freqsort = reversed(sorted([(ct, cls) for (cls, ct) in class_counts.items()]))
    freqmap = {}
    for idx, (ct, cls) in enumerate(freqsort):
        freqmap[cls] = idx
    return freqmap, class_counts

with open('cnn-baseline-strict-choices.csv', 'r') as f:
    freqmap, class_counts = get_freqmap(f)
            
with open('choices.cash.csv', 'r') as f:
    rnn_confs = get_confusions(f)

with open('cnn-baseline-strict-choices.csv', 'r') as f:
    cnn_confs = get_confusions(f)

mat = np.zeros((334,334), dtype=float)

for lbl in rnn_confs:
    for pred in rnn_confs[lbl]:
        #if pred != lbl:
        mat[freqmap[lbl], freqmap[pred]] += rnn_confs[lbl][pred]
for lbl in cnn_confs:
    for pred in cnn_confs[lbl]:
        #if pred != lbl:
        mat[freqmap[lbl], freqmap[pred]] -= cnn_confs[lbl][pred]
for cls,ct in class_counts.items():
    mat[freqmap[cls]] = np.true_divide(mat[freqmap[cls]], ct)

fig, ax = plt.subplots()
plt.matshow(mat, cmap='bwr', norm=DivergingNorm(0), fignum=0)
ax.set_ylabel('Labels')
ax.set_xlabel('Predictions')
ax.set_xticks(range(334))
ax.set_yticks(range(334))
ax.set_xticklabels([str(cls) for (cls, idx) in freqmap.items()], rotation=90)
ax.set_yticklabels([str(cls) for (cls, idx) in freqmap.items()])
plt.show()
