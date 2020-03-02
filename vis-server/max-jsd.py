import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
import numpy as np
from scipy.spatial.distance import jensenshannon

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

class_jsd = {}
rnn_accs = {}
cnn_accs = {}
diff_accs = {}
for lbl in range(334):
    p = np.zeros((334,), dtype=float)
    q = np.zeros((334,), dtype=float)
    for pred in rnn_confs[lbl]:
        p[pred] += rnn_confs[lbl][pred]
        if pred == lbl:
            rnn_accs[lbl] = float(rnn_confs[lbl][pred]) / class_counts[lbl]
    for pred in cnn_confs[lbl]:
        q[pred] += cnn_confs[lbl][pred]
        if pred == lbl:
            cnn_accs[lbl] = float(cnn_confs[lbl][pred]) / class_counts[lbl]
    p /= class_counts[lbl]
    q /= class_counts[lbl]
    jsd = jensenshannon(p, q)
    class_jsd[lbl] = jsd
    if lbl not in rnn_accs:
        rnn_accs[lbl] = 0.
    if lbl not in cnn_accs:
        cnn_accs[lbl] = 0.
    diff_accs[lbl] = rnn_accs[lbl] - cnn_accs[lbl]

max_divs = reversed(sorted([(div*diff_accs[cls], cls) for (cls, div) in class_jsd.items()]))

for (divacc, cls) in list(max_divs):
    if class_counts[cls] > 4:
        fmt = "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:d}\t{4:d}"
        print(fmt.format(float(divacc),
                         float(diff_accs[cls]),
                         float(class_jsd[cls]),
                         cls,
                         class_counts[cls]))
