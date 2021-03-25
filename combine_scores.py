import sys
import pandas as pd
from tqdm import tqdm
import pdb

class Combine:
    def __init__(self):
        pass

    def normalize(self, X):
        minimum = min(X)
        maximum = max(X)
        norm_X = [(x - minimum) / (maximum - minimum) for x in X]
        return norm_X

    def read_floats(self, _file):
        negs = []
        poss = []
        for i, _line in enumerate(_file):
            line = _line.split()
            try:
                neg = float(line[1])
            except:
                pdb.set_trace()
            pos = float(line[2])
            negs.append(neg)
            poss.append(pos)
        return negs, poss

if __name__ == "__main__":
    c = Combine()
    labels = [x.strip() for x in open('../genpara/alignment scripts/pattern_scoring/col_7.txt', 'r').readlines()]
    srcs = [x.strip() for x in open('../genpara/alignment scripts/pattern_scoring/col_3.txt', 'r').readlines()]
    aligns = [x.strip() for x in open('../genpara/alignment scripts/pattern_scoring/col_4.txt', 'r').readlines()]
    origs = [x.strip() for x in open('../genpara/alignment scripts/pattern_scoring/col_6.txt', 'r').readlines()]
    paras = [x.strip() for x in open('../genpara/alignment scripts/pattern_scoring/col_5.txt', 'r').readlines()]

    two_way_scores = pd.read_csv('../genpara/alignment scripts/pattern_scoring/2-way-scores.txt', sep='\t')
    four_way_scores = pd.read_csv('../genpara/alignment scripts/pattern_scoring/4-way-scores.txt', sep='\t')

    # pdb.set_trace()
    # two_way_neg, two_way_pos = c.read_floats(two_way_scores)
    # four_way_neg, four_way_pos = c.read_floats(four_way_scores)

    two_way_neg_norm = c.normalize(two_way_scores['log_0'].tolist())
    two_way_pos_norm = c.normalize(two_way_scores['log_1'].tolist())
    four_way_neg_norm = c.normalize(four_way_scores['log_0'].tolist())
    four_way_pos_norm = c.normalize(four_way_scores['log_1'].tolist())

    header = ['log_0_2', 'log_1_2', 'log_0_4', 'log_1_4', labels.pop(0), srcs.pop(0), aligns.pop(0), origs.pop(0), paras.pop(0)]

    outfile = open(sys.argv[1], 'w')

    outfile.write('\t'.join(header) + '\n')

    for twns, twps, fwns, fwps, label, src, align, orig, para in zip(two_way_neg_norm,
                                                                     two_way_pos_norm,
                                                                     four_way_neg_norm,
                                                                     four_way_pos_norm,
                                                                     labels,
                                                                     srcs,
                                                                     aligns,
                                                                     origs,
                                                                     tqdm(paras)
                                                                     ):
        outfile.write('\t'.join([str(twns),
                                 str(twps),
                                 str(fwns),
                                 str(fwps),
                                 label,
                                 src,
                                 align,
                                 orig,
                                 para]) + '\n')