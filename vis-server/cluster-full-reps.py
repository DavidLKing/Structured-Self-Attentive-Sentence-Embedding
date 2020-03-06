import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
                counts[i] = current_count
    distances = np.ones_like(model.children_[:,0])
    linkage_matrix = np.column_stack([model.children_, distances,
                                      counts]).astype(float)
    print(linkage_matrix.shape)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='',
                    help='choices.csv output file from model')
parser.add_argument('--fold', type=int, default=0,
                    help='which fold to analyze')
#parser.add_argument('--fold-size', type=int, default=683,
#                    help='number of points in the fold')
#parser.add_argument('--rep-size', type=int, default=512,
#                    help='number of dimensions in each head representation')

args = parser.parse_args()


sents, preds, labels, reps, folds = [], [], [], [], []
max_words = []

with open(args.data, 'r') as f:
    for line in f:
        sent, pred, label, rep, fold = line.strip().split('\t')
        fold = int(fold)
        if fold == args.fold:
            sent = sent.split(' ')
            sent = [tok.split('|') for tok in sent]
            nheads = (len(sent[0])-1)
            maxword = [""]*nheads
            max_atts = [0.]*nheads
            for tok in sent:
                for i in range(1,len(tok)):
                    tok[i] = float(tok[i])
                    if tok[i] > max_atts[i-1]:
                        maxword[i-1] = tok[0]
                        max_atts[i-1] = tok[i]
            max_words.append(maxword)
            #sent = [[el for el in tok if el > 0] for tok in sent]
            sents.append(sent)
            labels.append(int(label))
            preds.append(int(pred))
            folds.append(int(fold))
            rep = [[float(val) for val in v.split(' ')]
                   for v in rep.split('|')]
            reps.append(rep)

X = np.array(reps)
#clust = AgglomerativeClustering(n_clusters=None, distance_threshold=15.0)
#clust = DBSCAN(eps=0.4, min_samples=2, metric="cosine")
#clust = clust.fit(X[:,0,:].squeeze())
Z = linkage(X[:,0,:].squeeze())
lbls = [max_words[i][0] for i in range(len(max_words))]
fig, ax = plt.subplots()
dn = dendrogram(Z, labels=lbls, leaf_font_size=8)
plt.show()

##########################
#clusters = {}
#for i in range(len(clust.labels_)):
#    if not clust.labels_[i] in clusters:
#        clusters[clust.labels_[i]] = []
#    clusters[clust.labels_[i]] += [max_words[i][0]]

#for k,v in clusters.items():
#    print(k, v)
##########################


#print(clust.children_)
#print(clust.n_leaves_)
#print(clust.n_connected_components_)

#plot_dendrogram(clust)
#plt.show()
