from flask import Flask, render_template, request, url_for
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support
import mpld3
from mpld3 import plugins
import json

app = Flask(__name__)

sents, preds, labels, reps = [], [], [], []
base_sents, base_preds, base_labels, base_corrects = [], [], [], []
corrects = []
colors = ["51,204,204",
          "51,102,255",
          "255,0,255",
          "255,80,80",
          "255,255,0",
          "51,204,51",
          "0,102,102",
          "0,0,204",
          "153,0,204",
          "153,0,51",
          "153,102,51",
          "0,102,0"]


def discretize(head, val):
    val = float(val)
    styleclass = ""
    if val < 0.2:
        styleclass = ""
    elif val < 0.4:
        styleclass = "head"+str(head)+"-1"
    elif val < 0.6:
        styleclass = "head"+str(head)+"-2"
    elif val < 0.8:
        styleclass = "head"+str(head)+"-3"
    else:
        styleclass = "head"+str(head)+"-4"
    return styleclass

with open("choices.csv", 'r') as f:
    first = True
    for line in f:
        sent, pred, label, rep = line.strip().split('\t')
        sent = sent.split(' ')
        sent = [tok.split('|') for tok in sent]
        for tok in sent:
            for i in range(1,len(tok)):
                tok[i] = round(float(tok[i]), 2)
        #sent = [[el for el in tok if el > 0] for tok in sent]
        sents.append(sent)
        labels.append(int(label))
        preds.append(int(pred))
        if rep != "NA":
            rep = [[round(float(val),2) for val in v.split(' ')]
                   for v in rep.split('|')]
            if first:
                print(rep)
                first = False
        reps.append(rep)

with open("cnn-baseline-choices.csv", 'r') as f:
    for line in f:
        sent, pred, label, correct = line.strip().split('\t')
        base_sents.append(sent)
        base_labels.append(int(label))
        base_preds.append(int(pred))
        base_corrects.append(bool(int(correct)))

lb_lookup = {}
#with open("vp16.superfix+cs17.known.labels.txt") as f:
with open("vp16+cs17.strict.labels.txt") as f:
    for line in f:
        lb_int, lb_txt = line.strip().split('\t')
        lb_int = int(lb_int)
        lb_lookup[lb_int] = lb_txt
        
corrects = [a == b for a,b in zip(labels, preds)]
        
@app.route('/')
def showit():
    start = request.args.get('start', 0, type=int)
    show = request.args.get('show', 50, type=int)
    last = start+show
    label_selector = request.args.get('label', None, type=int)
    selected_sents = []
    selected_labels = []
    selected_preds = []
    selected_corrects = []
    selected_reps = []
    if label_selector:
        for i in range(len(sents)):
            if labels[i] == label_selector:
                selected_sents.append(sents[i])
                selected_labels.append(labels[i])
                selected_preds.append(preds[i])
                selected_corrects.append(corrects[i])
                selected_reps.append(reps[i])
    else:
        selected_sents = sents
        selected_labels = labels
        selected_preds = preds
        selected_corrects = corrects
        selected_reps = reps
    show_sents = selected_sents[start:last]
    show_labels = selected_labels[start:last]
    show_preds = selected_preds[start:last]
    show_corrects = selected_corrects[start:last]
    show_reps = selected_reps[start:last]
    next_start = last
    last_start = start - show if (start - show > 0) else 0
    selector = ""
    if label_selector:
        selector += "&label="+str(label_selector)
    next_page = url_for('showit')+"?start="+str(next_start)+"&show="+str(show)+selector
    last_page = url_for('showit')+"?start="+str(last_start)+"&show="+str(show)+selector
    nav = {"next_page":next_page, "last_page":last_page}
    return render_template('data.html', sents=show_sents, colors=colors,
                           labels=show_labels, preds=show_preds,
                           reps=show_reps, corrects=show_corrects, nav=nav)

def count_corrects(label_list, correct_list):
    class_counts = {}
    class_corrects = {}
    for i in range(len(label_list)):
        curr_lbl = label_list[i]
        if curr_lbl in class_counts:
            class_counts[curr_lbl] += 1
        else:
            class_counts[curr_lbl] = 1
            class_corrects[curr_lbl] = 0
        if correct_list[i]:
            class_corrects[curr_lbl] += 1
    return class_counts, class_corrects

@app.route('/summary')
def summary():
    # count
    class_counts, class_corrects = count_corrects(labels, corrects)
    base_cls_counts, base_cls_corrects = count_corrects(base_labels, base_corrects)
    # sort values
    sorted_freqs = sorted([(count, lbl) for (lbl, count) in class_counts.items()])
    # group into quintiles
    quint_break = len(labels) / 5.0
    cum_sum = 0
    quint_sums = []
    quint_corrects = []
    base_qnt_corrects = []
    k = 0
    for j in range(5):
        quint_sums.append(0)
        quint_corrects.append(0)
        base_qnt_corrects.append(0)
        while cum_sum < ((j+1)*quint_break):
            cum_sum += sorted_freqs[k][0]
            lbl = sorted_freqs[k][1]
            quint_sums[j] += class_counts[lbl]
            quint_corrects[j] += class_corrects[lbl]
            base_qnt_corrects[j] += base_cls_corrects[lbl]
            k += 1
    quint_accs = [float(a)/float(b) for a,b in zip(quint_corrects, quint_sums)]
    base_qnt_accs = [float(a)/float(b) for a,b in zip(base_qnt_corrects, quint_sums)]
    # plot
    width = 0.3
    fig, ax = plt.subplots()
    rects1 = plt.bar(np.arange(len(base_qnt_accs)) - (width/2),
                     base_qnt_accs,
                     width,
                     color='#74B8EF',
                     align='center',
                     label='CNN Accuracy')
    rects2 = plt.bar(np.arange(len(quint_accs)) + (width/2),
                     quint_accs,
                     width,
                     color='#5E7175',
                     align='center',
                     label='RNN Accuracy')
    ax.yaxis.grid(True, linestyle=':')
    ax.set_ylim([0.2, 1])
    ax.set_title('Model Accuracy by Quintiles')
    ax.set_ylabel('Accuracy')
    ax.legend()
    overall_acc = round(sum(quint_corrects)/float(sum(quint_sums)),3)
    baseline_acc = round(sum(base_qnt_corrects)/float(sum(quint_sums)),3)
    exp_prfs = precision_recall_fscore_support(np.array(labels),
                                               np.array(preds),
                                               average='macro')
    base_prfs = precision_recall_fscore_support(np.array(base_labels),
                                                np.array(base_preds),
                                                average='macro')
    plt.savefig('quintile-acc.png', bbox_inches='tight', dpi=300)
    chart = json.dumps(mpld3.fig_to_dict(fig))
    reverse_freqs = list(reversed(sorted_freqs))
    cls_labels = [lbl for (count, lbl) in reverse_freqs]
    cls_accs = [round(float(class_corrects[lbl])/float(class_counts[lbl]),3)
                for lbl in cls_labels]
    base_accs = [round(float(base_cls_corrects[lbl])/float(base_cls_counts[lbl]),3)
                 for lbl in cls_labels]
    beats = []
    for a,b in zip(cls_accs, base_accs):
        if a > b:
            beats.append(1)
        elif a == b:
            beats.append(0)
        else:
            beats.append(-1)
    cls_lb_txts = [lb_lookup[lbl] for lbl in cls_labels]
    return render_template('summary.html', chart=chart, cls_labels=cls_lb_txts,
                           accs=cls_accs, base_accs=base_accs, overall=overall_acc,
                           baseline=baseline_acc, beats=beats, exp_prfs=exp_prfs,
                           base_prfs=base_prfs)

@app.route('/clusters')
def show_tsne():
    np_rep = np.array(reps)
    tenth = len(labels)//10
    X = np_rep[:tenth,:,:]
    tsne = TSNE(n_components=2, init='random')
    np_corr = np.array(corrects[:tenth])
    mark_col = np.zeros_like(np_corr)
    mark_col[np_corr] = 1
    np_lbls = np.array(labels[:tenth])
    num_heads = X.shape[1]
    tX = np.zeros((tenth, num_heads, 2))
    #dumb = np.arange(tenth, dtype=np.float)/tenth
    #dumb = dumb.reshape((-1,1))
    fig, ax = plt.subplots(num_heads//2, 6, figsize=(24,12))
    for i in range(num_heads):
        tX[:,i,:] = tsne.fit_transform(X[:,i,:])
    #print(tX[:10,:])
    tips = np_lbls.tolist()
    for i in range(num_heads):
        points = ax[i//2,i%2].scatter(tX[:,i,0], tX[:,i,1], c=mark_col, s=30, edgecolors='k', alpha=0.6)
        tooltip = plugins.PointLabelTooltip(points, labels=tips)
        plugins.connect(fig, tooltip)
        points = ax[i//2,(i%2)+2].scatter(tX[:,i,0], tX[:,(i+1)%num_heads,0], alpha=0)
        points = ax[i//2,(i%2)+4].scatter(tX[:,i,0], tX[:,(i+2)%num_heads,1], alpha=0)
    #for i, txt in enumerate(labels[:680]):
    #    ax.annotate(txt, (tX[i,0], tX[i,1]))
    
    plugins.connect(fig, plugins.LinkedBrush(points))
    #chart = json.dumps(mpld3.fig_to_dict(fig))
    #ax.legend()
    mpld3.show()
    #return mpld3.fig_to_html(fig)
    return render_template('clusters.html', chart=chart)
