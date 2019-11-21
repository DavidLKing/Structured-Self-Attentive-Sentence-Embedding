from flask import Flask, render_template, request, url_for
import matplotlib.pyplot as plt
import numpy as np
import mpld3

app = Flask(__name__)

sents, preds, labels, reps = [], [], [], []
corrects = []
colors = ["51,204,204",
          "51,102,255",
          "255,0,255",
          "255,80,80",
          "255,255,0",
          "51,204,51"]


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
        rep = [[round(float(val),2) for val in v.split(' ')]
               for v in rep.split('|')]
        if first:
            print(rep)
            first = False
        reps.append(rep)

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

@app.route('/summary')
def summary():
    # count
    class_counts = {}
    class_corrects = {}
    for i in range(len(labels)):
        if labels[i] in class_counts:
            class_counts[labels[i]] += 1
        else:
            class_counts[labels[i]] = 1
            class_corrects[labels[i]] = 0
        if corrects[i]:
            class_corrects[labels[i]] += 1
    # sort values
    sorted_freqs = sorted([(count, lbl) for (lbl, count) in class_counts.items()])
    # group into quintiles
    quint_break = len(labels) / 5.0
    cum_sum = 0
    quint_sums = []
    quint_corrects = []
    k = 0
    for j in range(5):
        quint_sums.append(0)
        quint_corrects.append(0)
        while cum_sum < ((j+1)*quint_break):
            cum_sum += sorted_freqs[k][0]
            lbl = sorted_freqs[k][1]
            quint_sums[j] += class_counts[lbl]
            quint_corrects[j] += class_corrects[lbl]
            k += 1
    quint_accs = [float(a)/float(b) for a,b in zip(quint_corrects, quint_sums)]
    # plot
    width = 0.5
    fig, ax = plt.subplots()
    rects1 = plt.bar(np.arange(len(quint_accs)), quint_accs, width, color='#5E7175',
                     align='center', label='RNN Accuracy')
    ax.yaxis.grid(True, linestyle=':')
    ax.set_ylim([0.2, 1])
    ax.set_title('Model Accuracy by Quintiles')
    ax.set_ylabel('Accuracy')
    ax.legend()
    return mpld3.fig_to_html(fig)
