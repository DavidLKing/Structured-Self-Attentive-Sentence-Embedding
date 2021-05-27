from train import *
from util import Dictionary, get_base_parser

import torch
import torch.nn as nn
import torch.optim as optim

import time
import random
import os
import copy

# DLK hacking necessity!
import pandas as pd
import pdb
import sys
import json
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pickle as pkl

import nltk

def get_ranked_paras(paras, testing, metric):
    '''
    Convert pandas DataFrame into dict of {label : [ranked, paraphrases], label_2 : [...
    :param paras: Pandas DataFrame
    :param metric: column header
    :param testing: set of src and aligns that can't be used
    :return: dict of {label : [ranked, paraphrases], label_2 : [...
    '''
    ranked = {}
    ignored = 0
    print("ranking paraphrases")
    for label in tqdm(set(paras['label'])):
        ranked[label] = []
    for paraphrase in tqdm(paras[[metric, 'src', 'align', 'orig', 'para', 'label']].sort_values(by=metric).values):
        score = paraphrase[0]
        src = paraphrase[1]
        align = paraphrase[2]
        orig = paraphrase[3]
        para = paraphrase[4]
        para_label = paraphrase[5]
        # TODO I think there's a better way of iterating through all this
        # TODO remove print statements
        # if label == para_label:
        # if src in testing:
        #     # print("ignoring src", src, "in test")
        #     ignored += 1
        #     continue
        # elif align in testing:
        #     # print("ignoring align", align, "in test")
        #     ignored += 1
        #     continue
        # elif orig in testing:
        #     # print("ignoring orig", orig, "in test")
        #     ignored += 1
        #     continue
        # else:
        ranked[para_label].append((para, score))
        # clean
    pruned = 0
    clean_ranked = {}
    for label in ranked:
        if len(ranked[label]) > 0:
            clean_ranked[label] = ranked[label]
        else:
            pruned += 1
    print("Pruned {} empty labels".format(pruned))
    print("Ignored {} paraphrases from dev or test".format(ignored))
    return ranked

def jsonify_labels(labels):
    json_labels = []
    with open(labels, 'r') as labs:
        for label in labs:
            label = label.split(',')
            assert(len(label)==3)
            if label[0] != "":
                newlabel = {"label": int(label[0])}
                newlabel['text'] = nltk.tokenize.word_tokenize(label[1])
                json_labels.append(json.dumps(newlabel)  + '\n')
    return json_labels

def load_corrected(corr_lines, labels):
    pass

def load_splits(label_data, all_paras, metric, args):
    # modified from ye ol' get_splits funct.
    # data should be list of json entries:
    # ['{"label": 80, "text": ["does", "the", "pain", "radiate", "anywhere", "?"]}\n', ...]

    #make better dict than json stuff
    labels = {}
    for lab in label_data:
        lab = eval(lab)
        labels[' '.join(lab['text'])] = int(lab['label'])
    print("Loading data")
    datas = []
    for dataset in [args.train_data, args.val_data, args.test_data]:
        group = []
        num = 0
        print("dataset", dataset)
        testfile = open(dataset, 'r')
        for line in testfile:
            try:
                print("num", num)
                num += 1
                print("Line", line)
            except:
                pdb.set_trace()
        testfile.close()
        with open(dataset, 'r') as lines:
            for line in tqdm(lines):
                if not line.startswith(("#START")):
                    line = line.split('\t')
                    label = ' '.join(nltk.tokenize.word_tokenize(line[1]))
                    sent = nltk.tokenize.word_tokenize(line[0])
                    try:
                        label_num = labels[label]
                    except:
                        print("Failed to find label")
                        pdb.set_trace()
                    newstr = json.dumps({"label": int(label_num), "text": sent}) + '\n'
                    group.append(newstr)
        datas.append(group)
    data_train, data_val, data_test = datas
    # DLK para extraction
    if all_paras is not None:
        val_sents = set([' '.join(json.loads(x)['text']) for x in data_val])
        test_sents = set([' '.join(json.loads(x)['text']) for x in data_test])
        test_items = val_sents.union(test_sents)
        data_paras = get_ranked_paras(all_paras, test_items, metric)
    return data_train, data_val, data_test, data_paras
    
def sample(data_train, label_data, all_para, sample_rate, args):
    # DLK add paraphrases
    # dev_labels = [x.split(",")[0].split(":")[1].strip() for x in data_val]
    # test_labels = [x.split(",")[0].split(":")[1].strip() for x in data_test]
    # no_use_labels = dev_labels + test_labels
    # EXPERIMENTING
    # Quick dirtry rebuild of label dict
    sampled = 0

    string_to_label = {}
    label_to_string = {}
    for label_info in label_data:
        try:
            info = json.loads(label_info)
        except:
            print("error loating json")
            print("info = json.loads(label_info)")
            pdb.set_trace()
        assert ('label' in info)
        assert ('text' in info)
        label_text = ' '.join(info['text'])
        string_to_label[label_text] = info['label']
        label_to_string[info['label']] = label_text

    out_train = []
    uniform_train = []

    label_count = {}

    # TODO hack to make sure these don't effect other sampling schemas
    if args.indecrease == 'uniform':
        MAX = 100 # for uniform---should be same number as genpara infreq num
        REALMAX = 200

    for jsonitem in data_train:
        try:
            item = json.loads(jsonitem)
        except:
            print("error loating json")
            print("info = json.loads(jsonitem)")
            pdb.set_trace()
        item_label_int = item['label']
        item_label = label_to_string[item_label_int]

        if item_label not in label_count:
            label_count[item_label] = 0
        label_count[item_label] += 1

        # TODO some all_para[item_label] are still empty. How?
        if all_para is not None and random.random() < sample_rate and item_label in all_para and len(all_para[item_label]) > 0:
            # print("pre item", item)
            # TODO this isn't always working, why? - because of low freq filter
            paras = all_para[item_label]
            candidates = [x[0] for x in paras]
            scores = [x[1] for x in paras]
            # TODO
            '''
            Traceback (most recent call last):
              File "crossval.py", line 294, in <module>
                data_train = sample(pre_para_data_train, label_data, data_paras, sample_rate, args)
              File "crossval.py", line 118, in sample
                alts = random.choices(candidates, weights=scores)
              File "/home/david/miniconda2/envs/adam-rnn/lib/python3.6/random.py", line 362, in choices
                total = cum_weights[-1]
            IndexError: list index out of range
            '''
            alts = random.choices(candidates, weights=scores)
            # Weird bug from random.choices
            if type(alts) == list:
                alternative = random.choices(candidates, weights=scores)[0]
            elif type(alts) == str:
                alternative = random.choices(candidates, weights=scores)
            # alternative = alternative.replace('  ', ' ')
            splitalt = alternative.split()
            textalt = str(splitalt).replace(" '", ' "').replace("',", '",').replace("['", '["').replace("']", '"]')
            newline = '{"label": ' + str(item_label_int) + ',"text": ' + textalt + '}\n'

            # check the quality of the string
            try:
                json.loads(newline)
            except:
                print("Error loading json")
                print("json.loads(newline)")
                pdb.set_trace()

            if args.indecrease == 'uniform' and label_count[item_label] < MAX:
                uniform_train.append(newline)
            out_train.append(newline)

            sampled += 1
        else:
            out_train.append(jsonitem)
            if args.indecrease == 'uniform' and label_count[item_label] < MAX:
                uniform_train.append(jsonitem)

    if args.indecrease == 'uniform':
        for label in label_count:
            if label_count[label] < REALMAX:
                if all_para is not None and label in all_para and len(all_para[label]) > 0:
                    diff = REALMAX - label_count[label]
                    paras = all_para[label]
                    candidates = [x[0] for x in paras]
                    scores = [x[1] for x in paras]
                    label_int = string_to_label[label]
                    alts = random.choices(candidates, weights=scores, k=diff)
                    if type(alts) == list:
                        for alt in alts:
                            splitalt = alt.split()
                            textalt = str(splitalt).replace(" '", ' "').replace("',", '",').replace("['", '["').replace("']", '"]')
                            newline = '{"label": ' + str(label_int) + ',"text": ' + textalt + '}\n'
                            try:
                                json.loads(newline)
                            except:
                                print("Booooooo")
                                pdb.set_trace()
                            uniform_train.append(newline)
                    else:
                        print("Booooooo again!")
                        pdb.set_trace()

    print("Added {} samples to training data".format(sampled))
    return out_train, uniform_train

def get_quantiles(datas):
    label_num_str = [x.split('"')[2].strip(": ").strip(',') for x in datas]
    label_set = set(label_num_str)
    label_2_item_dict = {label_str: [] for label_str in label_set}
    # TODO maybe a readable for loop is better?
    [label_2_item_dict[x.split('"')[2].strip(": ").strip(',')].append(x) for x in datas]
    # I love python3
    label_dict = {key: 0 for key in label_set}
    for key in label_num_str: label_dict[key] += 1
    sorted_label_list = sorted(label_dict, key=label_dict.get, reverse=True)
    maximum = len(datas) // 5
    quant_1, quant_2, quant_3, quant_4, quant_5 = [[], [], [], [], []]
    quants = quant_1, quant_2, quant_3, quant_4, quant_5
    # quant_counts = []
    # print("sorted", len(sorted_label_list))
    for quant in quants:
        temp = []
        while len(temp) < maximum and len(sorted_label_list) > 0:
            label = sorted_label_list.pop(0)
            temp += label_2_item_dict[label]
            quant.append(label)
        # print(len(temp))
    # for i in quants:
    #     print(len(i))
    # print('total', sum([len(x) for x in quants]))
    # WHOOPS bad quants
    # 1, 2, 3, 4, and end
    # quant_idx_1, quant_idx_2, quant_idx_3, quant_idx_4, _ = [(len(sorted_label_list)//5) * (i+1) for i in range(5)]
    # quant_1 = sorted_label_list[0:quant_idx_1]
    # quant_2 = sorted_label_list[quant_idx_1:quant_idx_2]
    # quant_3 = sorted_label_list[quant_idx_2:quant_idx_3]
    # quant_4 = sorted_label_list[quant_idx_3:quant_idx_4]
    # quant_5 = sorted_label_list[quant_idx_4:]
    all_sorted = quant_1 + quant_2 + quant_3 + quant_4 + quant_5
    return quant_1, quant_2, quant_3, quant_4, quant_5, label_dict, all_sorted

def quantile_eval(quants, preds, targs, acc):
    quant_acc = []
    quant_macro_f1 = []
    quant_counts = []
    quant_num = 0
    for quant in quants:
        quant_num += 1
        quant_targs = []
        quant_preds = []
        quant_count = 0
        for targ, pred in zip(targs, preds):
            if str(targ) in quant:
                quant_count += 1
                quant_targs.append(targ)
                quant_preds.append(pred)
        quant_counts.append(quant_count)
        quant_macro_f1.append(f1_score(quant_targs, quant_preds, average='macro', labels=list(set(quant_targs))))
        quant_acc.append(accuracy_score(quant_targs, quant_preds))
    return quant_acc, quant_macro_f1, quant_counts

if __name__ == "__main__":
    # parse the arguments
    parser = get_base_parser()
    parser.add_argument('--data', type=str, default='',
                        help='location of the cross-validation data, should be a json file')
    parser.add_argument('--para-data', type=str, default=None,
                        help='location of weighed paraphrases')
    parser.add_argument('--train-data', type=str, default='',
                        help='location of the training data, should be a json file')
    parser.add_argument('--val-data', type=str, default='',
                        help='location of the development data, should be a json file')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file')
    parser.add_argument('--sample-rate', type=float, default=None,
                        help='rate to sample ranked paraphrases at')
    parser.add_argument('--save-pkl', type=str, default=None,
                        help='are we saving a pickle?')
    parser.add_argument('--metric', type=str, default=None,
                        help='head used for ranking')
    parser.add_argument('--indecrease', type=str, default=None,
                        help='1/x increasing or decrease per epoch training?')
    parser.add_argument('--label-data', type=str, default='',
                        help='location of the label map (int -> sentence) in json format')
    parser.add_argument('--device', type=str, default='',
                        help='cuda device number, ignored if GPU')
    args = parser.parse_args()
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda:{}".format(args.device))
            print("Using cuda device: {}".format(args.device))

    # pdb.set_trace()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) # ignored if not --cuda
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    n_token = len(dictionary)
    criterion = nn.CrossEntropyLoss()
    trip_criterion = nn.TripletMarginLoss(margin=0.0, p=2.0)
    print(args)
#    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
#    for i in range(args.batch_size):
#        for j in range(args.attention_hops):
#            I.data[i][j][j] = 1
#    I = I.to(device)

    print('Begin to load data.')
    all_data = open(args.data).readlines()
    # DLK hacking
    # quantiles = get_quantiles(all_data)
    # quant_1, quant_2, quant_3, quant_4, quant_5, label_counts, sorted_label_list = get_quantiles(all_data)
    # assert(sorted_label_list == quant_1 + quant_2 + quant_3 + quant_4 + quant_5)
    # quants = [quant_1, quant_2, quant_3, quant_4, quant_5]
    if args.para_data:
        all_para = pd.read_csv(args.para_data, sep='\t')
        sample_rate = float(args.sample_rate)
    else:
        all_para = None
        sample_rate = 0.0
    if args.save_pkl:
        SAVE = True
    else:
        SAVE = False
    # TODO possibly remove if we have a memroy issue
    relevants = {}
    # Please let this work
    # seems to work but it I think the vectors are leaking into between folds
    # if os.path.exists(args.word_vector):
    #     # config['word-vector']):
    #     print('Loading word vectors from', args.word_vector)
    #     word_embs = torch.load(args.word_vector)
    # else:
    #     sys.exit("Word vector file not found")
    # end
    label_data = jsonify_labels(args.label_data)
    fold_dev_losses = []
    fold_dev_accs = []
    fold_dev_boost_losses = []
    fold_dev_boost_accs = []
    fold_test_accs = []
    fold_test_losses = []
    logfile = open(args.out_log, "w")
    if args.sparsity == 'softmax':
        intrep = 'softmax'
    elif args.sparsity in ['L1', 'entropy', 'similarity']:
        intrep = 'sigmoid'

    fold_dev_quant_accs = []
    fold_dev_quant_macros = []
    fold_dev_quant_counts = []
    fold_test_quant_accs = []
    fold_test_quant_macros = []
    fold_test_quant_counts = []

    dev_all_preds = []
    dev_all_targs = []

    test_all_preds = []
    test_all_targs = []

    print('-' * 84)
    print('BEGINING')
    print('-' * 84)
    if args.no_bottleneck:
        model = Classifier({
                'dropout': args.dropout,
                'ntoken': n_token,
                'nlayers': args.nlayers,
                'nhid': args.nhid,
                'ninp': args.emsize,
                'pooling': args.pooling,
                'attention-unit': args.attention_unit,
                'attention-hops': args.attention_hops,
                'nfc': args.nfc,
                'ncat': args.ncat,
                'intrep': intrep,
                'dictionary': dictionary,
                'word-embs': args.word_vector,
                'class-number': args.class_number,
                'reserved': args.reserved
                })
    else:
        model = BottleneckClassifier({
                'dropout': args.dropout,
                'ntoken': n_token,
                'nlayers': args.nlayers,
                'nhid': args.nhid,
                'ninp': args.emsize,
                'pooling': 'all',
                'attention-unit': args.attention_unit,
                'attention-hops': args.attention_hops,
                'nfc': args.nfc,
                'ncat': args.ncat,
                'intrep': intrep,
                'dictionary': dictionary,
                'word-vector': args.word_vector,
                'class-number': args.class_number,
                'reserved': args.reserved
                })

    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
    print("Number of trainable params: " + str(pytorch_total_params))
    #print("bottleneck mtx device:")
    #print(model.bnWs[0].weight.device)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95)
    # remove SGD since disabling learning rate adjustments in training loop
    #elif args.optimizer == 'SGD':
    #    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    best_val_loss = None
    best_acc = None
    best_model = None

    best_preds = None
    best_targs = None

    #get the right splits
    pre_para_data_train, data_val, data_test, data_paras = load_splits(label_data, all_para, args.metric, args)
    for epoch in range(args.epochs):
        print('-' * 84)
        print('BEGIN STAGE 1 EPOCH ' + str(epoch))
        print('-' * 84)
        print(fold_test_accs)
        print('-' * 84)
        '''
        IDEA! EUREKA
        Okay, shuffle train but THEN do sampling. Should be able to import Adam/Prashant code wholish-sale
        '''
        # YEAH DLK HACKS
        if args.indecrease:
            indecrease = args.indecrease
            if indecrease:

                if indecrease == 'increase':
                    new_sample_rate = (float(epoch + 1) / args.epochs) * sample_rate
                elif indecrease == 'decrease':
                    new_sample_rate = (1 - (float(epoch) / args.epochs)) * sample_rate
                elif indecrease == 'uniform':
                    new_sample_rate = sample_rate
                elif indecrease == 'None':
                    indecrease = None
                    new_sample_rate = sample_rate
                else:
                    sys.exit("args.sample_rate can only be increase, decrease, uniform, or None")
                print("Changed sample rate from {} to {}".format(sample_rate, new_sample_rate))
        else:
            indecrease = None
            new_sample_rate = sample_rate


        data_train, uniform_train = sample(pre_para_data_train, label_data, data_paras, new_sample_rate, args)
        if args.indecrease == 'uniform':
            stage1_data_train = uniform_train
            stage2_data_train = data_train
        else:
            stage1_data_train = data_train
            stage2_data_train = data_train
        # <end>
        # pdb.set_trace()
        if args.shuffle:
            random.shuffle(stage1_data_train)
        # pdb.set_trace()
        model = train(model, stage1_data_train, dictionary, criterion,
                      optimizer, device, epoch, args)
        evaluate_start_time = time.time()
        val_loss, acc, preds, targs = evaluate(model, data_val, dictionary, criterion, device, args)
        # pdb.set_trace()
        print('-' * 84)
        fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
        macro = f1_score(targs, preds, average='macro', labels=list(set(targs)))
        # macro = f1_score(targs, preds, average='macro')
        print("| CHECKING MACRO F1:", macro)
        # quant_acc, quant_macro_f1, quant_counts = quantile_eval(quants, preds, targs, acc)
        # print("Quantiles:")
        # for quant, (acc, f1, count) in enumerate(zip(quant_acc, quant_macro_f1, quant_counts)):
        #     num = quant + 1
        #     print("\tNum: {}\tAccuracy: {}\tMacro F1: {}\tItem Count: {}".format(num, acc, f1, count))
        print('-' * 84)
        # Begin trip option
        #if not best_acc or acc > best_acc:
        #    save(model, args.save[:-3]+'.best_acc.pt')
        #    best_acc = acc
        #    best_model = copy.deepcopy(model)

        #wrongs, confusions, corrects, right_map, wrong_map = analyze_data(model,
        #                                                                 data_train,
        #                                                                  dictionary,
        #                                                                  device, args)
        #anchors, pos_exes, neg_exes = collect_triplets(data_train,
        #                                               wrongs,
        #                                               confusions,
        #                                               corrects,
        #                                               right_map,
        #                                               wrong_map)
        #if args.shuffle:
        #    z = list(zip(anchors, pos_exes, neg_exes))
        #    random.shuffle(z)
        #    anchors, pos_exes, neg_exes = zip(*z)
        #model = train_trips(model, anchors, pos_exes, neg_exes,
        #                    dictionary, criterion, trip_criterion, optimizer,
        #                    device, args, boost=True)

        #evaluate_start_time = time.time()
        #val_loss, acc = evaluate(model, data_val, dictionary,
        #                         criterion, device, args)
        #print('-' * 84)
        #fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
        #print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
        #print('-' * 84)
        # End trip option
        # Save the model, if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            save(model, args.save)
            best_val_loss = val_loss
        #else:  # if loss doesn't go down, divide the learning rate by 5.
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = param_group['lr'] * 0.2
        if not best_acc or acc > best_acc:
            save(model, args.save[:-3]+'.best_acc.pt')
            best_acc = acc
            best_model = copy.deepcopy(model)

            best_targs = targs
            best_preds = preds

        save(model, args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch))

    print('-' * 84)
    if args.stage2 > 0:
        print('ANALYZING ERRORS')
        model = best_model
        model.to(device)
        model.flatten_parameters()
        #wrongs, confusions, corrects, right_map, wrong_map = analyze_data(model,
        #                                                                  data_train,
        #                                                                  dictionary,
        #                                                                  device, args)
        #model.boost()
        #reinitialize optimizer
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr*0.25, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        elif args.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr*0.25, rho=0.95)
        trip_criterion = nn.TripletMarginLoss(margin=0.0, p=2.0)
        best_boost_val_loss = None
        best_boost_acc = None
        for epoch in range(args.stage2):
            print('-' * 84)
            print('BEGIN STAGE 2 EPOCH ' + str(epoch))
            print('-' * 84)
            #anchors, pos_exes, neg_exes = collect_triplets(data_train,
            #                                               wrongs,
            #                                               confusions,
            #                                               corrects,
            #                                               right_map,
            #                                               wrong_map)
            if args.shuffle:
                #z = list(zip(anchors, pos_exes, neg_exes))
                #random.shuffle(z)
                #anchors, pos_exes, neg_exes = zip(*z)
                random.shuffle(stage2_data_train)
            #model = train_trips(model, anchors, pos_exes, neg_exes,
            #                    dictionary, trip_criterion, optimizer,
            #                    device, args, boost=True)
            model = train(model, stage2_data_train, dictionary,
                          criterion, optimizer, device, epoch, args, boost=True)
            evaluate_start_time = time.time()
            val_loss, acc, preds, targs = evaluate(model, data_val, dictionary,
                                     criterion, device, args)
            print('-' * 84)
            fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
            print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
            # macro = f1_score(targs, preds, average='macro')
            macro = f1_score(targs, preds, average='macro', labels=list(set(targs)))
            print("| CHECKING MACRO F1:", macro)
            print("| btw, this is the stage 2 check")
            print('-' * 84)
            # quant_acc, quant_macro_f1, quant_counts = quantile_eval(quants, preds, targs, acc)
            # print("Quantiles:")
            # for quant, (acc, f1, count) in enumerate(zip(quant_acc, quant_macro_f1, quant_counts)):
            #     num = quant + 1
            #     print("\tNum: {}\tAccuracy: {}\tMacro F1: {}\tItem Count: {}".format(num, acc, f1, count))
            # Save the model, if the validation loss is the best we've seen so far.
            if not best_boost_val_loss or val_loss < best_boost_val_loss:
                save(model, args.save)
                best_boost_val_loss = val_loss
            #else:  # if loss doesn't go down, divide the learning rate by 5.
                #for param_group in optimizer.param_groups:
                #    param_group['lr'] = param_group['lr'] * 0.2
            if not best_boost_acc or acc > best_boost_acc:
                save(model, args.save[:-3]+'.best_acc.pt')
                best_boost_acc = acc
                best_model = copy.deepcopy(model)
            save(model, args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch))
        fold_dev_boost_losses += [best_boost_val_loss]
        fold_dev_boost_accs += [best_boost_acc]

    fold_dev_losses += [best_val_loss]
    fold_dev_accs += [best_acc]
    dev_all_preds += best_preds
    dev_all_targs += best_targs

    dev_quant_accs = []
    dev_quant_macros = []
    dev_quant_counts = []
    test_quant_accs = []
    test_quant_macros = []
    test_quant_counts = []

    # quant_acc, quant_macro_f1, quant_counts = quantile_eval(quants, best_preds, best_targs, best_acc)
    # print("| Quantiles:")
    # for quant, (acc, f1, count) in enumerate(zip(quant_acc, quant_macro_f1, quant_counts)):
        # num = quant + 1
        # print("| \tNum: {}\tAccuracy: {}\tMacro F1: {}\tItem Count: {}".format(num, acc, f1, count))
        # dev_quant_accs.append(acc)
        # dev_quant_macros.append(f1)
        # dev_quant_counts.append(count)
    print('-' * 84)

    if args.eval_on_test:
        evaluate_start_time = time.time()
        best_model.to(device)
        best_model.flatten_parameters()
        test_loss, acc, preds, targs = evaluate(best_model, data_test, dictionary, criterion, device, args, outlog=logfile)
        fold_test_losses += [test_loss]
        fold_test_accs += [acc]
        test_all_preds += preds
        test_all_targs += targs
        print('-' * 84)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))

        # macro = f1_score(targs, preds, average='macro')
        macro = f1_score(targs, preds, average='macro', labels=list(set(targs)))
        print("| CHECKING MACRO F1:", macro)
        print("| btw, this is the eval_on_test check")
        print('-' * 84)
        # quant_acc, quant_macro_f1, quant_counts = quantile_eval(quants, preds, targs, acc)
        # print("| Quantiles:")
        # for quant, (acc, f1, count) in enumerate(zip(quant_acc, quant_macro_f1, quant_counts)):
            # num = quant + 1
            # print("| \tNum: {}\tAccuracy: {}\tMacro F1: {}\tItem Count: {}".format(num, acc, f1, count))
            # test_quant_accs.append(acc)
            # test_quant_macros.append(f1)
            # test_quant_counts.append(count)

    # fold_dev_quant_accs.append(dev_quant_accs)
    # fold_dev_quant_macros.append(dev_quant_macros)
    # fold_dev_quant_counts.append(dev_quant_counts)
    # fold_test_quant_accs.append(test_quant_accs)
    # fold_test_quant_macros.append(test_quant_macros)
    # fold_test_quant_counts.append(test_quant_counts)


    relevants = {
        'fold dev quant accs': fold_dev_quant_accs,
        'fold dev quant macros': fold_dev_quant_macros,
        'fold dev quant counts': fold_dev_quant_counts,
        'fold test quant accs': fold_test_quant_accs,
        'fold test quant macros': fold_test_quant_macros,
        'fold test quant counts': fold_test_quant_counts
    }

    print('-' * 84)
    fmt = '| dev average | test loss (pure) {:5.4f} | Acc {:8.4f}'
    fmt2 = '| dev boost average | test loss (pure) {:5.4f} | Acc {:8.4f}'
    # avg_dev_loss = sum(fold_dev_losses)/float(args.xfolds)
    # avg_dev_acc = sum(fold_dev_accs)/float(args.xfolds)
    avg_dev_loss = sum(fold_dev_losses)
    avg_dev_acc = sum(fold_dev_accs)

    relevants['dev acc'] = avg_dev_acc

    print(fmt.format(avg_dev_loss, avg_dev_acc))
    if args.stage2 > 0:
        # avg_dev_boost_loss = sum(fold_dev_boost_losses)/float(args.xfolds)
        # avg_dev_boost_acc = sum(fold_dev_boost_accs)/float(args.xfolds)
        avg_dev_boost_loss = sum(fold_dev_boost_losses)
        avg_dev_boost_acc = sum(fold_dev_boost_accs)
        print(fmt.format(avg_dev_boost_loss, avg_dev_boost_acc))
    print('-' * 84)

    print('-' * 84)
    # dev_macro = f1_score(dev_all_targs, dev_all_preds, average="macro")
    dev_macro = f1_score(dev_all_targs, dev_all_preds, average='macro', labels=list(set(dev_all_targs)))

    relevants['dev macro'] = dev_macro

    # TODO is it better to jsut get all of those number here?

    dev_quant_accs_general = []
    dev_quant_macro_general = []
    dev_quant_counts_general = []

    print("macro on dev:", dev_macro)
    print('-' * 84)
    # quant_acc, quant_macro_f1, quant_counts = quantile_eval(quants, dev_all_preds, dev_all_targs, avg_dev_acc)
    # print("Quantiles:")
    # for quant, (acc, f1, count) in enumerate(zip(quant_acc, quant_macro_f1, quant_counts)):
        # num = quant + 1
        # dev_quant_accs_general.append(acc)
        # dev_quant_macro_general.append(f1)
        # dev_quant_counts_general.append(count)
        # print("| \tNum: {}\tAccuracy: {}\tMacro F1: {}\tItem Count: {}".format(num, acc, f1, count))

    test_quant_accs_general = []
    test_quant_macro_general = []
    test_quant_counts_general = []

    if args.eval_on_test:
        print('-' * 84)
        fmt = '| test average | test loss (pure) {:5.4f} | Acc {:8.4f}'
        # avg_test_loss = sum(fold_test_losses)/float(args.xfolds)
        # avg_test_acc = sum(fold_test_accs)/float(args.xfolds)
        avg_test_loss = sum(fold_test_losses)
        avg_test_acc = sum(fold_test_accs)

        relevants['test acc'] = avg_test_acc

        print(fmt.format(avg_test_loss, avg_test_acc))
        print('-' * 84)
        print('-' * 84)

        # test_macro = f1_score(test_all_targs, test_all_preds, average="macro")
        test_macro = f1_score(test_all_targs, test_all_preds, average='macro', labels=list(set(test_all_targs)))

        relevants['test macro'] = test_macro

        print("| macro on test:", test_macro)
        print('-' * 84)
        # quant_acc, quant_macro_f1, quant_counts = quantile_eval(quants, test_all_preds, test_all_targs, avg_test_acc)
        # print("| Quantiles:")
        # for quant, (acc, f1, count) in enumerate(zip(quant_acc, quant_macro_f1, quant_counts)):
            # num = quant + 1
            # test_quant_accs_general.append(acc)
            # test_quant_macro_general.append(f1)
            # test_quant_counts_general.append(count)
            # print("| \tNum: {}\tAccuracy: {}\tMacro F1: {}\tItem Count: {}".format(num, acc, f1, count))
        print('-' * 84)

    # relevants['dev quant accs general'] = dev_quant_accs_general
    # relevants['dev quant macro general'] = dev_quant_macro_general
    # relevants['dev quant counts general'] = dev_quant_counts_general

    # relevants['test quant accs general'] = test_quant_accs_general
    # relevants['test quant macro general'] = test_quant_macro_general
    # relevants['test quant counts general'] = test_quant_counts_general

    if SAVE:
        print("| Saving relevant info to pickle", args.save_pkl)
        pkl_file = open(args.save_pkl, 'wb')
        pkl.dump(relevants, pkl_file)
        pkl_file.close()

    logfile.close()
    exit(0)
    
