from __future__ import print_function
from models import *

from util import Dictionary, get_base_parser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.autograd import Variable

import json
import time
import random
import os



def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True),
                         2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(data, dictionary, is_train=True):
    """Package data for training / evaluation."""
    data = [json.loads(x) for x in data]
    dat = [[dictionary.word2idx[y] for y in x['text']] for x in data]
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = [x['label'] for x in data]
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    with torch.set_grad_enabled(is_train):
        dat = torch.tensor(dat, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
    return dat.t(), targets

def evaluate(model, data_val, dictionary, criterion, device, args, outlog=None):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
        last = min(len(data_val), i+args.batch_size)
        intoks = data_val[i:last]
        data, targets = package(intoks, dictionary, is_train=False)
        data, targets = data.to(device), targets.to(device)
        hidden = model.init_hidden(data.size(1))
        output, attention, intermediate = model.forward(data, hidden)
        #if (i == 0):
        #    print(model.encoder.ws2.weight)
        output_flat = output.view(data.size(1), -1)
        total_loss += criterion(output_flat, targets).item()
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float()).item()
        if outlog is not None:
            for i in range(len(intoks)):
                in_words = json.loads(intoks[i])["text"]
                atts = ["|".join([str(a) for a in attention[i,:,k].tolist()])
                        for k in range(len(in_words))]
                in_atts = [a+"|"+b for a,b in zip(in_words, atts)]
                inputstr = " ".join(in_atts)
                if intermediate is not None:
                    vbs = [" ".join([str(v) for v in intermediate[i,j,:].tolist()])
                           for j in range(intermediate.size(1))]
                    inter = "|".join(vbs)
                else:
                    inter = "NA"
                logstr = "\t".join((inputstr, str(prediction[i].item()),
                                    str(targets[i].item()), inter))
                outlog.write(logstr + "\n")
    avg_batch_loss = total_loss / (len(data_val) // args.batch_size)
    acc = total_correct / len(data_val)
    return avg_batch_loss, acc

def analyze_data(model, data_train, dictionary, device, args):
    """run model against training set to find confusions"""
    model.eval()  # turn on the eval() switch to disable dropout
    wrong_list = []
    confusions = []
    correct_lbls = []
    right_map = {}
    wrong_map = {}
    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        last = min(len(data_train), i+args.batch_size)
        intoks = data_train[i:last]
        data, targets = package(intoks, dictionary, is_train=False)
        data, targets = data.to(device), targets.to(device)
        hidden = model.init_hidden(data.size(1))
        output, attention, intermediate = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        # Need: wrong idxs, correct examples of same class, correct confusions
        wrong = prediction != targets
        wrong = wrong.tolist()
        py_tgts = targets.tolist()
        py_pred = prediction.tolist()
        for idx,j in enumerate(range(i, last)):
            tgt = py_tgts[idx]
            if wrong[idx]:
                wrong_list += [j]
                confusions += [py_pred[idx]]
                correct_lbls += [tgt]
                if tgt in wrong_map:
                    wrong_map[tgt] += [j]
                else:
                    wrong_map[tgt] = [j]
            else:
                if tgt in right_map:
                    right_map[tgt] += [j]
                else:
                    right_map[tgt] = [j]
    return wrong_list, confusions, correct_lbls, right_map, wrong_map

def collect_triplets(data_train, wrong_list, confusions,
                     correct_lbls, right_map, wrong_map):
    anchors = wrong_list
    pos_exes = []
    neg_exes = []
    for i in range(len(wrong_list)):
        tgt = correct_lbls[i]
        # if a member of the class was classified correctly, use it
        if tgt in right_map:
            pos_exes.append(random.choice(right_map[tgt]))
        else: # otherwise, use a random incorrect example
            # todo?: try random point
            pos_exes.append(random.choice(wrong_map[tgt]))
        # should always be a member of the confused class, or it
        # wouldn't have been confused.
        neg_exes.append(random.choice(right_map[confusions[i]]))
    return anchors, pos_exes, neg_exes


def train(model, data_train, dictionary, criterion, optimizer, device, args, boost=False):
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    I = I.to(device)
    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        data, targets = package(data_train[i:i+args.batch_size], dictionary, is_train=True)
        data, targets = data.to(device), targets.to(device)
        hidden = model.init_hidden(data.size(1))
        output, attention, intermediate = model.forward(data, hidden)
        
        if not boost:
            nheads = args.attention_hops - args.reserved
            intermediate = intermediate[:,:nheads,:]
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.item()

        if attention is not None:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += args.penalization_coeff * extra_loss
        if args.sparsity == 'L1':
            sparsity_penalty = torch.mean(torch.mean(intermediate.norm(p=1, dim=2), dim=1)) # TODO: -1?
            loss += args.sparsity_coeff * sparsity_penalty.item()
        elif args.sparsity == 'entropy':
            batch_sums = torch.sum(intermediate, dim=0)
            batch_head_ps = F.normalize(batch_sums, p=1, dim=1)
            batch_head_logps = torch.log(batch_head_ps)
            batch_head_plogp = batch_head_ps * batch_head_logps
            avg_entropy = -1.0 * torch.mean(torch.sum(batch_head_plogp, dim=1))
            maxent = torch.log(torch.tensor(intermediate.size(2)).double()).item()
            loss += args.sparsity_coeff * (maxent - avg_entropy.item())
        elif args.sparsity == 'similarity':
            # maximize similarity of representations *across a batch*,
            # independently for each head
            # first switch batch dimension to 1, and head dim to 0.
            int_normed = F.normalize(intermediate, p=1, dim=2)
            head_first = int_normed.transpose(0,1)
            head_firstT = head_first.transpose(1,2)
            # calculate "batch" (head) average frobenius norm as penalty (bonus)
            sim = torch.bmm(head_first, head_firstT)
            head_avg_fro = Frobenius(sim)
            max_fro = torch.ones_like(sim[0,:,:]).norm() # frobenius default
            loss += args.sparsity_coeff * (max_fro - head_avg_fro)
            
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:

            log(start_time, len(data_train), total_loss,
                total_pure_loss, batch, args)
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()
            
#            for item in model.parameters():
#                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
#            print model.encoder.ws2.weight.grad.data
#            exit()
    return model

def train_trips(model, anchors, pos_exes, neg_exes,
                dictionary, criterion, optimizer, device, args, boost=False):
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    I = I.to(device)
    for batch, i in enumerate(range(0, len(anchors), args.batch_size)):
        data_a, targets_a = package(anchors[i:i+args.batch_size],
                                    dictionary, is_train=True)
        data_p, targets_p = package(pos_exes[i:i+args.batch_size],
                                    dictionary, is_train=True)
        data_n, targets_n = package(neg_exes[i:i+args.batch_size],
                                    dictionary, is_train=True)
        data_a, targets_a = data_a.to(device), targets_a.to(device)
        data_p, targets_p = data_p.to(device), targets_p.to(device)
        data_n, targets_n = data_n.to(device), targets_n.to(device)
        hidden = model.init_hidden(data_a.size(1))
        output_a, attention_a, intermediate_a = model.forward(data_a, hidden)
        #todo?: freeze model for non-anchors
        output_p, attention_p, intermediate_p = model.forward(data_p, hidden)
        output_n, attention_n, intermediate_n = model.forward(data_n, hidden)
        nheads = args.attention_hops - args.reserved
        if not boost:
            intermediate_a = intermediate_a[:,:nheads,:]
            intermediate_p = intermediate_p[:,:nheads,:]
            intermediate_n = intermediate_n[:,:nheads,:]
        sub_loss = torch.tensor(0.)
        for i in range(nheads):
            a = intermediate_a[:,i,:].squeeze()
            p = intermediate_p[:,i,:].squeeze()
            n = intermediate_n[:,i,:].squeeze()
            sub_loss += criterion(a, p, n)
        h = torch.tensor(nheads, dtype=torch.double)
        loss = sub_loss / h
        total_pure_loss += loss.item()

        # todo?: add back cross-entropy
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            log(start_time, len(anchors), total_loss,
                total_pure_loss, batch, args)
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()
            
#            for item in model.parameters():
#                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
#            print model.encoder.ws2.weight.grad.data
#            exit()
    return model

def log(start_time, size, total_loss, total_pure_loss, batch, args):
    elapsed = time.time() - start_time
    total_batches = size // args.batch_size
    batch_time = elapsed * 1000 / args.log_interval
    batch_loss = total_loss / args.log_interval
    pure_batch_loss = total_pure_loss / args.log_interval
    print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'
          .format(batch, total_batches, batch_time,
                  batch_loss, pure_batch_loss))

def save(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model, f)
        f.close()
    

if __name__ == '__main__':
    # parse the arguments
    parser = get_base_parser()
    parser.add_argument('--train-data', type=str, default='',
                        help='location of the training data, should be a json file')
    parser.add_argument('--val-data', type=str, default='',
                        help='location of the development data, should be a json file')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file')
    parser.add_argument('--test-model', type=str, default='',
                        help='path to load model to test from')
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) # ignored if not --cuda
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)

    criterion = nn.CrossEntropyLoss()

    if args.test_model == '':
        best_val_loss = None
        best_acc = None

        n_token = len(dictionary)
        model = Classifier({
            'dropout': args.dropout,
            'ntoken': n_token,
            'nlayers': args.nlayers,
            'nhid': args.nhid,
            'ninp': args.emsize,
            'pooling': 'all',
            'attention-unit': args.attention_unit,
            'attention-hops': args.attention_hops,
            'nfc': args.nfc,
            'dictionary': dictionary,
            'word-vector': args.word_vector,
            'class-number': args.class_number
        })
        model = model.to(device)

        print(args)

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise Exception('For other optimizers, please add it yourself. '
                            'supported ones are: SGD and Adam.')
        print('Begin to load data.')
        data_train = open(args.train_data).readlines()
        data_val = open(args.val_data).readlines()
        try:
            for epoch in range(args.epochs):
                print('-' * 84)
                print('BEGIN EPOCH ' + str(epoch))
                print('-' * 84)
                model = train(model, data_train, dictionary, criterion, optimizer, device, args)
                evaluate_start_time = time.time()
                val_loss, acc = evaluate(model, data_val, dictionary, criterion, device, args)
                print('-' * 84)
                fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
                print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
                print('-' * 84)
                # Save the model, if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    save(model, args.save)
                    best_val_loss = val_loss
                else:  # if loss doesn't go down, divide the learning rate by 5.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.2
                if not best_acc or acc > best_acc:
                    save(model, args.save[:-3]+'.best_acc.pt')
                    best_acc = acc
                save(model, args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch))

            print('-' * 84)
        except KeyboardInterrupt:
            print('-' * 84)
            print('Exit from training early.')
            data_val = open(args.test_data).readlines()
            evaluate_start_time = time.time()
            test_loss, acc = evaluate(model, data_val, dictionary, criterion, device, args)
            print('-' * 84)
            fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
            print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
            print('-' * 84)
            exit(0)
    else:
        model = torch.load(args.test_model)
        model = model.to(device)

    if args.eval_on_test:
        data_val = open(args.test_data).readlines()
        evaluate_start_time = time.time()
        test_loss, acc = evaluate(model, data_val, dictionary, criterion, device, args)
        print('-' * 84)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
        print('-' * 84)
    exit(0)
