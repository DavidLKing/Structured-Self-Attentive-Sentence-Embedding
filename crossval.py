from train import *
from util import Dictionary, get_base_parser

import torch
import torch.nn as nn
import torch.optim as optim

import time
import random
import os
import copy

def get_splits(all_data, fold, label_data, args):
    fold_size = len(all_data) // args.xfolds
    leftover = len(all_data) % fold_size
    fold_sizes = [fold_size] * args.xfolds
    leftovers = ([1] * leftover) + ([0] * (args.xfolds - leftover))
    fold_sizes = [a+b for a,b in zip(fold_sizes, leftovers)]
    # range of test indices
    first = sum(fold_sizes[:fold])
    last = first + fold_sizes[fold]
    # range of dev indices; ensure that every fold has a different dev set
    dev_first = last % len(all_data)
    dev_last = dev_first + fold_sizes[(fold+1) % args.xfolds]
    data_test = all_data[first:last]
    data_val = all_data[dev_first:dev_last]
    if dev_first > first:
        data_train = all_data[:first] + all_data[dev_last:]
    else: #should only happen on last fold
        data_train = all_data[dev_last:first]
    data_train += label_data
    return data_train, data_val, data_test
    

if __name__ == "__main__":
    # parse the arguments
    parser = get_base_parser()
    parser.add_argument('--data', type=str, default='',
                        help='location of the cross-validation data, should be a json file')
    parser.add_argument('--label-data', type=str, default='',
                        help='location of the label map (int -> sentence) in json format')
    parser.add_argument('--xfolds', type=int, default=10, help='number of cross-val folds')
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
    assert os.path.exists(args.data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    n_token = len(dictionary)
    criterion = nn.CrossEntropyLoss()
    
    print(args)
#    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
#    for i in range(args.batch_size):
#        for j in range(args.attention_hops):
#            I.data[i][j][j] = 1
#    I = I.to(device)

    print('Begin to load data.')
    all_data = open(args.data).readlines()
    label_data = open(args.label_data).readlines()
    fold_dev_losses = []
    fold_dev_accs = []
    fold_test_accs = []
    fold_test_losses = []
    for fold in range(args.xfolds):
        print('-' * 84)
        print('BEGIN FOLD ' + str(fold))
        print('-' * 84)
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
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise Exception('For other optimizers, please add it yourself. '
                            'supported ones are: SGD and Adam.')
        best_val_loss = None
        best_acc = None
        best_model = None
        #get the right splits
        data_train, data_val, data_test = get_splits(all_data, fold, label_data, args)
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
                best_model = copy.deepcopy(model)
            save(model, args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch))

        print('-' * 84)
        fold_dev_losses += [best_val_loss]
        fold_dev_accs += [best_acc]
        
        if args.eval_on_test:
            evaluate_start_time = time.time()
            test_loss, acc = evaluate(best_model, data_test, dictionary, criterion, device, args)
            fold_test_losses += [test_loss]
            fold_test_accs += [acc]
            print('-' * 84)
            fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
            print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
            print('-' * 84)

    print('-' * 84)
    fmt = '| dev average | test loss (pure) {:5.4f} | Acc {:8.4f}'
    avg_dev_loss = sum(fold_dev_losses)/float(args.xfolds)
    avg_dev_acc = sum(fold_dev_accs)/float(args.xfolds)
    print(fmt.format(avg_dev_loss, avg_dev_acc))
    print('-' * 84)

    if args.eval_on_test:
        print('-' * 84)
        fmt = '| test average | test loss (pure) {:5.4f} | Acc {:8.4f}'
        avg_test_loss = sum(fold_test_losses)/float(args.xfolds)
        avg_test_acc = sum(fold_test_accs)/float(args.xfolds)
        print(fmt.format(avg_test_loss, avg_test_acc))
        print('-' * 84)

    
    exit(0)
    
