import json
import argparse


class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-bottleneck', action='store_true',
                        help='whether to use bottleneck model')
    parser.add_argument('--pooling', type=str, default='all',
                        help='pooling strategy; choices: [all, mean, max]')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers in BiLSTM')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--reserved', type=int, default=0,
                        help='number of representation heads to reserve for boosting')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--ncat', type=int, default=8,
                        help='number of categories for each random variable in the sparse intermediate representation')
    parser.add_argument('--sparsity', type=str, default='softmax',
                        help='strategy for inducing sparsity of intermediate representation; softmax, L1, entropy, or similarity')
    parser.add_argument('--penalty', type=str, default='overlap',
                        help='attention penalty; options: overlap, uncover')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--stage2', type=int, default=0,
                        help='number of epochs to run in boosting stage')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='',
                        help='path to save the final model')
    parser.add_argument('--dictionary', type=str, default='',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--shuffle', action='store_true',
                        help='re-shuffle training data at every epoch')
    parser.add_argument('--class-number', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=1, 
                        help='the attention orthogonality penalization coefficient')
    parser.add_argument('--penalization-growth', type=float, default=None, 
                        help='the attention orthogonality penalization growth rate')
    parser.add_argument('--sparsity-coeff', type=float, default=1, 
                        help='the intermediate rep sparsity penalization coefficient')
    parser.add_argument('--sparsity-growth', type=float, default=None, 
                        help='intermediate rep sparsity penalization growth rate')
    parser.add_argument('--eval-on-test', action='store_true',
                        help='evaluate test set after training')
    parser.add_argument('--out-log', type=str, default='choices.csv',
                        help='file to store test output of the model')
    
    return parser

