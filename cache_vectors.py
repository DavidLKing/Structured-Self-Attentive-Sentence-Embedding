from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors
import torch
import json
import argparse

parser = argparse.ArgumentParser(description='Pre-caches word vectors in a file')
parser.add_argument('-i', '--input', nargs='+', type=str, default=None, required=True, help="JSON file(s) each containing asingle list of vocabulary words")
parser.add_argument('-o', '--output', type=str, default="data/glove", help="directory in which to place cached vectors")
parser.add_argument('-v', '--vectors', type=str, default=None, required=True, help="relative path of pretrained vector file (e.g., glove.42B.300d.txt)")


args = parser.parse_args()                    
cache = args.output                    
glovepath = args.vectors
infiles = args.input

def unk(t):
    return torch.zeros_like(t).uniform_(-0.25,0.25)

words = []

for fn in infiles:
    new_words = json.loads(open(fn, 'r').readline())
    words += new_words

words = list(set(words))
vecs = _PretrainedWordVectors(glovepath, cache=cache, unk_init=unk,
                              is_include=lambda x: x in words)

