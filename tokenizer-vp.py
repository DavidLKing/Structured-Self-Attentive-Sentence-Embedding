from __future__ import print_function
import argparse
import json
import random
from util import Dictionary
import spacy


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenizer')
    parser.add_argument('--input', type=str, default='', help='input file')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument('--labels', type=str, default='', help='label file')
    parser.add_argument('--dict', type=str, default='', help='dictionary file')
    parser.add_argument('--label-data', action='store_true', help='to parse label file into json format')
    parser.add_argument('--shuffle', action='store_true', help='output shuffled data to file')
    args = parser.parse_args()
    tokenizer = spacy.load('en_core_web_md')
    dictionary = Dictionary()
    dictionary.add_word('<pad>')  # add padding word
    lab2int = {}
    int2lab = {}
    with open(args.labels, 'r') as labfile:
        for line in labfile:
            labint, labtext = line.strip().split('\t')
            labint = int(labint)
            lab2int[labtext] = labint
            int2lab[labint] = labtext
    with open(args.output, 'w') as fout:
        lines = open(args.input).readlines()
        if args.shuffle: random.shuffle(lines)
        for i, line in enumerate(lines):
            if not line.startswith("#STARTDIALOGUE"):
                # data: input<tab>label<tab>response<tab>interp<tab>correct<tab>...
                item = line.strip().split('\t')
                if args.label_data:
                    sent = item[1]
                    labint = int(item[0])
                else:
                    sent = item[0]
                    labint = lab2int[item[1]]
                words = tokenizer(' '.join(sent.split()))
                data = {
                    'label': labint,
                    'text': [x.text.lower() for x in words]
                }
                #map(lambda x: x.text.lower(), words)
                fout.write(json.dumps(data) + '\n')
                for item in data['text']:
                    dictionary.add_word(item)
                if i % 100 == 99:
                    print('%d/%d files done, dictionary size: %d' %
                          (i + 1, len(lines), len(dictionary)))
        fout.close()

    with open(args.dict, 'w') as fout:  # save dictionary for fast next process
        fout.write(json.dumps(dictionary.idx2word) + '\n')
        fout.close()
