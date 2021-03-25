from torchtext import data
import os
import pdb
import random
import math
import numpy as np
import re
import torch


class VP(data.Dataset):
    """modeled after Shawn1993 github user's Pytorch implementation of Kim2014 - cnn for text categorization"""

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, bound_field=None, path=None, filename=None, examples=None, 
                 idxs=None, alt_dict=None, prob_dict=None, alt_p=0.0, random_state=None, **kwargs):
        """Create a virtual patient (VP) dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the directory containing the data file.
            filename: The name of the data file.
            examples: The examples contain all the data.
            idxs: list of indices of examples, in order, used to match alternatives
            alt_file: Optional file containing alternative forms of each example
            alt_p: probability of choosing an alternative, if provided
            random_state: seed for rng; random if not provided
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        #no preprocessing needed 
        self.two_ch = True if bound_field is not None else False

        fields = [('text', text_field), ('label', label_field)]
        if self.two_ch:
            fields.insert(1, ('bounds', bound_field))

        if examples is None:
            path = self.dirname if path is None else path
            filename = pkg_resources.resource_filename('vpcnn', 'data/wilkins_corrected.shuffled.51.txt') \
                if filename is None else filename
            examples = []
            with open(os.path.join(path, filename)) as f:
                lines = f.readlines()
                #pdb.set_trace()
                for line in lines:
                    try:
                        label, text = line.split("\t")
                        if self.two_ch:
                            text, bounds = split_bounds(text)
                    except:
                        print(line)
                    ex = [text, label]
                    if self.two_ch:
                        ex.insert(1, bounds)
                    this_example = data.Example.fromlist(ex, fields)
                    examples += [this_example]
            #assume "target \t source", one instance per line
        # print(examples[0].text)
        
        self.alt_p = alt_p
        self.alt_dict = alt_dict
        self.prob_dict = prob_dict
        self.rng = random.Random(random_state)    

        if alt_dict is not None:
            assert(prob_dict is not None)
            assert(idxs is not None)
        self.idxs = []
        if idxs is not None:
            self.idxs = idxs
        
        super(VP, self).__init__(examples, fields, **kwargs)
        
    def __getitem__(self, index):
        # rng
        gold = super().__getitem__(index)
        if self.alt_dict is not None and self.rng.random() < self.alt_p:
            # lookup dial/turn key for this index
            key = self.idxs[index]
            if key not in self.alt_dict:
                return gold
            else:
                # sample with np.random.choice
                alt = self.rng.choices(self.alt_dict[key], weights=self.prob_dict[key])[0]
                if self.two_ch:
                    alt, bounds = split_bounds(alt)
                ex = [alt, gold.label]
                if self.two_ch:
                    ex.insert(1, bounds)
                alt_ex = data.Example.fromlist(ex, self.fields.items())
                return alt_ex
        else:
            return gold


    @classmethod
    def splits(cls, text_field, label_field, bound_field=None, numfolds=10, foldid=None, dev_ratio=.1, shuffle=False, root='.',
               filename=None, test_filename=None, label_filename=None, train_idxs=None, alt_file=None, alt_p=0.0, num_experts=0, **kwargs):
        
        """Create dataset objects for splits of the VP dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        # alt_file is a tsv with four fields:
        # dialog idx, turn idx, content string, count
        # each entry constitutes an errorful speech transcript alternative for 
        # the corresponding example indexed by the same dialog/turn idx.
        # count is an integer count corresponding to the likelihood of the alt.
        # if alt_file is not None, an alternative will be chosen at a frequency
        # specified by alt_p, with identity of the alternative determined by the 
        # count value provided in `alts`.

        alt_dict = None
        prob_dict = None
        #denom_dict = {}

        if alt_file is not None:
            alt_dict = {}
            prob_dict = {}
            # read file
            with open(os.path.join(root, alt_file), 'r') as f:
                for line in f:
                    # store alts and accumulate freqs
                    dial, turn, content, count = line.strip().split('\t')
                    count = float(count)
                    key = (int(dial), int(turn))
                    if key not in alt_dict:
                        alt_dict[key] = []
                        prob_dict[key] = []
                        #denom_dict[key] = 0.0
                    alt_dict[key].append(content)
                    prob_dict[key].append(count)
            #        denom_dict[key] += count
            # renorm freqs to probs, store as numpy array
            #for key in prob_dict:
            #    prob_dict[key] = np.divide(prob_dict[key], denom_dict[key])
 

        #path = cls.download_or_unzip(root)
        #examples = cls(text_field, label_field, path=path, **kwargs).examples
        examples = cls(text_field, label_field, bound_field, path=root, filename=filename, **kwargs).examples
        if test_filename is not None:
            test_examples = cls(text_field, label_field, bound_field, path=root, filename=test_filename, **kwargs).examples
        if shuffle: random.shuffle(examples)
        fields = [('text', text_field), ('label', label_field)]
        if bound_field is not None:
            fields.insert(1, ('bounds', bound_field))
        label_examples = []
        label_filename = os.path.join(root, label_filename) if label_filename is not None else 'data/labels.txt'
        with open(label_filename) as f:
            lines = f.readlines()
            # pdb.set_trace()
            for line in lines:
                label, text = line.split("\t")
                if bound_field is not None:
                    text, bounds = split_bounds(text)
                ex = [text, label]
                if bound_field is not None:
                    ex.insert(1, bounds)
                this_example = data.Example.fromlist(ex, fields)
                label_examples += [this_example]
        label_idxs = [(999,999) for ex in label_examples]
        if train_idxs is None:
            traindev_idxs = [(999,999) for ex in examples]
        else:
            traindev_idxs = []
        if foldid==None:
            if num_experts > 0:
                assert num_experts <= 5
                dev_length = math.floor(len(examples) * dev_ratio)
                if test_filename is not None:
                    test = test_examples
                    traindev = examples
                    traindev_idxs = train_idxs
                else:
                    test = examples[:dev_length]
                    traindev = examples[dev_length:]
                    traindev_idxs = train_idxs[dev_length:]
                trains = []
                devs = []
                # print(dev_length)
                for i in range(num_experts):
                    devs.append(cls(text_field, label_field, bound_field=bound_field, examples=traindev[dev_length*i:dev_length*(i+1)]))
                    trains.append(cls(text_field, label_field, bound_field=bound_field, 
                                      examples=traindev[:dev_length*i]+traindev[dev_length*(i+1):]+label_examples,
                                      idxs=traindev_idxs[:dev_length*i]+traindev_idxs[dev_length*(i+1):]+label_idxs, 
                                      alt_dict=alt_dict, prob_dict=prob_dict, alt_p=alt_p, random_state=42))
                return (trains, devs, cls(text_field, label_field, bound_field=bound_field, examples=test))

            else:
                dev_index = -1 * int(dev_ratio*len(examples))
                if test_filename is not None:
                    return (cls(text_field, label_field, bound_field=bound_field, 
                                examples=examples[:dev_index]+label_examples,
                                idxs=train_idxs[:dev_index]+label_idxs, 
                                alt_dict=alt_dict, prob_dict=prob_dict, alt_p=alt_p, random_state=42),
                            cls(text_field, label_field, bound_field=bound_field, examples=examples[dev_index:]),
                            cls(text_field, label_field, bound_field=bound_field, examples=test_examples))
                else:
                    return (cls(text_field, label_field, bound_field=bound_field, 
                                examples=examples[:dev_index]+label_examples,
                                idxs=train_idxs[:dev_index]+label_idxs, 
                                alt_dict=alt_dict, prob_dict=prob_dict, alt_p=alt_p, random_state=42),
                            cls(text_field, label_field, bound_field=bound_field, examples=examples[dev_index:]))
        else:
            # assuming we don't want to use cross-validation if we have a fixed pre-defined test set
            # (if we change our minds, this needs some updating)

            #get all folds
            fold_size = math.ceil(len(examples)/numfolds)
            folds = []
            fold_idxs = []
            for fold in range(numfolds):
                startidx = fold*fold_size
                endidx = startidx+fold_size if startidx+fold_size < len(examples) else len(examples)
                folds += [examples[startidx:endidx]]
                fold_idxs += [train_idxs[startidx:endidx]]

            #take all folds except foldid as training/dev
            traindev = [fold for idx, fold in enumerate(folds) if idx != foldid]
            traindev = [item for sublist in traindev for item in sublist]
            traindev_idxs = [fold for idx, fold in enumerate(fold_idxs) if idx != foldid]
            traindev_idxs = [item for sublist in traindev_idxs for item in sublist]
            dev_index = -1 * int(dev_ratio*len(traindev))

            #test will be entire held out section (foldid)
            test = folds[foldid]
            # print(len(traindev[:dev_index]), 'num_experts', num_experts)
            if num_experts > 0:
                assert num_experts <= 5
                trains = []
                devs = []
                dev_length = math.floor(len(traindev) * dev_ratio)
                # print(dev_length)
                for i in range(num_experts):
                    devs.append(cls(text_field, label_field, bound_field=bound_field, examples=traindev[dev_length*i:dev_length*(i+1)]))
                    trains.append(cls(text_field, label_field, bound_field=bound_field, 
                                      examples=traindev[:dev_length*i]+traindev[dev_length*(i+1):]+label_examples,
                                      idxs=traindev_idxs[:dev_length*i]+traindev_idxs[dev_length*(i+1):]+label_idxs, 
                                      alt_dict=alt_dict, prob_dict=prob_dict, alt_p=alt_p, random_state=42))
                return (trains, devs, cls(text_field, label_field, bound_field=bound_field, examples=test))

            else:
                return (cls(text_field, label_field, bound_field=bound_field, 
                            examples=traindev[:dev_index]+label_examples,
                            idxs=traindev_idxs[:dev_index]+label_idxs,
                            alt_dict=alt_dict, prob_dict=prob_dict, alt_p=alt_p, random_state=42),
                        cls(text_field, label_field, bound_field=bound_field, examples=traindev[dev_index:]),
                        cls(text_field, label_field, bound_field=bound_field, examples=test))

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub("\'s", " \'s", string)
  string = re.sub("\'m", " \'m", string)
  string = re.sub("\'ve", " \'ve", string)
  string = re.sub("n\'t", " n\'t", string)
  string = re.sub("\'re", " \'re", string)
  string = re.sub("\'d", " \'d", string)
  string = re.sub("\'ll", " \'ll", string)
  string = re.sub(",", " , ", string)
  string = re.sub("!", " ! ", string)
  string = re.sub("\(", " ( ", string)
  string = re.sub("\)", " ) ", string)
  string = re.sub("\?", " ? ", string)
  string = re.sub("\s{2,}", " ", string)
  return pad2(string.strip().lower().split(" "))

def pad2(x):
    x = ['<pad>', '<pad>', '<pad>', '<pad>'] + x
    return x

def split_bounds(string):
    symbols = []
    bounds = []
    toks = string.split(" ")
    for tok in toks:
        sym, bd = tok.split("|")
        symbols += [sym]
        bounds += [bd]
    text = " ".join(symbols)
    bounds = " ".join(bounds)
    return text, bounds
