# -*- coding: utf-8 -*-
import cPickle as pickle
import random

import numpy as np
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import chainer


class Network(chainer.Chain):
    def __init__(self, n_vocab, n_units, dropout_ratio=0.0, train=True):
        super(Network, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.LSTM(n_units, n_units),
            l5=L.LSTM(n_units, n_units),
            l6=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

        self.train = train
        self.dropout_ratio = dropout_ratio

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, ratio=self.dropout_ratio, train=self.train))
        h2 = self.l2(F.dropout(h1, ratio=self.dropout_ratio, train=self.train))
        h3 = self.l3(F.dropout(h2, ratio=self.dropout_ratio, train=self.train))
        h4 = self.l4(F.dropout(h3, ratio=self.dropout_ratio, train=self.train))
        h5 = self.l5(F.dropout(h4, ratio=self.dropout_ratio, train=self.train))
        y = self.l6(F.dropout(h5, ratio=self.dropout_ratio, train=self.train))

        return y

    def predict(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, ratio=self.dropout_ratio, train=self.train))
        h2 = self.l2(F.dropout(h1, ratio=self.dropout_ratio, train=self.train))
        h3 = self.l3(F.dropout(h2, ratio=self.dropout_ratio, train=self.train))
        h4 = self.l4(F.dropout(h3, ratio=self.dropout_ratio, train=self.train))
        h5 = self.l5(F.dropout(h4, ratio=self.dropout_ratio, train=self.train))
        y = self.l6(F.dropout(h5, ratio=self.dropout_ratio, train=self.train))

        return F.softmax(y)

    def add_embed(self, add_counts, dimension):
        add_W = np.random.randn(add_counts, dimension).astype(np.float32)
        add_gW = np.empty((add_counts, dimension)).astype(np.float32)
        self.embed.W = np.r_[self.embed.W, add_W]
        self.embed.gW = np.r_[self.embed.gW, add_gW]


def predict(model_path, vocab_path, primetext,
            seed, unit, dropout, sample, length, use_mecab=False):

    np.random.seed(seed)

    # load vocabulary
    vocab = pickle.load(open(vocab_path, 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c
    n_units = unit

    lm = Network(len(vocab), n_units, dropout_ratio=dropout, train=False)
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity

    serializers.load_npz(model_path, model)
    model.predictor.reset_state()  # initialize state
    prev_char = np.array([0])
    ret = []
    if not isinstance(primetext, unicode):
        primetext = unicode(primetext, 'utf-8')
    ret.append(primetext)
    prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[primetext])
    prob = model.predictor.predict(prev_char)

    for i in xrange(length):
        prob = model.predictor.predict(prev_char)

        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        if ivocab[index] == "<eos>":
            ret.append(".")
        else:
            ret.append(ivocab[index])
        prev_char = Variable(np.ones((1,)).astype(np.int32) * vocab[ivocab[index]])
    return ret


def main():
    epoch = 1000
    primetext = '2008'
    seed = int(random.random() + 10000)
    trained_model = "./output/model{0:0>4}".format(epoch)
    result = predict(trained_model, './output/vocab2.bin', primetext, seed,
                     128, 0.0, 1, 40)
    print '\n'.join(result)
    return


if __name__ == "__main__":
    main()
