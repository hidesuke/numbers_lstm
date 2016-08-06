# -*- encoding:utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time
import math
import six
import cPickle as pickle
from chainer import cuda
from chainer import optimizers
from chainer import serializers


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


def train(
    train_data,
    words,
    vocab,
    gpu=-1,
    n_epoch=100,
    rnn_size=128,
    learning_rate=2e-3,
    learning_rate_decay=0.97,
    learning_rate_decay_after=10,
    decay_rate=0.95,
    dropout=0.0,
    bprop_len=50,
    batchsize=50,  # minibatch size
    grad_clip=5    # gradient norm threshold to clip
):

    xp = cuda.cupy if gpu >= 0 else np

    pickle.dump(vocab, open('./output/vocab2.bin', 'wb'))

    # Prepare model
    # TODO: train=False??? True???
    lm = Network(len(vocab), rnn_size, dropout_ratio=dropout, train=True)
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity

    # Setup optimizer
    optimizer = optimizers.RMSprop(lr=learning_rate, alpha=decay_rate, eps=1e-8)
    optimizer.setup(model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # Learning loop
    whole_len = train_data.shape[0]
    jump = whole_len // batchsize
    cur_log_perp = xp.zeros(())
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    accum_loss = 0
    batch_idxs = list(range(batchsize))
    print 'Goint to train {} iterations'.format(jump * n_epoch)

    for i in six.moves.range(jump * n_epoch):
        x = chainer.Variable(
            xp.asarray([train_data[(jump * j + i) % whole_len] for j in batch_idxs]))
        t = chainer.Variable(
            xp.asarray([train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]))
        loss_i = model(x, t)
        accum_loss += loss_i

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = 0
            optimizer.update()

        if (i + 1) % 500 == 0:
            now = time.time()
            throuput = 10000. / (now - cur_at)
            perp = math.exp(float(cur_log_perp) / 10000)
            print 'iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(i + 1, perp, throuput)
            cur_at = now
            cur_log_perp.fill(0)

        if (i + 1) % jump == 0:
            epoch += 1
            now = time.time()
            cur_at += time.time() - now  # skip time of evaluation

            if epoch >= 6:
                optimizer.lr /= 1.2
                print 'learning rate = {:.10f}'.format(optimizer.lr)
            # Save the model and the optimizer
            serializers.save_npz('./output/model%04d' % epoch, model)
            print '--- epoch: {} ------------------------'.format(epoch)
            serializers.save_npz('./output/rnnlm.state', optimizer)

    print '===== finish train. ====='

# prepare =========================================


def prepare(
    csv_file_path,
    pretrained_vocab_path=None,
):
    vocab = {}
    words = []
    with open(csv_file_path) as csv:
        for line in csv:
            words.append(csv_to_number(line))
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset, words, vocab


def csv_to_number(csv_line):
    arr = csv_line.split(',')
    return '{}{}{}{}'.format(arr[0], arr[1], arr[2], arr[3])


def main():
    csv_path = './data/N4.csv'
    dataset, words, vocab = prepare(csv_path)
    train(dataset, words, vocab, n_epoch=1000)
    return


if __name__ == '__main__':
    main()
