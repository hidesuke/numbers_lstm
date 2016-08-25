# -*- encoding:utf-8 -*-
import os
import random
import datetime
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time
import math
import six
import smtplib
from email.mime.text import MIMEText
import cPickle as pickle
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import link
from chainer import Variable

import urllib2
import xlrd
import chardet


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


# gotten from http://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614
def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore {0} because of parameter mismatch'.format(child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy {0}'.format(child.name)


def train(
    output_path,
    train_data,
    words,
    vocab,
    fine_tuning=True,
    pretrained_vocab_size=0,
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

    # Prepare model
    lm = Network(len(vocab), rnn_size, dropout_ratio=dropout, train=True)
    model = L.Classifier(lm)
    model.compute_accuracy = False  # we only want the perplexity

    # load pre-trained model
    pretrained_model_path = os.path.join(output_path, 'model.npz')
    if fine_tuning and os.path.exists(pretrained_model_path):
        lm2 = Network(pretrained_vocab_size, rnn_size, dropout_ratio=dropout, train=True)
        model2 = L.Classifier(lm2)
        model2.compute_accuracy = False
        serializers.load_npz(pretrained_model_path, model2)
        copy_model(model2, model)

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
            print 'epoch {} iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(epoch, i + 1, perp, throuput)
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
            serializers.save_npz('{}/model.npz'.format(output_path), model)
            serializers.save_npz('{}/rnnlm.state.npz'.format(output_path), optimizer)

    print '===== finish train. ====='


def predict(type, model_path, vocab_path, primetext, seed,
            length=40, unit=128, dropout=0.0, sample=1.0):

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


def prepare(
    csv_file_path,
    pretrained_vocab=None,
    type='n4',
    fine_tuning=True
):
    vocab = {}
    pretrained_vocab_size = 0
    if fine_tuning and pretrained_vocab and os.path.exists(pretrained_vocab):
        vocab = pickle.load(open(pretrained_vocab, 'rb'))
        pretrained_vocab_size = len(vocab)
    words = []
    with open(csv_file_path) as csv:
        for line in csv:
            words.append(csv_to_number(line, type))
    if type in ['n4_one_by_one', 'n3_one_by_one', 'l6']:
        temp = []
        for word in words:
            temp.extend(word)
            temp.append('<eos>')
        words = temp
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    pickle.dump(vocab, open(pretrained_vocab, 'wb'))
    return dataset, words, vocab, pretrained_vocab_size


def csv_to_number(csv_line, type):
    arr = csv_line.split(',')
    if type == 'n4':
        return '{}{}{}{}'.format(arr[0], arr[1], arr[2], arr[3])
    elif type == 'n4_one_by_one':
        return arr[0:4]
    elif type == 'n3':
        return '{}{}{}'.format(arr[0], arr[1], arr[2])
    elif type == 'n3_one_by_one':
        return arr[0:3]
    elif type == 'l6':
        return arr[0:6]


def write_to_file(primetext, out_dir, result, type):
    with open(os.path.join(out_dir, 'result.txt'), 'w') as result_txt:
        result_txt.write('--------------------\n')
        result_txt.write('type: ' + type + '\n')
        result_txt.write('primetext: ' + primetext + '\n')
        if type in ['n4_one_by_one', 'n3_one_by_one', 'l6']:
            buf = ''
            for word in result:
                if word == '.':
                    buf += '\n'
                else:
                    if type == 'l6':
                        buf += ',' + word
                    else:
                        buf += word
            result_txt.write(buf)
            print buf
        else:
            result_txt.write('\n'.join(result))
            print '\n'.join(result)


def prepare_train_predict(src, pretrained_vocab, out_dir, epoch, type, fine_tuning=True):
    vocab_file_path = os.path.join(out_dir, 'vocab2.bin')
    dataset, words, vocab, pretrained_vocab_size = prepare(src, vocab_file_path, type, fine_tuning=fine_tuning)
    index = 1
    while True:
        primetext = words[-index]
        if primetext != '<eos>':
            break
        index += 1
    train(out_dir, dataset, words, vocab, pretrained_vocab_size=pretrained_vocab_size, n_epoch=epoch, fine_tuning=fine_tuning)
    result = predict(
        type,
        os.path.join(out_dir, 'model.npz'),
        vocab_file_path,
        primetext,
        int(random.random() + 10000)
    )
    write_to_file(primetext, out_dir, result, type)


def download_data(data_path):
    xls_source = urllib2.urlopen('http://r7-yosou.hippy.jp/T-data.xls')
    with open(data_path, 'wb') as xls_file:
        xls_file.write(xls_source.read())


def make_csv_data():
    xls = xlrd.open_workbook('./data/data.xls')
    num_sheets = xls.nsheets
    for sheet_idx in range(num_sheets):
        sheet = xls.sheet_by_index(sheet_idx)
        with open(os.path.join('data', sheet.name + '.csv'), 'w') as csv:
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    try:
                        csv.write(str(int(sheet.cell(row, col).value)) + ',')
                    except:
                        pass
                csv.write('\n')


def get_numbers_latest_result():
    data_path = os.path.join('data', 'data.xls')
    if os.path.exists(data_path):
        file_timestamp = datetime.datetime.fromtimestamp(os.stat(data_path).st_mtime).strftime('%Y-%m-%d')
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if today > file_timestamp:
            download_data(data_path)
    else:
        download_data(data_path)


'''
CSVファイルは1行の長さが不定長。ただし、最初のN桁が数値。
ナンバーズ4であれば 1,2,3,4,,,のようなのが1レコード。5番目以降の数値は捨てる
'''
if __name__ == '__main__':
    # get_numbers_latest_result()
    prepare_train_predict('./data/N4.csv', './output/n4_ft/vocab2.bin', './output/n4_ft', 1000, 'n4')
    prepare_train_predict('./data/N4.csv', './output/n4_1x1_ft/vocab2.bin', './output/n4_1x1_ft', 1000, 'n4_one_by_one')
    prepare_train_predict('./data/N4.csv', './output/n4/vocab2.bin', './output/n4', 1000, 'n4', fine_tuning=False)
    prepare_train_predict('./data/N4.csv', './output/n4_1x1/vocab2.bin', './output/n4_1x1', 1000, 'n4_one_by_one', fine_tuning=False)
    prepare_train_predict('./data/L6.csv', './output/l6/vocab2.bin', './output/l6/', 2000, 'l6', fine_tuning=False)



