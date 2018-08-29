# df1 df2 df3 df4类型为: pandas.core.frame.DataFrame.分别引用输入桩数据
# topai(1, df1)函数把df1内容写入第一个输出桩
# embedding from :https://kexue.fm/archives/4304
import gzip
import os
import pickle
import jieba
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer, InputSpec
from keras.layers import CuDNNLSTM, subtract, Conv1D, Dense, Input, Dropout, Lambda, Embedding, Flatten, concatenate, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold

index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

try:
    from input_online import *
except ImportError:
    print('we are online!')
    online = True
# setting parameters
max_len = 20
max_len_c = 20
num_iter = 10
epochs = 15
batch_size = 128
dropout_embedding = 0.2
clear_cache = False
use_embedding_cache = False
structure_shuffle = True
char_word_merge = True
if clear_cache:
    if os.path.exists(model_dir + 'tok.pkl'):
        os.remove(model_dir + 'tok.pkl')
    if os.path.exists(model_dir + 'tok_c.pkl'):
        os.remove(model_dir + 'tok_c.pkl')
    if os.path.exists(model_dir + 'embedding_matrix.npy'):
        os.remove(model_dir + 'embedding_matrix.npy')
    if os.path.exists(model_dir + 'embedding_matrix_c.npy'):
        os.remove(model_dir + 'embedding_matrix_c.npy')


def load(filename):
    """Loads a compressed object from disk
    """
    print('loading from %s' % filename)
    if filename.endswith('.npy'):
        obj = np.load(filename)
    else:
        with gzip.open(filename, 'rb') as f:
            obj = pickle.load(f)
    return obj


def safe_load(path):
    path = model_dir + path
    try:
        if os.path.exists(path):
            return load(path)
    except Exception as e:
        print(e)
    return None


def save(obj, filename, save_npz=False):
    """Saves a compressed object to disk
    """
    if type(obj) is np.ndarray and save_npz:
        np.save(filename, obj)
    else:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(obj, file=f, protocol=0)


def safe_save(data, path, override=True, save_npz=True):
    path = model_dir + path
    if not os.path.exists(path) or override:
        save(data, path, save_npz=save_npz)


def seg_text_by_char(text_list, use_space=False):
    text_seged = []
    for text in text_list:
        if hasattr(text, 'decode'):
            text = text.decode('utf-8')

        char_list = []
        for c in text:
            char_list.append(c)
        if use_space:
            text_seged.append(' '.join(char_list))
        else:
            text_seged.append(char_list)
    return text_seged


def seg_text(text_list, use_space=False):
    """
    segment input sentence using jieba
    :param text_list: iterable of raw chinese sentences
    :param use_space:  if True, return chinese words separated by
    :return:
    """
    text_seged = []
    for text in text_list:
        if not use_space:
            text_seged.append(list(jieba.cut(text, cut_all=False)))
        else:
            text_seged.append(' '.join(jieba.cut(text, cut_all=False)))
    return text_seged


# custom keras models
class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=2, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)

    def get_config(self):
        config = {'k': self.k}
        base_config = super(KMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def matching_layer(a, b):
    mul = Lambda(lambda x: -x[0] * x[1])([a, b])  # Multiply()([a, b])
    sub = Lambda(lambda x: K.abs(x))(subtract([a, b]))
    return concatenate([mul, sub])


# split sentence
jieba.load_userdict(model_dir + 'dict.txt')
text_x1_s = seg_text(df1['sent1'], use_space=True)
text_x2_s = seg_text(df1['sent2'], use_space=True)
label = df1['label'].values
# fit tokenizer
tok = safe_load('tok.pkl')
sent_total = text_x1_s + text_x2_s
if tok is None:
    tok = Tokenizer(num_words=None, lower=False)
    tok.fit_on_texts(sent_total)
    safe_save(tok, 'tok.pkl')
x1 = tok.texts_to_sequences(text_x1_s)
x2 = tok.texts_to_sequences(text_x2_s)
x1_p = pad_sequences(x1, maxlen=max_len)
x2_p = pad_sequences(x2, maxlen=max_len)
# split sentence by char
tok_c = safe_load('tok_c.pkl')
text_x1_s = seg_text_by_char(df1['sent1'], use_space=True)
text_x2_s = seg_text_by_char(df1['sent2'], use_space=True)
sent_total = text_x1_s + text_x2_s
if tok_c is None:
    tok_c = Tokenizer(num_words=None, lower=False)
    tok_c.fit_on_texts(sent_total)
    safe_save(tok_c, 'tok_c.pkl')
x1_c = tok_c.texts_to_sequences(text_x1_s)
x2_c = tok_c.texts_to_sequences(text_x2_s)
x1_p_c = pad_sequences(x1_c, maxlen=max_len_c)
x2_p_c = pad_sequences(x2_c, maxlen=max_len_c)
safe_save(x1_p_c, 'x1.npy', override=False)
safe_save(x2_p_c, 'x2.npy', override=False)
safe_save(x1_p, 'x1_word.npy', override=False)
safe_save(x2_p, 'x2_word.npy', override=False)
safe_save(label, 'label.pkl', save_npz=False, override=False)
sk = KFold(n_splits=num_iter, shuffle=True, random_state=1)
use_char_embedding = True
df = df2
for e, (tr_index, te_index) in enumerate(sk.split(x1_p_c, label.ravel(), )):
    print('iter %d' % e)
    if char_word_merge:
        use_char_embedding = (e % 5) % 2 == 0
        print('use char embedding ' + str(use_char_embedding))
    # read embedding  char
    if use_char_embedding:
        word_index = tok_c.word_index
    else:
        word_index = tok.word_index
    words = df[df.columns[0]]
    embedding = df.drop([df.columns[0]], axis=1).astype('float').as_matrix()
    if use_embedding_cache:
        embedding_matrix_c = safe_load('embedding_matrix_c%d.npy' % i)
    else:
        embedding_matrix_c = None
    if embedding_matrix_c is None:
        num_words = len(word_index)
        key = dict()
        for i, word in words.iteritems():
            key[word] = i
        embedding_size = embedding.shape[1]
        embedding_matrix_c = np.zeros((num_words, embedding_size))
        oov = 0
        oov_char = 0
        for word, i in word_index.items():
            if i >= num_words:
                continue
            if word in key:
                embedding_vector = embedding[key[word], :]
                embedding_matrix_c[i] = embedding_vector
            else:
                if use_char_embedding:
                    oov += 1
                    embedding_matrix_c[i] = np.random.normal(size=(1, embedding_size))
                else:
                    hit = 0
                    for c in word:
                        if c in key:
                            embedding_vector = embedding[key[c], :]
                            embedding_matrix_c[i] += embedding_vector
                            hit += 1
                    if hit == 0:
                        oov += 1
                    else:
                        oov_char += 1
                        embedding_matrix_c[i] = embedding_matrix_c[i] / hit
        print('oov %d/%d/%d' % (oov, oov_char, num_words))
        if use_char_embedding:
            safe_save(embedding_matrix_c, 'embedding_matrix_c%d.npy' % e)
        else:
            safe_save(embedding_matrix_c, 'embedding_matrix%d.npy' % e)

    if use_char_embedding:
        input_shape_c = (max_len_c,)
    else:
        input_shape_c = (max_len,)
    if online and e not in index:
        continue
    if not use_char_embedding:
        x1_tr, x2_tr, y_tr = x1_p[tr_index], x2_p[tr_index], label[tr_index]
        x1_te, x2_te, y_te = x1_p[te_index], x2_p[te_index], label[te_index]
    else:
        x1_tr, x2_tr, y_tr = x1_p_c[tr_index], x2_p_c[tr_index], label[tr_index]
        x1_te, x2_te, y_te = x1_p_c[te_index], x2_p_c[te_index], label[te_index]
    input1c = Input(shape=input_shape_c, )
    input2c = Input(shape=input_shape_c, )
    embed1c = Embedding(embedding_matrix_c.shape[0], embedding_matrix_c.shape[1], weights=[embedding_matrix_c],
                        trainable=False, input_shape=input_shape_c)
    dropout_layer = SpatialDropout1D(dropout_embedding)
    lstm0 = Bidirectional(CuDNNLSTM(128, return_sequences=True))
    v1c = dropout_layer((embed1c(input1c)))
    v2c = dropout_layer((embed1c(input2c)))

    v1h = (lstm0(v1c))
    v2h = (lstm0(v2c))
    if e % 3 != 0 or not structure_shuffle:
        lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))
        v1hh = lstm1(v1h)
        v2hh = lstm1(v2h)
    else:
        conv = Conv1D(128, kernel_size=2, padding='same', kernel_initializer='he_uniform', activation='relu')
        conv2 = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_uniform', activation='relu')
        v1hh = concatenate([conv(v1h), conv2(v1h)])  # matt(v1h, v1h, v1h)
        v2hh = concatenate([conv(v2h), conv2(v2h)])
    if e <= num_iter / 2 or not structure_shuffle:
        mp = GlobalMaxPooling1D()
        ap = GlobalAveragePooling1D()
        v1, v2 = concatenate([mp(v1hh), ap(v1hh)]), concatenate([mp(v2hh), ap(v1hh)])
    else:
        kp = KMaxPooling(k=2)
        v1, v2 = kp(v1hh), kp(v2hh)
    aggregated = matching_layer(v1, v2)
    feature = aggregated
    feature = Dropout(0.5)(feature)
    feature = Dense(256, activation='relu')(feature)
    feature = Dropout(0.2)(feature)
    feature = Dense(256, activation='relu')(feature)
    res_c = Dense(1, activation='sigmoid')(feature)
    model_c = Model(inputs=[input1c, input2c], outputs=res_c)
    model_c.compile(optimizer=Adam(), loss="binary_crossentropy")
    if not online:
        model_c.summary()
    model_c.fit(x=[x1_tr, x2_tr], y=y_tr, batch_size=batch_size, epochs=epochs)
    model_c.save(model_dir + 'model_final_try_%d.h5' % e)
