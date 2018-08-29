# ***********scripts******************
# df1 df2 df3 df4类型为: pandas.core.frame.DataFrame.分别引用输入桩数据
# topai(1, df1)函数把df1内容写入第一个输出桩
import gzip
import pickle
import time
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.engine.topology import Layer, InputSpec
from keras.layers import Flatten
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# setting parameters
max_len = 20
max_len_c = 20
merge_char_word = True
semi_supervised_training = False
model_name = 'model_final_try'
try:
    from input_predict import *

    online = False
except ImportError:
    online = True
    print('we are online')

timestart = time.time()

from keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# defining useful functions
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


from  keras.callbacks import Callback


class TimeOutCallBack(Callback):
    def __init__(self, start, budget=6600):
        super(TimeOutCallBack, self).__init__()
        self.budget = budget
        self.start = start

    def on_epoch_begin(self, epoch, logs=None):
        cur = time.time()
        if (cur - self.start) >= self.budget:
            print('used time: %f' % (cur - self.start))
            self.model.stop_training = True
            print('timeout')


# load word
jieba.load_userdict(model_dir + 'dict.txt')
text_x1_s = seg_text(df1['sent1'], use_space=True)
text_x2_s = seg_text(df1['sent2'], use_space=True)
tok = safe_load('tok.pkl')
sent1 = text_x1_s
sent2 = text_x2_s
x1 = tok.texts_to_sequences(sent1)
x2 = tok.texts_to_sequences(sent2)
x1_p = pad_sequences(x1, maxlen=max_len)
x2_p = pad_sequences(x2, maxlen=max_len)
# load char
text_x1_s = seg_text_by_char(df1['sent1'], use_space=True)
text_x2_s = seg_text_by_char(df1['sent2'], use_space=True)
tok = safe_load('tok_c.pkl')
sent1 = text_x1_s
sent2 = text_x2_s
x1 = tok.texts_to_sequences(sent1)
x2 = tok.texts_to_sequences(sent2)
x1_p_c = pad_sequences(x1, maxlen=max_len_c)
x2_p_c = pad_sequences(x2, maxlen=max_len_c)
p_sum = np.zeros((x1_p_c.shape[0],), dtype=np.float32)
total = 10
weights = np.ones((total,))
predictions_array = np.zeros((total, x1_p_c.shape[0]), dtype=np.float32)
for i in range(total):
    if weights[i] != 0:
        if merge_char_word:
            use_char_embedding = (i % 5) % 2 == 0
        else:
            use_char_embedding = True

        if use_char_embedding:
            input_shape_c = (max_len_c,)
            x1 = x1_p_c
            x2 = x2_p_c
        else:
            input_shape_c = (max_len,)
            x1 = x1_p
            x2 = x2_p
        model = load_model(model_dir + model_name + '_%d.h5' % i,
                           custom_objects={
                               'KMaxPooling': KMaxPooling})
        predictions = model.predict([x1, x2], batch_size=1024).ravel()
        predictions_array[i] = predictions
        p_sum += predictions * weights[i]
p_sum /= np.sum(weights)
np.random.seed(233)
if semi_supervised_training:
    x1_tr_old1 = safe_load('x1.npy')
    x2_tr_old1 = safe_load('x2.npy')
    x1_tr_old_word1 = safe_load('x1_word.npy')
    x2_tr_old_word1 = safe_load('x2_word.npy')
    label_tr_old1 = safe_load('label.pkl')
    p_semi = np.zeros_like(p_sum)
    index_to_semi = np.random.permutation(10)
    index_to_semi = index_to_semi[:5]
    for i in range(total):
        if i not in index_to_semi:
            continue
        np.random.seed(233 + i)
        index = np.random.permutation(len(label_tr_old1))
        x1_tr_old = x1_tr_old1[index]
        x2_tr_old = x2_tr_old1[index]
        x2_tr_old_word = x2_tr_old_word1[index]
        x1_tr_old_word = x1_tr_old_word1[index]
        label_tr_old = label_tr_old1[index]
        if weights[i] != 0:
            if merge_char_word:
                use_char_embedding = (i % 5) % 2 == 0
            else:
                use_char_embedding = True
        if use_char_embedding:
            input_shape_c = (max_len_c,)
            x1 = x1_p_c
            x2 = x2_p_c
        else:
            input_shape_c = (max_len,)
            x1 = x1_p
            x2 = x2_p
        pseudo_label_index = np.bitwise_or(p_sum < 0.25, p_sum > 0.75)
        pseudo_label = p_sum[pseudo_label_index]
        x1_ps = x1[pseudo_label_index]
        x2_ps = x2[pseudo_label_index]
        if use_char_embedding:
            x1_tr = np.vstack([x1_tr_old, x1_ps, x1])
            x2_tr = np.vstack([x2_tr_old, x2_ps, np.random.permutation(x2)])
        else:
            x1_tr = np.vstack([x1_tr_old_word, x1_ps, x1])
            x2_tr = np.vstack([x2_tr_old_word, x2_ps, np.random.permutation(x2)])
        y_tr = np.concatenate([label_tr_old, pseudo_label, np.zeros_like(p_sum)])
        model = load_model(model_dir + model_name + '_%d.h5' % i,
                           custom_objects={
                               'KMaxPooling': KMaxPooling})
        # shuffle
        index = np.random.permutation(len(y_tr))
        x1_tr = x1_tr[index]
        x2_tr = x2_tr[index]
        y_tr = y_tr[index]
        model.fit([x1_tr, x2_tr], y_tr, epochs=5, batch_size=128,
                  callbacks=[TimeOutCallBack(start=timestart)])
        predictions = model.predict([x1, x2], batch_size=1024).ravel()
        predictions_array[i] = predictions

    p_semi = np.mean(predictions_array, axis=0)
    p_sum = p_semi
predictions_hard = pd.DataFrame({'id': df1['id'], 'label': np.array(p_sum > 0.4, dtype=np.int32)})
topai(1, predictions_hard)
