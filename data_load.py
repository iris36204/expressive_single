






from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *
import codecs
import re
import os
import unicodedata

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = re.sub(u"[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode in ("train", "eval"):
        # Parse
        fpaths, texts = [], []
        transcript = os.path.join(hp.data, 'metadata.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        total_hours = 0
        if mode=="train":
            lines = lines[1:]
        else: # We attack only one sample!
            lines = lines[:1]
        
        for line in lines:
            fname, _, text = line.strip().split("|")
            fpath = os.path.join(hp.data, "wavs", fname + ".wav")

            fpaths.append(fpath)

            text = text_normalize(text) + u"␃"  # ␃: EOS
            text = [char2idx[char] for char in text]
            texts.append(np.array(text, np.int32).tostring())
        return fpaths, texts
    else:
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + u"␃" for line in lines]  # text normalization, E: EOS
        texts = np.zeros((len(lines), hp.Tx), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, texts = load_data() # list

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths)
        texts = tf.convert_to_tensor(texts)

        # Create Queues
        fpath, text = tf.compat.v1.train.slice_input_producer([fpaths, texts], shuffle=True)

        #fpath, text = tf.data.Dataset.from_tensor_slices(tuple([fpaths, texts])).shuffle(num_batch)
        # Text parsing
        text = tf.compat.v1.decode_raw(text, tf.int32)  # (None,)

        # Padding
        text = tf.pad(text, ([0, hp.Tx], ))[:hp.Tx] # (Tx,)

        '''if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                mag = "mags/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:'''
        fname, mel, mag = tf.numpy_function(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        #fname, mel, mag = load_spectrograms(fpath[0])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((hp.Tx,))
        mel.set_shape((None, hp.n_mels*hp.r))
        mag.set_shape((None, hp.n_fft//2+1))
        
        # Batching
        texts, mels, mags, fnames = tf.compat.v1.train.batch([text, mel, mag, fname],
                                                   num_threads=8,
                                                   batch_size=hp.batch_size,
                                                   capacity=hp.batch_size * 64,
                                                   allow_smaller_final_batch=False,
                                                   dynamic_pad=True)
        
    return texts, mels, mags, fnames, num_batch