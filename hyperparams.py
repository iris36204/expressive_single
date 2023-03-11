# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/expressive_tacotron
'''
class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.

    vocab = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz''' # ␀: Padding ␃: End of Sentence

    # data
    #data = "../expressive_tacotron/data"
    data = "data"
    test_data = 'test_sents.txt'
    ref_audio = 'ref1/*.wav'
    target_speaker = 'target_speaker/16_am_m.wav'
    #test_data = '../expressive_tacotron/test_sents.txt'
    #ref_audio = '../expressive_tacotron/ref1/*.wav'
    Tx = 188 # Fixed length of text length.

    # signal processing
    sr = 22050 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor.
    dropout_rate = .5

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir"
    sampledir = 'samples'
    batch_size = 32
    num_iterations = 50000
