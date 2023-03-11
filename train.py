






from __future__ import print_function

import sys
import os
from hyperparams import Hyperparams as hp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
from data_load import get_batch, load_vocab
from modules import *
from networks import transcript_encoder, reference_encoder, decoder1, decoder2
from utils import *
#from torch.utils.tensorboard import SummaryWriter
#from playsound import playsound
class Graph:
    def __init__(self, mode="train"):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set phase
        is_training=True if mode=="train" else False

        # Graph
        # Data Feeding
        # x: Text. int32. (N, Tx) or (32, 188)
        # y: Reduced melspectrogram. float32. (N, Ty//r, n_mels*r) or (32, ?, 400)
        # z: Magnitude. (N, Ty, n_fft//2+1) or (32, ?, 1025)
        # ref: Melspectrogram of Reference audio. float32. (N, Ty, n_mels) or (32, ?, 80)
        if mode=="train":
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
            self.ref = tf.reshape(self.y, (hp.batch_size, -1, hp.n_mels))#(32, ?, 80)
        elif mode=="eval":
            self.x = tf.compat.v1.placeholder(tf.int32, shape=(None, None))
            self.y = tf.compat.v1.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
            self.z = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
            self.fnames = tf.compat.v1.placeholder(tf.string, shape=(None,))
            self.ref = tf.compat.v1.placeholder(tf.float32, shape=(None, None, hp.n_mels))
        else: # Synthesize
            self.x = tf.compat.v1.placeholder(tf.int32, shape=(hp.batch_size, hp.Tx))
            self.y = tf.compat.v1.placeholder(tf.float32, shape=(hp.batch_size, None, hp.n_mels*hp.r))
            self.ref = tf.compat.v1.placeholder(tf.float32, shape=(hp.batch_size, None, hp.n_mels))

        # Get encoder/decoder inputs
        self.transcript_inputs = embed(self.x, len(hp.vocab), hp.embed_size) # (N, Tx, E)(32, 188, 256)
        self.reference_inputs = tf.expand_dims(self.ref, -1)#(32, ?, 80, 1)

        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)(32, ?, 400)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)(32, ?, 80)

        # Networks
        with tf.compat.v1.variable_scope("net"):
            # Encoder
            self.texts = transcript_encoder(self.transcript_inputs, is_training=is_training) # (N, Tx=188, E)

            self.prosody = reference_encoder(self.reference_inputs, is_training=is_training) # (N, 128)

            self.prosody = tf.expand_dims(self.prosody, 1) # (N, 1, 128)
            self.prosody = tf.tile(self.prosody, (1, hp.Tx, 1)) # (N, Tx=188, 128)
            self.memory = tf.concat((self.texts, self.prosody), -1) # (N, Tx, E+128)

            # Decoder1
            self.y_hat, self.alignments = decoder1(self.decoder_inputs,
                                                     self.memory,
                                                     is_training=is_training) # (N, T_y//r, n_mels*r)

    # Decoder2 or postprocessing
            self.z_hat = decoder2(self.y_hat, is_training=is_training) # (N, T_y//r, (1+n_fft//2)*r)

        # monitor
        self.audio = tf.compat.v1.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)

        if mode in ("train", "eval"):
            # Loss
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
            self.loss = self.loss1 + self.loss2

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            #self.global_step = tf.compat.v1.train.get_or_create_global_step()
            
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            '''with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.initialize_all_variables())
                print(sess.run(self.global_step))'''

            # Summary
            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss2'.format(mode), self.loss2)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)

            tf.summary.audio("{}/sample".format(mode), tf.expand_dims(self.audio, 0), hp.sr)
            self.merged = tf.compat.v1.summary.merge_all()
'''
            from datetime import datetime
            logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
            model.fit(x=x_train, 
                        y=y_train, 
                        epochs=5, 
                        validation_data=(x_test, y_test), 
                        callbacks=[tensorboard_callback])
             
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")               
            root_logdir = "save_dir3"     
            save_logdir = "{}/run-{}/".format(root_logdir, now) 

            sess = tf.compat.v1.Session()
            sess.run(self.merged)


            file_writer =tf.compat.v1.summary.FileWriter(save_logdir,tf.compat.v1.get_default_graph())
            file_writer.add_graph(sess.graph)

            file_writer.close()
            print("suc")

'''

if __name__ == '__main__':
    g = Graph(); print("Training Graph loaded")

    sv = tf.compat.v1.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
    gs = 0

    with sv.managed_session() as sess:
        if len(sys.argv) == 2:

            sv.saver.restore(sess, sys.argv[1])
            print("Model restored.")

        #Tensorboard
        #writer = tf.summary.create_file_writer("tensorboardOutput")
        writer = tf.compat.v1.summary.FileWriter("tensorboardOutput/", sess.graph)
        while 1:
            print("gs_1:")
            print(gs)
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, gs = sess.run([g.train_op, g.global_step])
                print("gs:")
                print(gs)
                print()
                print(sess.run(g.loss))
                print()
           
                # Write checkpoint files
                if gs % 1000 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_100_gs_{}k'.format(gs//1000))

                    # plot the first alignment for logging
                    al = sess.run(g.alignments)
                    plot_alignment(al[0], gs)

                #Tensorboard
                if gs % 50 == 0:
                    result = sess.run(g.merged)
                    writer.add_summary(result, gs)

            if gs > 54050:#hp.num_iterations:
                break

    print("Done")
''''''
