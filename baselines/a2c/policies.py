import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class AutoencoderPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, enc_size=32):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            input = tf.cast(X, tf.float32)/255.

            # TODO: Same as CNN?
            enc1conv = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[3, 3], padding="SAME", use_bias = True, activation=tf.nn.leaky_relu)
            enc2pool = tf.layers.max_pooling2d(inputs=enc1conv, pool_size=[3, 3], strides=3)
            enc2conv = tf.layers.conv2d(inputs=enc2pool, filters=16, kernel_size=[3, 3], padding="SAME", use_bias = True, activation=tf.nn.leaky_relu)
            enc3pool = tf.layers.max_pooling2d(inputs=enc2conv, pool_size=[3, 3], strides=3)
            enc3conv = tf.layers.conv2d(inputs=enc3pool, filters=16, kernel_size=[3, 3], padding="SAME", use_bias = True, activation=tf.nn.leaky_relu)
            enc_pool = tf.layers.max_pooling2d(inputs=enc3conv, pool_size=[3, 3], strides=3, name="enc_pool")

            enc_fc = conv_to_fc(enc_pool)

            # Policy
            h = fc(enc_fc, 'fc1', nh=enc_size, init_scale=np.sqrt(2))
            pi = fc(h, 'pi', nact, act=lambda x:x)
            vf = fc(h, 'v', 1, act=lambda x:x)

            # Decoder
            dec1conv = tf.layers.conv2d(enc_pool, filters=16, kernel_size=(3, 3), strides=(1, 1), name='dec1conv', padding='SAME', use_bias=True, activation=tf.nn.leaky_relu)
            dec2up = tf.layers.conv2d_transpose(dec1conv, filters=16, kernel_size=3, padding='same', strides=3, name='dec2up')
            dec3conv = tf.layers.conv2d(dec2up, filters=16, kernel_size=(3, 3), strides=(1, 1), name='dec3conv', padding='SAME', use_bias=True, activation=tf.nn.leaky_relu)
            dec4up = tf.layers.conv2d_transpose(dec3conv, filters=16, kernel_size=3, padding='same', strides=2, name='dec4up')
            dec5conv = tf.layers.conv2d(dec4up, filters=16, kernel_size=(3, 3), strides=(1, 1), name='dec5conv', padding='SAME', use_bias=True, activation=tf.nn.leaky_relu)
            dec6up = tf.layers.conv2d_transpose(dec5conv, filters=16, kernel_size=3, padding='same', strides=2, name='dec6up')

            decoded = tf.layers.conv2d(dec6up, filters=4, kernel_size=(3, 3), strides=(1, 1), name='decoded',
                                        padding='SAME', use_bias=True, activation=tf.nn.leaky_relu)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful
        dec = decoded
        enc = enc_fc
        orig = input

        def step(ob, *_args, **_kwargs):
            a, v, d, e = sess.run([a0, v0, dec, enc], {X:ob})
            return a, v, d, e, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.decoded = decoded
        self.orig = orig
