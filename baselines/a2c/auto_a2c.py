import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger
from matplotlib import pyplot as plt

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, dec_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))

        dec_loss = tf.reduce_mean(mse(train_model.decoded, train_model.orig))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef + dec_loss*dec_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, decoded):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, decoded_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, dec_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, decoded_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            folder = "models/"
            make_path(folder)
            joblib.dump(ps, folder + save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99, render=False):
        self.render = render
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_decoded = [],[],[],[],[],[]
        mb_states = self.states
        if self.render:
            plt.ion()  ## Note this correction
            fig = plt.figure(figsize=(5,4))
            axes = []
            axes.append(fig.add_subplot(3,4,1))
            axes.append(fig.add_subplot(3,4,2))
            axes.append(fig.add_subplot(3,4,3))
            axes.append(fig.add_subplot(3,4,4))

            axes.append(fig.add_subplot(3,1,2))

            axes.append(fig.add_subplot(3,4,9))
            axes.append(fig.add_subplot(3,4,10))
            axes.append(fig.add_subplot(3,4,11))
            axes.append(fig.add_subplot(3,4,12))
        for n in range(self.nsteps):
            actions, values, decoded, encoded, states = self.model.step(self.obs, self.states, self.dones)
            if self.render:
                obs = self.obs[0]
                obs = np.swapaxes(obs, 0, 2)
                obs = np.swapaxes(obs, 1, 2)
                imgs = decoded[0]
                imgs = np.swapaxes(imgs, 0, 2)
                imgs = np.swapaxes(imgs, 1, 2)
                for i in range(len(imgs)):
                    ob = obs[i]
                    axes[i].clear()
                    axes[i].set_yticklabels([])
                    axes[i].imshow(ob, cmap='gray', interpolation='nearest', aspect='equal')
                    img = imgs[i]
                    axes[i+5].clear()
                    axes[i+5].set_yticklabels([])
                    axes[i+5].imshow(img, cmap='gray', interpolation='nearest', aspect='equal')
                axes[4].clear()
                axes[4].imshow(encoded[0].reshape(8,98), interpolation='nearest', aspect='equal')
                plt.show()
                plt.pause(0.000001)  # Note this correction
            mb_obs.append(np.copy(self.obs))
            mb_decoded.append(decoded)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_decoded = np.asarray(mb_decoded, dtype=np.float32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_decoded = mb_decoded.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_decoded

def learn(policy, policy_name, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, save_interval=25000):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, decoded = runner.run()
        policy_loss, value_loss, decoded_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values, decoded)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("decoded_loss", float(decoded_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
        if update % save_interval == 0 or update == 1:
            model.save("./atari_a2c_" + policy_name + ".pkl")
    env.close()

def enjoy(policy, policy_name, env, seed, nsteps=5, nstack=4, total_timesteps=1000):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=1)
    model.load('models/atari_a2c_' + policy_name + '.pkl')
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=0, render=True)

    for update in range(1, total_timesteps):
        runner.run()


    env.close()

if __name__ == '__main__':
    main()
