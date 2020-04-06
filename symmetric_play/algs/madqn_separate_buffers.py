from functools import partial

import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from .deepq.build_graph import build_train
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.policies import DQNPolicy


class MADQN(OffPolicyRLModel):
    """
    The DQN model class.
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param exploration_initial_eps: (float) initial value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False,
                 n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None,
                 num_agents=1): # MA-MOD

        super(MADQN, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, policy_base=DQNPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        # print("POLICY TYPE", policy)
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.num_agents = num_agents

        self.graph = None
        self.sess = None
        self._train_step = [] # MA-MOD
        self.step_model = [] # MA-MOD
        self.update_target = [] # MA-MOD
        self.act = [] # MA-MOD
        self.proba_step = [] # MA-MOD
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        assert False, "MAKE SURE THIS FUNCTION ISNT CALLED"
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def setup_model(self):

        with SetVerbosity(self.verbose):
            for i in range(self.num_agents):
                assert not isinstance(self.action_space, gym.spaces.Box), \
                    "Error: DQN cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            # print(test_policy.type)
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                self.params = []

                print("AC SPC", self.action_space)
                for i in range(self.num_agents):
                    with tf.variable_scope("agent"+str(i)):
                        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                        act, _train_step, update_target, step_model = build_train(
                            q_func=partial(self.policy, **self.policy_kwargs),
                            ob_space=self.observation_space,
                            ac_space=self.action_space,
                            optimizer=optimizer,
                            gamma=self.gamma,
                            grad_norm_clipping=10,
                            param_noise=self.param_noise,
                            sess=self.sess,
                            full_tensorboard_log=False, #self.full_tensorboard_log,
                            double_q=self.double_q
                        )
                        self.act.append(act)
                        self._train_step.append(_train_step)
                        self.step_model.append(step_model)
                        self.proba_step.append(step_model.proba_step)
                        self.update_target.append(update_target)
                        self.params.extend(tf_util.get_trainable_vars("agent"+str(i) + "/deepq"))


                print(self.params)

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess) # TODO: copy this file, make two versions of the algorithm.
                for i in range(self.num_agents):
                    self.update_target[i](sess=self.sess) # TODO: Not sure, seems like the best thing to do is try using each agents own target first.

                # self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        # callback = self._init_callback(callback)

        # with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
        #         as writer:
        self._setup_learn()


        # Create the replay buffer
        if self.prioritized_replay:
            self.replay_buffer = [PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha) for _ in range(self.num_agents)] # MA-MOD
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            else:
                prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.prioritized_replay_beta0,
                                                final_p=1.0)
        else:
            self.replay_buffer = [ReplayBuffer(self.buffer_size) for _ in range(self.num_agents)] # MA-MOD
            self.beta_schedule = None

        if replay_wrapper is not None:
            assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
            self.replay_buffer = [replay_wrapper(self.replay_buffer[i]) for i in range(self.num_agents)] # MA-MOD

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                            initial_p=self.exploration_initial_eps,
                                            final_p=self.exploration_final_eps)

        episode_rewards = [[0.0]*self.num_agents] #MA-MOD
        episode_successes = []

        #callback.on_training_start(locals(), globals())
        #callback.on_rollout_start()

        reset = True
        obs = self.env.reset()

        for _ in range(total_timesteps):
            # Take action and update exploration to the newest value
            kwargs = {}
            if not self.param_noise:
                update_eps = self.exploration.value(self.num_timesteps)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = \
                    -np.log(1. - self.exploration.value(self.num_timesteps) +
                            self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            with self.sess.as_default():
                env_action = [] # MA-MOD
                for i in range(self.num_agents): # MA-MOD. This is fine for one policy.
                    action = self.act[i](np.array(obs[i])[None], update_eps=update_eps, **kwargs)[0] # TODO: Is this the correct way to get the correct agent obs?
                    env_action.append(action)
            reset = False
            new_obs, rew, done, info = self.env.step(env_action) # NOUPDATE - env.step should take a vector of actions

            '''
            Obs: x_me, x_opp --- agent 1. In env: x_1, x_2
            Obs: x_me, x_opp -- agent 2. In env: x_2, x_1
            Env: (n_agents, state_dim)
            '''

            self.num_timesteps += 1

            # Stop training if return value is False
            # if callback.on_step() is False:
            #    break

            # Store transition in the replay buffer.
            # Loop for replay buffer -- either separate or joined. obs[agent_index], action[agent_index], reward[agent_index]
            # Joey: Does this look right to you?
            # print(obs, action, rew, new_obs, done)
            #print("obs",obs[0])
            #print(action)
            #print("ac", action[0])
            #print("rew", rew[0])
            #print("done", done[0])
            for num_agent in range(self.num_agents):
                self.replay_buffer[num_agent].add(obs[num_agent], env_action[num_agent], rew[num_agent], new_obs[num_agent], float(done[num_agent])) # MA-MOD
            obs = new_obs

            # if writer is not None:
            #     ep_rew = np.array([rew]).reshape((1, -1))
            #     ep_done = np.array([done]).reshape((1, -1))
            #     tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
            #                                         self.num_timesteps)

            # TODO: current episode_rewards is a list, make it a list of lists where each list is the reward for each agent in all timesteps
            #     append the newest reward to the end of each list for each agent
            for num_agent in range(self.num_agents): #MA-MOD
                episode_rewards[-1][num_agent] += rew[num_agent]
                if done.any():
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append([0.0] * self.num_agents)
                    reset = True

            # Do not train if the warmup phase is not over
            # or if there are not enough samples in the replay buffer
            for i in range(self.num_agents): # MA-MOD
                can_sample = self.replay_buffer[i].can_sample(self.batch_size)  # MA-MOD
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:

                    # callback.on_rollout_end()

                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert self.beta_schedule is not None, \
                                "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer[i].sample(self.batch_size,
                                                                beta=self.beta_schedule.value(self.num_timesteps))  # MA-MOD
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer[i].sample(self.batch_size)  # MA-MOD
                        weights, batch_idxes = np.ones_like(rewards), None
                    # pytype:enable=bad-unpacking

                    # if writer is not None:
                    #     # run loss backprop with summary, but once every 100 steps save the metadata
                    #     # (memory, compute time, ...)
                    #     if (1 + self.num_timesteps) % 100 == 0:
                    #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    #         run_metadata = tf.RunMetadata()
                    #         summary, td_errors = self._train_step[i](obses_t, actions, rewards, obses_tp1, obses_tp1,
                    #                                               dones, weights, sess=self.sess, options=run_options,
                    #                                               run_metadata=run_metadata)
                    #         writer.add_run_metadata(run_metadata, 'step%d_agent%d' % (self.num_timesteps, i))
                    #     else:
                    #         summary, td_errors = self._train_step[i](obses_t, actions, rewards, obses_tp1, obses_tp1,
                    #                                               dones, weights, sess=self.sess)
                    #     writer.add_summary(summary, self.num_timesteps)
                    # else:
                    td_errors = self._train_step[i](obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps # NOUPDATE
                        assert isinstance(self.replay_buffer[i], PrioritizedReplayBuffer) # MA-MOD
                        self.replay_buffer[i].update_priorities(batch_idxes, new_priorities) # MA-MOD

                    # callback.on_rollout_start()

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target[i](sess=self.sess) # MA-MOD

            if len(episode_rewards[-101:-1]) == 0: # MA-MOD
                mean_100ep_reward = -np.inf
            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1) #MA-MOD

            # below is what's logged in terminal.
            num_episodes = len(episode_rewards) #MA-MOD
            if self.verbose >= 1 and done.any() and log_interval is not None and len(episode_rewards) % log_interval == 0: #MA-MOD
                logger.record_tabular("steps", self.num_timesteps)
                logger.record_tabular("episodes", num_episodes)
                if len(episode_successes) > 0:
                    logger.logkv("success rate", np.mean(episode_successes[-100:]))
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring",
                                        int(100 * self.exploration.value(self.num_timesteps)))
                logger.dump_tabular()

        return self

    def predict(self, observation, agent_idx, state=None, mask=None, deterministic=True): # MA-MOD - added `agent_idx` as a parameter
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model[agent_idx].step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, None


    # No one ever calls this, so we don't need it?
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        print("Should not be called")
        return None
        '''
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Discrete)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))
            if logp:
                actions_proba = np.log(actions_proba)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba
        '''

    def get_parameter_list(self):
        print(self.params)
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "num_agents" : self.num_agents
        }

        params_to_save = self.get_parameters()
        # print(params_to_save)

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
