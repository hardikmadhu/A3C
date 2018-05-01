import sys
import tensorflow as tf
import gym
import cv2
import scipy.signal
import threading
import queue
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
# from universe import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger


WORKERS = 8
LEARNING_RATE = 1e-4
#LEARNING_RATE = 0.00025
STACKED_FRAMES = 4                  # Number of frames to stack together. To understand which direction is ball moving.
THREADS = 4                         # Number of Async threads
EPOCHS = 50                         # Number of Epochs for training
TOTAL_FRAMES = 4000000              # Number of actions in each epoch
FREQ_EPOCHS = 4                     # Number of epochs when epsilon reduces to Tolerance
FREQ = 1/(TOTAL_FRAMES*FREQ_EPOCHS) # Frequency for Cos function for epsilon
TOLERANCE = 0.1                     # The minimum value of epsilon
CHANNELS = 1                        # Number of channels for processed frame, 1 - Grayscale, 3 - RGB.
count = 0                           # The master count of actions

print_distance = 0
plot_rewards = []
plot_count = []

training_done = False


def plot_reward_data(plot_iteration):
    """
    This function is called every 500,000 actions.
    The function takes all the training rewards, divides them in window of 50, and plots two lines.
    One for average score of the window, another for maximum score of the window.
    Also, saves the plot as a file for every iteration of function call.

    Args:
        plot_iteration: the postfix number

    """

    global plot_rewards, plot_count
    plot_rewards = [x for _, x in sorted(zip(plot_count, plot_rewards))]
    plot_count = sorted(plot_count)

    episode_count = 50
    new_plot_rewards = [np.average(plot_rewards[i*episode_count:(i+1)*episode_count]) for i in
                        range(int(len(plot_rewards)/episode_count))]

    new_max_rewards = [max(plot_rewards[i*episode_count:(i+1)*episode_count]) for i in
                       range(int(len(plot_rewards)/episode_count))]

    new_plot_count = [plot_count[int((i*episode_count)+(episode_count/2))] for
                      i in range(int(len(plot_rewards)/episode_count))]

    fig = plt.figure(figsize=(15, 10))
    min_len = min(len(new_plot_count), len(new_plot_rewards))
    plt.plot(new_plot_count[0:min_len], new_plot_rewards[0:min_len], label="Average Score")

    min_len = min(len(new_plot_count), len(new_max_rewards))
    plt.plot(new_plot_count[:min_len], new_max_rewards[:min_len], label="Maximum Score")

    plt.xlabel("Number of Actions")
    plt.ylabel("Score")
    plt.title("Base A3C 42*42")
    plt.legend()

    plt.savefig("TrainingScore"+str(plot_iteration)+".jpg")
    plt.close()

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def discount(x, gamma):
   return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def preprocess(obs):
    """
    Preprocess received state before the DNN processing
    Convert from RGB to Grayscale, resize to 84*84, scale all the pixel values from 0 to 1.
    Processed frame is reshaped to (1, 84, 84, 1), so multiple frames can be stacked together
    and multiple batches of stacked frames can be trained together.

    Args:
        obs: raw image frame received from "act" function
    """

    if CHANNELS == 1:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(obs, (84, 84))
    scaled = resized.astype('float32') / 255.0
    processed = scaled.reshape(1, scaled.shape[0], scaled.shape[1], 1)
    #print("preprocessed", processed.shape)

    return processed


class RunnerThread(threading.Thread):

    def __init__(self, env, policy):
        threading.Thread.__init__(self)
        self.env = env
        self.policy = policy
        self.queue = queue.Queue(5)


    def start_runner(self, sess):
        self.sess = sess
        self.start()

    def run(self):
        with self.sess.as_default():
            self.training_thread(self.env)

    def training_thread(self, env):
        """
        The training function, that is primary function of each thread.
        This function sets up the game, takes action for its on environment, calculate value and advantage,
        send data to DNN for processing.

        Args:
            agent: Only one agent is shared throughout the process of training across multiple threads.
            game: Name of the game.
            thread_index: Index of this particular thread

        """

        global count, training_done, prev_print_count, plot_rewards, plot_count

        # print(env.action_space.n)

        # Setting up the environment
        actions = list(range(env.action_space.n))
        action_space_len = env.action_space.n
        print(env.unwrapped.get_action_meanings())

        # Initialisation of local variables.
        stacked_state = []
        terminal = True
        raw_rewards = []
        state = None

        # Total run time : EPOCHS(50) * TOTAL_FRAMES(4 Million) = 200 Million Frames
        while count < EPOCHS*TOTAL_FRAMES:

            states = []
            rewards = []
            actions_taken = []
            values = []
            r = []

            # If a terminal state is reached, reset the game environment and start with a new run
            if terminal:
                terminal = False
                reset_state = env.reset()
                state = preprocess(reset_state)

                stacked_state = np.repeat(state, STACKED_FRAMES, axis=3)
                #print("Stacked State First terminal", stacked_state.shape)

            skip_count = 4         # Number of frames to be skipped

            while not terminal and len(states) < 20:
                count += 1
                states.append(stacked_state)
                # print(len(states))
                #policy, value = sess.run([tf_policy, tf_value], {tf_state: stacked_state, rnn_state_in: rnn_state_batch})

                # Get the policy and value for current state
                # Policy will be used to take action, and value will be used to calculate advantage
                policy, value = self.policy.act(stacked_state)

                # Select an action according to the policy
                # action_idx = np.random.choice(action_space_len, p=policy)
                action_idx = policy.argmax()
                skip_rewards = 0
                """
                prev_state = None

                for d in range(skip_count):
                    state, reward, terminal, _ = env.step(actions[action_idx])
                    skip_rewards += reward
                    if terminal:
                        break

                    prev_state = state
                    
                # Atari 2600 had a problem where some objects can be missing between consecutive frames.
                # In a situation like that, we take a pixel by pixel maximum of two previous frames to decide
                # which one to select
                if prev_state is not None:
                    state = np.maximum.reduce([state, prev_state])

                reward = skip_rewards
                """

                state, reward, terminal, _ = env.step(actions[action_idx])

                if len(raw_rewards) > 2400:
                    terminal = True

                # Print performance of each round of game
                if terminal:
                    if True:
                        prev_print_count = count
                        print("Iterations", count)
                        print("Rewards", len([r for r in raw_rewards if r > 0]))
                        print("Game steps", len(raw_rewards))

                    plot_rewards.append(len([r for r in raw_rewards if r > 0]))
                    plot_count.append(count)
                    raw_rewards = []

                processed_state = preprocess(state)

                stacked_state = np.append(processed_state, stacked_state[:, :, :, :CHANNELS*(STACKED_FRAMES-1)], axis=3)

                reward = np.clip(reward, -1, 1)
                raw_rewards.append(reward)
                rewards.append(reward)
                actions_taken.append(policy)
                values.append(value[0])

            # Get what is the value of current state to calculate advantage.
            target_value = 0
            if not terminal:
                #target_value = sess.run(tf_value, {tf_state: stacked_state, rnn_state_in: rnn_state_batch}).flatten()[0]
                #target_value = sess.run(tf_value, {tf_state: stacked_state}).flatten()[0]

                target_value = self.policy.get_value(stacked_state)
                target_value = target_value[0]

            r.append(target_value)

            # Calculate the advantage
            target_values = []
            for reward in reversed(rewards):
                target_value = reward + 0.99 * target_value
                target_values.append(target_value)

            target_values.reverse()

            total_advantages = np.array(target_values) - np.array(values)

            # Generalised Advantage Estimation
            gamma = 0.99
            lambda_ = 1.0
            vpred_t = np.asarray(values + r)
            rewards_plus_v = np.asarray(rewards + r)
            batch_r = discount(rewards_plus_v, gamma)[:-1]
            delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
            batch_adv = discount(delta_t, gamma * lambda_)

            states = np.vstack(states)

            #mutex.acquire()

            batch_train_op = [states, actions_taken, target_values, total_advantages]
            self.queue.put(batch_train_op, block=True)

            #mutex.release()

        training_done = True


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class Policy(object):
    def __init__(self, ac_space):
        self.input_size = [84, 84, 4]
        height = self.input_size[0]
        width = self.input_size[1]
        channels = self.input_size[2]
        outputs = [16, 32, 256]
        kernel_sizes = [[8, 8], [4, 4]]
        strides = [[4, 4], [2, 2]]
        paddings = ['SAME', 'SAME']

        self.x = x = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='state')

        cnn1 = tf.contrib.layers.convolution2d(inputs=x, num_outputs=outputs[0], kernel_size=kernel_sizes[0],
                                               stride=strides[0], padding=paddings[0], activation_fn=tf.nn.relu,
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               biases_initializer=tf.zeros_initializer())

        cnn2 = tf.contrib.layers.convolution2d(inputs=cnn1, num_outputs=outputs[1], kernel_size=kernel_sizes[1],
                                               stride=strides[1], padding=paddings[1], activation_fn=tf.nn.relu,
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               biases_initializer=tf.zeros_initializer())

        flatten = tf.contrib.layers.flatten(inputs=cnn2)

        fc = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=outputs[2], activation_fn=tf.nn.relu,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())

        self.policy = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=ac_space,
                                                   activation_fn=tf.nn.softmax,
                                                   weights_initializer=normalized_columns_initializer(0.01),
                                                   biases_initializer=None)

        self.value = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=1, activation_fn=None,
                                                  weights_initializer=normalized_columns_initializer(1.0),
                                                  biases_initializer=None)

        self.sample_policy = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=ac_space, activation_fn=None,
                                                        weights_initializer=normalized_columns_initializer(0.01),
                                                        biases_initializer=None)

        self.sample = categorical_sample(self.sample_policy, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, ob):
        sess = tf.get_default_session()
        policy, value = sess.run([self.sample, self.value], {self.x: ob})

        policy, value = policy.flatten(), value.flatten()
        return policy, value

    def get_value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.value, {self.x: ob})[0]


class A3C(object):
    def __init__(self, env, task):
        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = Policy(env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = Policy(env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.sample_policy)
            prob_tf = tf.nn.softmax(pi.sample_policy)

            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.value - self.r))

            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            self.runner = RunnerThread(env, pi)
            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))

            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            opt = tf.train.AdamOptimizer(1e-4)

            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.local_steps = 0

    def start(self, sess):
        self.runner.start_runner(sess)

    def process(self, sess):
        sess.run(self.sync)
        batch_train_op = self.runner.queue.get(block=True)

        # print("ac", batch_train_op[1])
        # print("value", batch_train_op[2])
        # print("advantages", batch_train_op[3])

        feed_dict = {
            self.local_network.x: batch_train_op[0],
            self.ac: batch_train_op[1],
            self.r: batch_train_op[2],
            self.adv: batch_train_op[3]
        }

        sess.run([self.train_op, self.global_step], feed_dict=feed_dict)

        self.local_steps += 1


def create_atari_env(env_id):
    env = gym.make(env_id)
    """
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    """
    return env


def worker_run(server, game, task):
    env = create_atari_env(game)
    trainer = A3C(env, task)

    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.initialize_all_variables()

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def init_fn(ses):
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(task)])

    sv = tf.train.Supervisor(is_chief=(task == 0),
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess)
        global_step = sess.run(trainer.global_step)

        while not sv.should_stop() and (global_step < EPOCHS * TOTAL_FRAMES):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)


def cluster_spec(num_workers, num_ps):
    """ More tensorflow setup for data parallelism """

    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def atari_bot(job, task, game="Breakout-v0"):
    print(game)
    spec = cluster_spec(WORKERS, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    if job == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        worker_run(server, game, task)

    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=0,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        print("PS Server Started")
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    jobs = ["ps"] + THREADS * ["worker"]
    job_idxs = [0] + list(range(0, THREADS))

    if len(sys.argv) > 1:
        game = sys.argv[1]
    else:
        game = 'Breakout-v0'

    trainer_threads = []
    for i in range(len(job_idxs)):
        job = jobs[i]
        trainer_threads.append(threading.Thread(target=atari_bot, args=(jobs[i], job_idxs[i], game,)))

    plot_distance = 500000
    prev_plot_count = 0
    plot_iteration = 0

    for thread in trainer_threads:
        thread.daemon = True
        thread.start()

    while not training_done:
        if count - prev_plot_count > plot_distance:
            prev_plot_count = count
            plot_reward_data(plot_iteration)
            plot_iteration = plot_iteration + 1

        time.sleep(0.01)

    for thread in trainer_threads:
        thread.join()
