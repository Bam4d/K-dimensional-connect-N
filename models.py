import tensorflow as tf
import numpy as np
import random
import uuid

class FeedForward:

    def __init__(self, name=None):
        if name:
            self.scope = name
        else:
            self.scope = str(uuid.uuid4())

        self.param_copy_ops = []

    # create a graph here that allows the parameters of the entire network to be set
    def create_set_params_operation(self):
        param_copy_ops = []
        for current_param in self.get_params():
            new_param_placeholder = tf.placeholder(tf.float32, shape=current_param.shape)
            param_copy_ops.append((current_param.assign(new_param_placeholder), new_param_placeholder))

        return param_copy_ops

    def set_session(self, session):
        self.session = session

    def create_layers(self, input, layers, use_bias):

        prev_input = input
        # initialize weights and biases

        for i, layer in enumerate(layers):
            layer_size = layer[0]
            layer_activation = layer[1]

            w = tf.get_variable("W_ff%d"%i, shape=(prev_input.shape.dims[1].value, layer_size), initializer=tf.contrib.layers.xavier_initializer())

            if use_bias:

                b_initialize = tf.zeros(layer_size)

                b = tf.Variable(b_initialize)

            wX = tf.matmul(prev_input, w)
            prev_input = layer_activation(tf.add(wX, b) if use_bias else wX)

        return prev_input

    def l2_regularization_cost(self, params):
        l2_reg_sum = tf.constant(0.0)
        for p in params:
            l2_reg_sum += tf.reduce_sum(tf.square(p))

        return l2_reg_sum

    def set_params(self, new_params):
        for copy_op, input in zip(self.param_copy_ops, new_params):
            self.session.run(copy_op[0], feed_dict={ copy_op[1]: self.session.run(input) })



    def get_params(self):
        return sorted([t for t in tf.trainable_variables() if t.name.startswith(self.scope)], key=lambda v: v.name)

class SimpleFeedForward(FeedForward):

    def __init__(self, n_inputs, n_outputs, layers, learning_rate, use_bias=True, use_l2 = False, name=None):
        FeedForward.__init__(self, name)

        with tf.variable_scope(name):
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs

            self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name='states')
            self.A = tf.placeholder(tf.uint8, shape=(None,), name='actions')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='returns')

            pred = self.create_layers(self.X, layers, use_bias)

            error = self.G - tf.reduce_sum(tf.one_hot(self.A, n_outputs) * pred, axis=1)

            self.cost = tf.reduce_sum(tf.square(error))

            if use_l2:
                self.cost += self.l2_regularization_cost(self.get_params())

            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            self.predict_op = pred

            self.param_copy_ops = self.create_set_params_operation()


    def predict(self, state):
        return self.session.run(self.predict_op, feed_dict={self.X: np.atleast_2d(state)})

    def partial_fit(self, state, actions, G):
        return self.session.run((self.cost, self.train_op),
            feed_dict = {
                self.X: np.atleast_2d(state),
                self.A: np.atleast_1d(actions),
                self.G: np.atleast_1d(G)
            }
        )

    def sample_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.n_outputs)
        else:
            return np.argmax(self.predict(state))

class DeepQ:

    def __init__(self, gamma, model, target_model, batch_sz, episode_state_history_max=10000, episode_state_history_min=100, update_period=100):
        self.episode_state_history_max = episode_state_history_max
        self.episode_state_history_min = episode_state_history_min
        self.episode_state_history = list()
        self.batch_sz = batch_sz

        self.model = model
        self.target_model = target_model

        self.gamma = gamma

        self.train_counter = 0
        self.update_period = update_period

    def train(self):
        # Some condition here for updating the states
        episode_state_history_size = len(self.episode_state_history)
        if episode_state_history_size > self.episode_state_history_min:

            training_sample = random.sample(self.episode_state_history, self.batch_sz)

            states, actions, rewards, next_states, dones = map(np.array, zip(*training_sample))

            G = rewards + np.invert(dones) * self.gamma * np.amax(self.target_model.predict(next_states), axis=1)

            error, _ = self.model.partial_fit(states, actions, G)

            if self.train_counter % self.update_period == 0:
                print("Updating target network at %d steps" % self.train_counter)
                self.update_target_network()
            self.train_counter += 1
            return error

        return 0.0


    def update_target_network(self):
        params = self.model.get_params()
        self.target_model.set_params(params)

    def reset_experience(self):
        self.episode_state_history = list()

    def add_experience(self, experience):
        if len(self.episode_state_history) > self.episode_state_history_max:
            self.episode_state_history.pop(0)
        self.episode_state_history.append(experience)

    def sample_action(self, state, epsilon):
        return self.model.sample_action(state, epsilon)

    def set_session(self,session):
        self.model.set_session(session)
        self.target_model.set_session(session)

class RandomPlayer:

    def __init__(self, n_outputs):
        self.n_outputs = n_outputs

    def train(self):
        return 0.0

    def add_experience(self, experience):
        pass

    def update_target_network(self):
        pass

    def sample_action(self, state, epsilon):
        return np.random.choice(self.n_outputs)