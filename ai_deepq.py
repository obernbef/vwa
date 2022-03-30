# %%
from keras.layers import Dense
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import numpy as np
from numpy.typing import NDArray, ArrayLike

InputShape = ArrayLike
"""The Input Shape of the Environment. Has to be a one dimensional Vector.
Example ConnectFour: can be achieved by using np.array(board).reshape(-1, self.number_of_inputs)"""


class ReplayBuffer(object):
    """docstring for ReplayBuffer."""
    def __init__(self,
                 max_size: int,
                 input_shape: InputShape,
                 n_actions: int,
                 discrete: bool = False) -> None:
        self.mem_cntr: int = 0
        self.mem_size: int = max_size
        self.discrete: bool = discrete
        self.state_memory: NDArray = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_trainsition(self, state, action, reward, new_state, done) -> None:
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        # done == false --> terminal_memory stores 1, so true
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem: int = min(self.mem_cntr, self.mem_size)
        # batch: NDArray = np.random.choice(max_mem, batch_size)
        batch: NDArray = np.array([i for i in range(max_mem - batch_size, max_mem)])
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


# fcl = fully connected layers
def build_dqn(learning_rate, n_actions, input_dims) -> Sequential:
    model: Sequential = Sequential([
        Dense(50, input_shape=(input_dims, ), activation='sigmoid'),
        Dense(50, activation='sigmoid'),
        Dense(50, activation='sigmoid'),
        Dense(50, activation='sigmoid'),
        Dense(50, activation='sigmoid'),
        Dense(n_actions)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model


class ConnectFourAgent(object):
    def __init__(self,
                 alpha,
                 gamma,
                 n_actions,
                 epsilon,
                 batch_size,
                 input_dims,
                 epsilon_decrement=0.996,
                 epsilon_end=0.01,
                 mem_size=1_000_000,
                 fname='dqn_model.h5') -> None:
        self.action_space = [i for i in range(n_actions)]
        # Set of available Actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_decrement
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory: ReplayBuffer = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        self.q_eval = build_dqn(alpha, n_actions, input_dims)
        # Q Network

    def remember(self, state, action, reward, new_state, done) -> None:
        self.memory.store_trainsition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # Pass state through network, get value for all actions for
            # that particular state and select the action that has the
            # maximum value
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
            self.epsilon_min else self.epsilon_min

    def save_model(self):
        with open(self.model_file, "w") as f:
            f.truncate(0)
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)


# %%
