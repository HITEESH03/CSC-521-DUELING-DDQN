# Import Libraries
import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Add, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Fix NumPy compatibility issue
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Environment Setup
env = gym.make('ALE/Breakout-v5')
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Preprocess function
def preprocess(state):
    state = np.mean(state, axis=2).astype(np.uint8)  # grayscale
    state = state[::2, ::2]  # downsample by factor of 2
    return state / 255.0  # normalize

# Dueling DDQN Class
class DuelingDDQN:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.train_start = 1000

        # Main and target models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        inputs = Input(shape=(105, 80, 1))

        layer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
        layer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(layer)
        layer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(layer)
        layer = Flatten()(layer)

        # Dueling streams
        value_fc = Dense(512, activation='relu')(layer)
        value = Dense(1)(value_fc)

        advantage_fc = Dense(512, activation='relu')(layer)
        advantage = Dense(self.action_size)(advantage_fc)

        # Combining streams
        def combine_streams(inputs):
            value, advantage = inputs
            return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

        q_values = Lambda(combine_streams)([value, advantage])

        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)[0]
            if done:
                target[action] = reward
            else:
                a = np.argmax(self.model.predict(next_state, verbose=0)[0])
                t = self.target_model.predict(next_state, verbose=0)[0][a]
                target[action] = reward + self.gamma * t
            
            states.append(state[0])
            targets.append(target)

        self.model.train_on_batch(np.array(states), np.array(targets))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the agent
episodes = 100
agent = DuelingDDQN(state_shape, action_size)
scores, episodes_list = [], []

for e in range(episodes):
    done = False
    state, _ = env.reset()
    state = preprocess(state)
    state = np.reshape(state, [1, 105, 80, 1])
    score = 0

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess(next_state)
        next_state = np.reshape(next_state, [1, 105, 80, 1])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            agent.update_target_model()
            scores.append(score)
            episodes_list.append(e)
            print(f"episode: {e}/{episodes}, score: {score}, epsilon: {agent.epsilon:.2}")

        agent.replay()

# Plotting results
plt.plot(episodes_list, scores)
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Dueling DDQN on Atari Breakout')
plt.show()
