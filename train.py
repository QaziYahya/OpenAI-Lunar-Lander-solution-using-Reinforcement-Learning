import gym
import numpy as np
import PIL.Image
from collections import namedtuple, deque

import tensorflow as tf
from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

import random

Display(visible=False, size=(840, 480)).start()
tf.random.set_seed(0)

env = gym.make("LunarLander-v2")
env.reset()

MEMORY_SIZE = 100_000
GAMMA = 0.99
ALPHA = 0.001
NUM_STEPS_FOR_UPDATE = 4

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

num_states = env.observation_space.shape
num_actions = env.action_space.n

q_network = Sequential([
    Input(shape=num_states),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_actions, activation="linear")
])

target_q_network = Sequential([
    Input(shape=num_states),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_actions, activation="linear")
])

optimizer = Adam(learning_rate=ALPHA)

def compute_loss(experiences, gamma, q_network, target_q_network):
  states, actions, rewards, next_states, done_vals = experiences
  max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
  y_targets = rewards + (gamma * max_qsa * (1-done_vals))
  q_values = q_network(states)
  q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
  loss = MSE(y_targets, q_values)

  return loss

def update_target_network(q_network, target_q_network):
  TAU=1e-3
  for target_weights, q_network_weights in zip(target_q_network.weights, q_network.weights):
    target_weights.assign(TAU * q_network_weights + (1.0-TAU) * target_weights)

@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(experiences, gamma, q_network, target_q_network)
  gradients = tape.gradient(loss, q_network.trainable_variables)
  optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
  update_target_network(q_network, target_q_network)

def get_action(q_values, epsilon=0):
  if random.random() > epsilon:
    return np.argmax(q_values.numpy()[0])
  else:
    return random.choice(np.arange(4))

def check_update_conditions(j, NUM_STEPS_FOR_UPDATE, memory_buffer):
  if(j+1) % NUM_STEPS_FOR_UPDATE == 0 and len(memory_buffer) > 64:
    return True
  else:
    return False

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=64)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)

def get_new_epsilon(epsilon):
  E_MIN = 0.01
  E_DECAY = 0.99
  return max(E_MIN, E_DECAY * epsilon)

def train():

  NUM_EPISODES = 2000
  MAX_TIMESTEPS = 1000

  memory_buffer = deque(maxlen=MEMORY_SIZE)
  target_q_network.set_weights(q_network.get_weights())

  epsilon = 1.0

  points_history = []

  for i in range(NUM_EPISODES):

    state = env.reset()
    total_points = 0

    for j in range(MAX_TIMESTEPS):

      state_qn = np.expand_dims(state, axis=0)
      q_values = q_network(state_qn)
      action = get_action(q_values, epsilon)
      next_state, reward, done, _ = env.step(action)

      memory_buffer.append(experience(state, action, reward, next_state, done))

      update = check_update_conditions(j, NUM_STEPS_FOR_UPDATE, memory_buffer)

      if update:
        experiences = get_experiences(memory_buffer)
        agent_learn(experiences, GAMMA, q_network, target_q_network, optimizer)

      state = next_state.copy()
      total_points += reward

      if done:
        break

    points_history.append(total_points)
    avg_points = np.mean(points_history[-100:])

    epsilon = get_new_epsilon(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {100} episodes: {avg_points:.2f}", end="")

    if (i+1) % 100 == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {100} episodes: {avg_points:.2f}")

    if(avg_points >= 200):
      print(f"Environment solved in {i+1} episodes!")
      q_network.save('./lunar_lander_model.h5')
      break

train()