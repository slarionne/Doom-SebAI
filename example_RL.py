import numpy as np

class QLearningModel:
  def __init__(self, num_states, num_actions, learning_rate, discount_factor):
    self.num_states = num_states
    self.num_actions = num_actions
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.q_values = np.zeros((num_states, num_actions))

  def choose_action(self, state, epsilon):
    """
    Choose an action for the given state using an epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) < epsilon:
      # Explore: choose a random action
      action = np.random.choice(self.num_actions)
    else:
      # Exploit: choose the action with the highest Q-value
      action = np.argmax(self.q_values[state])
    return action

  def update(self, state, action, next_state, reward):
    """
    Update the Q-value for the given state-action pair using the Bellman equation.
    """
    q_value = self.q_values[state][action]
    next_max_q = np.max(self.q_values[next_state])
    new_q = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.discount_factor * next_max_q)
    self.q_values[state][action] = new_q
