import random
import numpy as np

# potentially use deque for quicker access if notice speed issues during training
class ReplayMemory:
    def __init__(self, max_mem_size, input_dims):
        self.mem_size = max_mem_size 
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
    def append(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch =random.sample(range(max_mem), batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, new_states, dones

    def __len__(self):
        return self.mem_cntr
