import random
import numpy as np
import torch

class ReplayMemory:
    def __init__(self, max_mem_size, input_dims):
        self.mem_size = max_mem_size 
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, *input_dims), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size, *input_dims), dtype=torch.float32)
        self.action_memory = torch.zeros(self.mem_size, dtype=torch.int32)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool)
    def append(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_.unsqueeze(0)
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
