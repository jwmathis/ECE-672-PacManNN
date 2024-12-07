# Import Dependencies
import torch # PyToch library for building and training neural networks
from torch import nn
from torch.optim import Adam
import numpy as np # for numerical calculations
from collections import namedtuple, deque # provides useful data structures may not need
import random # for random sampling 
from mss import mss # for grabbing a screen shot of a monitor 
import pydirectinput # for mouse and keyboard input on windows
import cv2 as cv # for image and video processing
import pytesseract # OCR tool for reading text from images
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import env_checker  # Import the environment checker
from collections import deque

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
	fig=plt.figure()
	ax=fig.add_subplot(111, label="1")
	ax2=fig.add_subplot(111, label="2", frame_on=False)

	ax.plot(x, epsilons, color="C0")
	ax.set_xlabel("Training Steps", color="C0")
	ax.set_ylabel("Epsilon", color="C0")
	ax.tick_params(axis='x', colors="C0")
	ax.tick_params(axis='y', colors="C0")

	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

	ax2.scatter(x, running_avg, color="C1")
	ax2.axes.get_xaxis().set_visible(False)
	ax2.yaxis.tick_right()
	ax2.set_ylabel('Score', color="C1")
	ax2.yaxis.set_label_position('right')
	ax2.tick_params(axis='y', colors="C1")

	if lines is not None:
		for line in lines:
			plt.axvline(x=line)

	plt.savefig(filename)


# Designing DQN Model
class DQN(nn.Module): # defines a new neural network model that inherits from Pytorch's base class nn.module
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, num_actions): 
		super(DQN, self).__init__() # calls the initializer of the parent class nn.module 
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=8, stride=4) # convolutional layer with 32 filters, each of size 8 x8, applied with a stride of 4
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # convolutional layer with 64 filters, each of size 4 x 4, applied with a stride of 2
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # convolutional layer with 64 filters, each of size 3 x 3
		self.fc_input_size = self._calculate_fc_input_size(self.input_dims)
		self.fc1 = nn.Linear(self.fc_input_size, self.fc1_dims) # fully connected layer with 512 units
		self.fc2 = nn.Linear(self.fc2_dims, num_actions) # final fully connected layer with output units equal to the number of possible actions
		self.optimizer = Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)
		
	def _calculate_fc_input_size(self, input_dims):
		with torch.no_grad():
			dummy_input = torch.zeros(1, input_dims[0], 50, 80)
			x = torch.relu(self.conv1(dummy_input))
			x = torch.relu(self.conv2(x))
			x = torch.relu(self.conv3(x))
			
			return x.view(1, -1).size(1)  # Flatten and get the size
		
	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.relu(self.conv3(x))
		x = x.view(x.size(0), -1) # Flatten the output from conv layers
		x = torch.relu(self.fc1(x))
		actions =  self.fc2(x)  # Output Q-Values for each action
		
		return actions
	
# Creating Replay Buffer
class ReplayBuffer:
    def __init__(self, max_mem_size, input_dims):
        self.mem_size = max_mem_size 
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch =random.sample(range(max_mem), batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, new_states, dones
  
# Creating DQN Agent
class DQNAgent:       
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_actions,
				 max_mem_size=10000, eps_end=0.01, eps_dec=0.99, replace_target=1000):
		self.gamma = gamma # Determines the weighting of future rewards
		self.epsilon = epsilon
		self.epsilon_end = eps_end
		self.epsilon_decay = eps_dec
		self.lr = lr
		self.action_space = [i for i in range(num_actions)]
		self.num_actions = num_actions
		self.batch_size = batch_size
		self.replace_target = replace_target # Number of steps before updating target network
		self.learn_step_counter = 0 # Track the steps for target network update
		
		# Initialize evaluation (q_eval) and target (q_target) network
		self.q_eval = DQN(self.lr, input_dims, fc1_dims=512, fc2_dims=512, num_actions=num_actions)
		self.q_target = DQN(self.lr, input_dims, fc1_dims=512, fc2_dims=512, num_actions=num_actions)
  
		# Initially, the target network has the same weights as the evaluation network
		self.q_target.load_state_dict(self.q_eval.state_dict())
		self.q_target.eval()
  
		# Initialize the replay buffer
		self.memory = ReplayBuffer(max_mem_size, input_dims)
		self.optimizer = Adam(self.q_eval.parameters(), lr=self.lr)
		self.loss = nn.MSELoss()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)
  
	# Method for choosing an action
	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			observation = observation / 255.0
			state = torch.tensor([observation], dtype=torch.float32).to(self.q_eval.device)
			actions = self.q_eval.forward(state)
			action = torch.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)
		return action

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_target == 0:
			self.q_target.load_state_dict(self.q_eval.state_dict())
			print('Target network updated.')
	
	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return
		self.optimizer.zero_grad()
		
		self.replace_target_network()
  
		states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
		
		states = torch.tensor(states).to(self.q_eval.device)
		rewards = torch.tensor(rewards).to(self.q_eval.device)
		dones = torch.tensor(dones, dtype=torch.float32).to(self.q_eval.device)
		actions = torch.tensor(actions).to(self.q_eval.device)
		states_ = torch.tensor(states_).to(self.q_eval.device)
		
		# Q-values for the next state from the target network (for stability)
		q_next = self.q_target.forward(states_).max(dim=1)[0]

		# Q-values for current state from evaluation network
		q_pred = self.q_eval.forward(states)[range(self.batch_size), actions]
		
		# Calculate target Q-values using Bellman equation
		q_target = rewards + self.gamma *  q_next * (1 - dones)
		
		# Compute loss between predicted Q-values and target Q-values
		loss = self.loss(q_pred, q_target)
		loss.backward()
		self.optimizer.step()

		# Increment the step counter		
		self.learn_step_counter += 1
  
		return loss.item()