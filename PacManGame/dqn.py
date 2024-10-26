# Import Dependencies
import torch # PyTorch library for building and training neural networks
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


# Designing DQN Model
class DQN(nn.Module): # defines a new neural network model that inherits from Pytorch's base class nn.module
	def __init__(self, input_dims, num_actions, lr=0.001, fc1_dims=512, fc2_dims=512): 
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
			dummy_input = torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])
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
	
# if __name__ == '__main__':
# 	lr = 0.001
# 	input_dims = (6, 50, 80)
# 	fc1_dims = 512
# 	fc2_dims = 256
# 	num_actions = 4
# 	net = DQN(lr, input_dims, fc1_dims, fc2_dims, num_actions)
# 	state = torch.rand(1, *input_dims)
# 	action = net(state)
# 	print(action)
