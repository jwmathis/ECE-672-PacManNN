import torch
import itertools
import yaml
import random
from custom_environment import PacMan
from dqn import DQN
from experience_replay import ReplayMemory

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameter_set.yaml', 'r') as file:
            all_hyperparameters_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_set[hyperparameter_set]
            
        self.replay_memory_size = hyperparameters['replay_memory_size']    
        self.gamma = hyperparameters['gamma']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.lr = hyperparameters['lr']
        self.batch_size = hyperparameters['batch_size']
    def run(self, is_training=True, render=False):
        env = PacMan()

        num_actions = env.action_space.n
        input_dims = env.observation_space.shape[0]
        
        rewards_per_episode = []
        epsilon_history = []
        
        policy_dqn = DQN(input_dims, num_actions).to(device)
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size, input_dims)
            
            epsilon = self.epsilon_init
            
        for episode in itertools.count():    
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            
            terminated = False
            episode_reward = 0.0
            
            while not terminated:
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax()
                
                # Processing
                new_state, reward, terminated, _, info = env.step(action)
                
                # Accumulate reward
                episode_reward += reward
                
                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                
                if is_training:
                    memory.append(state, action, reward, new_state, terminated)
                
                # Move to new state
                state = new_state
            
            rewards_per_episode.append(episode_reward)    
            
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)