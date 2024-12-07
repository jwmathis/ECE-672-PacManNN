import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN
from custom_environment import PacMan, DinoGame
from datetime import datetime, timedelta
import argparse
import itertools

import os


# For printing date and time
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent:
    def __init__(self, hyperparameter_set):
        with open("c:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\hyperparameters.yml", 'r') as file:
            all_hyperparameters_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_set[hyperparameter_set]
        # Hyperparameters (adjustable)    
        self.env_id = hyperparameters['env_id']                            # Environment ID
        self.learning_rate_a = hyperparameters['learning_rate_a']          # Learning rate (alpha)
        self.discount_factor_g = hyperparameters['discount_factor_g']      # Discount factor (gamma)
        self.network_sync_step = hyperparameters['network_sync_step']      # Number of steps before updating target network
        self.replay_memory_size = hyperparameters['replay_memory_size']    # Size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size']          # Size of the training data set sampled from the replay memory
        self.epsilon_init = hyperparameters['epsilon_init']                # 1 = 100% random actions, 0.1 = 10% random actions
        self.epsilon_decay = hyperparameters['eps_decay']                  # Rate at which epsilon decreases
        self.epsilon_end = hyperparameters['eps_end']                      # Minimum value of epsilon
        # Neural Network
        self.loss_fn = nn.MSELoss() # NN Loss function. MSE = Mean Squared Error can be swapped to something else
        self.optimizer = None       # NN optimizer. Initialize later
        self.step_count = 0
        self.total_steps = 10000
        # Path to Run Info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pth")
        self.REWARDS_GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}_rewards.png")
        self.LOSS_GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}_loss.png")
        self.EPSILON_GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}_epsilon.png")
        self.Q_VALUES_GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}_q_values.png") 
    def load_model(self, model_path, policy_dqn):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load the state dictionary into the policy network
        policy_dqn.load_state_dict(checkpoint['state_dict'])
        
        # Print out the saved epsidoe and epsilon
        episode = checkpoint.get('episode', 'N/A')
        epsilon = checkpoint.get('epsilon', 'N/A')
        print(f"Model loaded from '{model_path}'")
        print(f"Episode at save: {episode}")
        print(f"Epsilon at save: {epsilon}")
        
        # Switch model to evaluation mode
        policy_dqn.eval()
        
        # Return the loaded model for use in 'run'
        return policy_dqn      
    
    def run(self, is_training=True, render=False, model_path=None):
        env = DinoGame()

        num_actions = env.action_space.n
        input_dims = env.observation_space.shape
        
        rewards_per_episode = []
        epsilon_history = []
        loss_history = []
        last_graph_update_time = datetime.now()
        self.q_value_deltas = []
        
        policy_dqn = DQN(input_dims, num_actions).to(device)
        
        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init
            
            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size, input_dims)
            
            # Create the target network and make it identical to the policy network
            target_dqn = DQN(input_dims, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Policy network optimizer. "Adam" optimizer can be swapped to something else
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            
            # Track number of steps taken. Used for syncing policy => target network
            step_count = 0
            
            # List to keep track of epsilon decay
            epsilon_history = []
            
            # Track best reward
            best_reward = -99999999
        else:
            if model_path:
                # Load learned policy
                policy_dqn = self.load_model(model_path, policy_dqn)
            else:
                print("No model file specified.")
                return
        
        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():    
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            
            terminated = False
            episode_reward = 0.0
            
            while not terminated:
                
                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # Select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    # Select best action
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                
                # Execute action. Truncated and info is not used
                new_state, reward, terminated, _, info = env.step(action.item())
                # print(f"Obstacle type: {obstacle_type}")
                # Accumulate reward
                episode_reward += reward
                
                # Increment step count
                self.step_count += 1
                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                
    
                if is_training:
                    # Save experience into memory
                    memory.append(state, action, reward, new_state, terminated)

                    # Increment step counter
                    step_count += 1
                    
                # Move to new state
                state  = new_state
             
            rewards_per_episode.append(episode_reward)    
            
            if is_training:
                if episode % 100 == 0:
                    checkpoint = {'state_dict': policy_dqn.state_dict(), 
                                  'optimizer': self.optimizer.state_dict(), 
                                  'step_count': self.step_count, 
                                  'epsilon': epsilon,
                                  'episode': episode}
                    model_filename = self.MODEL_FILE.replace('.pth', f'_{episode}.pth')
                    torch.save(checkpoint, model_filename)   
                                       
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_rewards_graph(rewards_per_episode)
                    self.save_epsilon_graph(epsilon_history)
                    last_graph_update_time = current_time
                    self.save_q_value_graph(self.q_value_deltas)
                epsilon = max(self.epsilon_end + (self.epsilon_init - self.epsilon_end) * (1 - self.step_count / self.total_steps), self.epsilon_end)
                epsilon_history.append(epsilon)
                print(f"Episode {episode+1}, episode reward: {episode_reward:.1f}, epsilon: {epsilon:.2f}")
                
                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    
                    # Sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    
                    loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                    loss_history.append(loss)
                    self.save_loss_graph(loss_history)
                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_step:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
            if not is_training:
                print(f"Test Episode {episode+1}, episode reward: {episode_reward:.1f}")
                      
    def optimize(self, mini_batch, policy_dqn, target_dqn):
       
       # Stack tensors to create batch tensors
       states = mini_batch[0].to(device)
       actions = mini_batch[1].to(device)
       rewards = mini_batch[2].to(device)
       new_states = mini_batch[3].to(device)
       terminations = mini_batch[4].float().to(device) # Convert bool to float for later calculation
       
       with torch.no_grad():
           # Calculate target Q-values (expected returns)
           next_q = target_dqn(new_states).max(dim=1)[0]
           target_q = rewards + (self.discount_factor_g * next_q * (1 - terminations))
            
        # Calculate current Q-values from current policy
       current_q = policy_dqn(states).gather(1, index=actions.unsqueeze(-1).long()).squeeze(-1)
       
       # Track Q-value deltas
       q_value_delta = torch.abs(current_q - target_q).mean().item()
       self.q_value_deltas.append(q_value_delta) # Store the mean delta for this batch
            
        # Compute loss for the whole minibatch
       loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
       self.optimizer.zero_grad() # Clear gradients
       loss.backward()            # Compute gradients (backpropagation)
       self.optimizer.step()      # Update network parameters i.e. weights and biases
       print(f'Loss: {loss}')
       return loss
   
    def save_rewards_graph(self, rewards_per_episode):
        # save plots
        fig = plt.figure(1)
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(rewards_per_episode)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.ylabel("Average reward")
        plt.plot(mean_rewards)
        # Save figure
        plt.savefig(self.REWARDS_GRAPH_FILE)
        plt.close(fig)
        
    def save_epsilon_graph(self, epsilon_history):
        fig = plt.figure(2)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)
        # Save figure
        plt.savefig(self.EPSILON_GRAPH_FILE)
        plt.close(fig)
        
    def save_loss_graph(self, loss_history):
        loss_history = torch.tensor(loss_history)
        loss_history_array = loss_history.cpu().numpy()
        fig = plt.figure(3)
        plt.ylabel("Loss")
        plt.plot(loss_history_array)
        plt.savefig(self.LOSS_GRAPH_FILE)
        plt.close(fig)
        
    def save_q_value_graph(self, q_value_deltas):
        fig = plt.figure(4)
        plt.ylabel("Mean Q-Value Delta")
        plt.plot(q_value_deltas)
        plt.savefig(self.Q_VALUES_GRAPH_FILE)
        plt.close(fig)
        
def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)
          
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument("hyperparameters", help='')
    parser.add_argument("--train", help='Training mode', action='store_true')
    parser.add_argument("--test", help='Testing mode', action='store_true')
    parser.add_argument("--load-model", help='Load pre-trained model', type=str)
    parser.add_argument("--yaml", help='Path to YAML configuration file', type=str, required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.yaml)
    
    dql = Agent(hyperparameter_set=args.hyperparameters)
    
    if args.train:
        dql.run(is_training=True)
        
    elif args.test:
        model_path = args.load_model if args.load_model else config.get('model_path', None)
        
        if model_path:
            dql.run(is_training=False, model_path=model_path)
        else:
            print("Error: --load-model must be specified when testing and is missing from the YAML config.")
            parser.print_help()