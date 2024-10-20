from custom_environment import PacMan
from dqn_model import DQN, DQNAgent
import numpy as np
from dqn_model import plot_learning_curve
import torch
import itertools

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = PacMan()
    
    agent = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, num_actions=4, 
                     eps_end=0.01, eps_dec=0.998, input_dims=(6, 50, 80), lr=0.001)
    agent.q_eval.to(device)
    agent.q_target.to(device)
    
    scores, eps_history = [], []
    n_games = 220
    best_score = -np.inf # initialize best score to a low value
    save_folder = 'saved_models'
    learn_interval = 10 # Call learn() every 10 steps
    warmup_eps = 20 # Episodes before the agent starts learning
    step_counter = 0 
    
    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        loss = None
        previous_loss = float('inf')
        has_learned = False
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            
            agent.store_transition(observation, action, reward, observation_, done)
            
            step_counter += 1
            # Start learning only after warm-up period and learn periodically
            if i > warmup_eps and step_counter % learn_interval == 0:
                loss = agent.learn() # this is batch learning
                has_learned = True
            observation = observation_
            # env.render()
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        # Calcualte average score over the last 100 games
        avg_score = np.mean(scores[-100:])
        
        print('------------------------')
        print(f'| episodes   |   {i}    |')
        print(f'| score      |  {score} |')
        print(f'| epsilon    |  {agent.epsilon:.2f}   |')
        print('------------------------')
        if loss is not None:
            print(f', loss {loss:.4f}')
        else: 
            print() # Just print a new line if loss is None
            
        if has_learned and loss < previous_loss:
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
            # print(f"Updated epsilon: {agent.epsilon}") # debug statement
        previous_loss = loss if loss is not None else previous_loss

        # Save the model every 10 episodes
        if has_learned and i % 10 == 0:
            model_filename = f'{save_folder}/best_pacman_dqn_model_{i}.pth'
            torch.save(agent.q_eval.state_dict(), model_filename)
            print(f'Model saved as {model_filename}')

    x = [i+1 for i in range(n_games)]
    filename = 'pacman_plot.png'
    plot_learning_curve(x, scores, eps_history, filename)
    
    