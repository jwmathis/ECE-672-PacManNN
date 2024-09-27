from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN

class WebGame(Env):
    def __init__(self):
        
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        
        self.cap = mss()
        self.game_location = {'top':400, 'left':-2500, 'width':600, 'height':500}
        self.done_location = {'top':200, 'left':-1830, 'width':1030, 'height':300}     
    
    # What is called to do something in the game
    def step(self, action):
        # Action key - 0 = Space, 1 = Duck(down), 2 = No action (no op)
        action_map = {
            0:'space',
            1:'down',
            2:'no_op'
        }
        if action != 2:
            pydirectinput.press(action_map[action])
        
        # Checking whether the game is done
        done, done_cap = self.get_done()
        # Get the next observation
        new_observation = self.get_observation()
        # Reward - we get a point for every frame we are alive
        reward = 1
        # Info dictionary
        info = {} # needed for stablebaselines what it expects
        
        return new_observation, reward, done, info
    
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
    
    # Restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=-590, y=465)
        pydirectinput.press('space')
        return self.get_observation()
    
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        # Get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        #Grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, (100,83))
        # Add channels first
        channel = np.reshape(resized, (1,83,100))
        return channel
    
    # Get the done text using OCR
    def get_done(self):
        # Get done screen
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        # Valid done text
        done_strings = ['GAME', 'GAHE']
        
        # Apply OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res == 'GAME':
            done = True
        return done, done_cap
    
class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True
       
env = WebGame()


# Initialize environment and game
obs = env.reset()
done = False

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs'
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=1000)
model.learn(total_timesteps=5000, callback=callback)


# # Game loop
# while not done:
#     env.render()  # Render the game screen
#     action = env.action_space.sample()  # Sample random action
#     obs, reward, done, info = env.step(action)  # Take the step
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Graceful exit if 'q' is pressed
#         done = True

# Play 10 games
# for episode in range(10):
#     obs = env.reset()
#     done =False
#     total_reward = 0
    
#     while not done:
#         obs, reward, done, info = env.step(env.action_space.sample())
#         total_reward += reward
#     print(f'Total Reward for episode {episode} is {total_reward}')
    

for episode in range(10):
    obs = env.reset()
    done =False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        time.sleep(0.01)
        total_reward += reward
    print(f'Total Reward for episode {episode} is {total_reward}')
    time.sleep(2)