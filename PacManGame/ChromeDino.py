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
from collections import deque
class ChromeDino(Env):
    def __init__(self):
        
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(6,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.frame_stack = deque(maxlen=6)
        
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
        
        reward = 1
        
        # Checking whether the game is done
        done = self.get_done()
        
        # Get the next observation
        new_frame = self.get_observation()
        self.frame_stack.append(new_frame)
        stacked_observation = self.get_stacked_observation()
        # Reward - we get a point for every frame we are alive

        
        return stacked_observation, reward, done, False, {}
    
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
    
    # Restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=-590, y=465)
        pydirectinput.press('space')
        # Reset frame stack
        self.frame_stack.clear() # Delete all items from Deque
        # Update deque with reset state
        for _ in range(6):
            initial_frame = self.get_observation()
            self.frame_stack.append(initial_frame)
            
        return self.get_stacked_observation(), {}
    
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
    def get_stacked_observation(self):
        # Stack the frames in the deque and convert to the required shape
        return np.concatenate(list(self.frame_stack), axis=0)
    
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
        return done