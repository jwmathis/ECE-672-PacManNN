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
from selenium import webdriver
class DinoGame(Env):
    def __init__(self):
        
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(6,50,200), dtype=np.uint8)
        self.action_space = Discrete(3) # number of possible actions
        self.cap = mss()
        self.game_location = {'top':690, 'left':-2270, 'width':800, 'height':300} # defines viewing area
        self.done_location = {'top':550, 'left':-1730, 'width':900, 'height':120} # defines 'GAME OVER' location    
        self.frame_stack = deque(maxlen=6) # stack frames to provide a sense of motion; DQN benefits from this
        
    
    # observation of the state of the environment
    def get_observation(self):
        # Get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        #Grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, (200,50))
        # Add channels first
        channel = np.reshape(resized, (1,50,200))
        return channel
    
    def get_stacked_observation(self):
        # stack the frames in the deque and convert to the required shape
        return np.concatenate(list(self.frame_stack), axis=0)
    
    # Get the done text using OCR
    def get_done(self):
        # Get done screen
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        
        # Apply OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res == 'GAME': # NOTE: doesn't recognize 'OVER'
            done = True
        return done, done_cap, res
    
    # Resets the environment to its initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(0.3)
        pydirectinput.click(x=-1385, y=527)
        pydirectinput.press('space')
        # driver = webdriver.Chrome()
        # driver.get("chrome://dino")
        # driver.execute_script("Runner.instance_.setSpeed(15);")
        # Reset the frame stack
        self.frame_stack.clear()
        for _ in range(6):
            initial_frame = self.get_observation()
            self.frame_stack.append(initial_frame)
        return self.get_stacked_observation(), {}
    
    
    # Detect objects
    def is_obstacle_nearby(self):
        # Capture current frame
        current_frame = self.get_observation()
        
        # Convert to grayscale and resize if necessary
        gray_frame = current_frame[0, :, :]
        
        # Define a threshold for detecting obstacles
        obstacle_threshold = 100
        obstacle_detected = np.sum(gray_frame < obstacle_threshold) > 200
        return obstacle_detected
    
    # method to take an action as an input and applies it to the environment
    def step(self, action):
        #               Jump        Duck      No Action
        action_map = {0:'space', 1:'down', 2:'no_op'}
        total_reward = 0 
        
        # Check if obstacle is nearby before performing action
        obstacle_nearby = self.is_obstacle_nearby()
        
        # Perform the action
        if action != 2:
            pydirectinput.press(action_map[action])
            
        # Checking whether the game is done
        done = self.get_done()

        # Reward - we get a point for every frame we are alive
        reward = 1
        if done:
            reward = -30
        total_reward += reward

        if action == 0:
            if obstacle_nearby:
                total_reward += 10
            if not done:
                total_reward += 40
        elif action == 1:
            if obstacle_nearby:
                total_reward -= 2
        elif action == 2:
            if obstacle_nearby:
                total_reward -= 1
                
        # Get the latest frame
        new_frame = self.get_observation()
        # Update frame stack
        self.frame_stack.append(new_frame)
        # Get stacked observation for the next state
        stacked_observation = self.get_stacked_observation()
        
        return stacked_observation, total_reward, done, False, {}
    

    