# Import Dependencies
import torch # PyToch library for building and training neural networks
from torch import nn
from torch.optim import AdamW
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



class PacMan(Env):
    def __init__(self):
        super().__init__()
        # Define spaces
        self.observation_space = Box(low=0, high=255, shape=(6,50,80), dtype=np.uint8)
        self.action_space = Discrete(5) # number of possible actions
        
        self.previous_lives = 2
        self.current_lives = self.previous_lives
        self.previous_score = 0
        
        self.pellet_address = 0x7268
        self.file_path = "pellet_count.txt"
        self.previous_pellet_count = self.read_pellet_count_from_file()
        
        # Define capture locations
        self.cap = mss()
        self.game_location = {'top':50, 'left':-2280, 'width':1400, 'height':1300}# defines game viewing location
        self.lives_location = {'top':1070, 'left':-902, 'width':600, 'height':200} # defines lives location
        self.frame_stack = deque(maxlen=6) # stack frames to provide a sense of motion
        #self.score_location = {'top':380, 'left':-920, 'width':600, 'height':80} # defines score location
        #self.done_location = {'top':508, 'left':-1810, 'width':450, 'height':80}     

        # Define templates for tracking
        self.ghost_template = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\ghost_template.png', 0)
        self.ghost_template2 = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\ghost_template3.png', 0)
        self.ghost_template3 = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\ghost_template4.png', 0)
        self.pacman_life_template = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\pacman_life_icon.png', 0)
        self.pacman_template_left = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\pacman_template_left.png', 0)
        self.pacman_template_right = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\pacman_template_right.png', 0)
        self.pacman_template_up = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\pacman_template_up.png', 0)
        self.pacman_template_down = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\pacman_template_down.png', 0)
        self.pacman_template_closed = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\pacman_template_closed.png', 0)
        
    # observation of the state of the environment
    def get_observation(self):
        # Get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        # Grayscale
        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        # Resize
        resized = cv.resize(gray, (80,50))
        # Add channels first
        channel = np.reshape(resized, (1,50,80))
        return channel
    
    def get_stacked_observation(self):
        # stack the frames in the deque and convert to the required shape
        return np.concatenate(list(self.frame_stack), axis=0)
    
    # get number of lives left
    def get_lives(self):   
        # Capture the area where the lives are displayed
        lives_cap = np.array(self.cap.grab(self.lives_location))[:,:,:3]
        # Convert to grayscale
        lives_gray = cv.cvtColor(lives_cap, cv.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv.matchTemplate(lives_gray, self.pacman_life_template, cv.TM_CCORR_NORMED)
        threshold = 0.8
        locations = np.where(result >= threshold)
        
        lives_value = len(list(zip(*locations[::-1])))
        
        # Determine number of lives
        if lives_value == 684:
            num_lives = 2
        elif lives_value == 344:
            num_lives = 1
        else:
            num_lives = 0
            
        return num_lives
    
    # Get game over
    def get_done(self):
        # Get the number of lives left 
        num_lives = self.get_lives()
        return num_lives == 0 # return bool
    
    def read_pellet_count_from_file(self):
        try:
            with open(self.file_path, "r") as file:
                return int(file.read().strip())
        except (FileNotFoundError, ValueError):
            return 0
        
    # Resets the environment to its initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # restart the game
        pydirectinput.click(x=-890, y=374) # select game window
        pydirectinput.press('f1') # Start state 1 save
        
        # reset pellet count
        self.previous_pellet_count = self.read_pellet_count_from_file()
        
        # reset frame stack
        self.frame_stack.clear()
        for _ in range(6):
            initial_frame = self.get_observation()
            self.frame_stack.append(initial_frame)
            
        return self.get_stacked_observation(), {}
    def render(self):
        frame = self.render_positions()
        
        cv.imshow('Game', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            self.close()
            
    def close(self):
        cv.destroyAllWindows()
               
    def get_character_positions(self):
        # Capture the area where the lives are displayed
        screen_capture = np.array(self.cap.grab(self.game_location))[:,:,:3]
        cv.imwrite('game_capture.png', screen_capture)
        # Convert to grayscale
        gray_screen = cv.cvtColor(screen_capture, cv.COLOR_BGR2GRAY)
        # Match the templates to find Pac-Man
        result_left = cv.matchTemplate(gray_screen, self.pacman_template_left, cv.TM_CCOEFF_NORMED)
        result_right = cv.matchTemplate(gray_screen, self.pacman_template_right, cv.TM_CCOEFF_NORMED)
        result_up = cv.matchTemplate(gray_screen, self.pacman_template_up, cv.TM_CCOEFF_NORMED)
        result_down = cv.matchTemplate(gray_screen, self.pacman_template_down, cv.TM_CCOEFF_NORMED)
        result_closed = cv.matchTemplate(gray_screen, self.pacman_template_closed, cv.TM_CCOEFF_NORMED)
        result_ghost = cv.matchTemplate(gray_screen, self.ghost_template, cv.TM_CCOEFF_NORMED)
        result_ghost2 = cv.matchTemplate(gray_screen, self.ghost_template2, cv.TM_CCOEFF_NORMED)
        result_ghost3 = cv.matchTemplate(gray_screen, self.ghost_template3, cv.TM_CCOEFF_NORMED)
        threshold = 0.6 # Adjust this value based on testing
        locations_left = np.where(result_left >= threshold)
        locations_right = np.where(result_right >= threshold)
        locations_up = np.where(result_up >= threshold)
        locations_down = np.where(result_down >= threshold)
        locations_closed = np.where(result_closed >= threshold)
        location_ghost = np.where(result_ghost >= 0.5)
        location_ghost2 = np.where(result_ghost2 >= 0.5)
        location_ghost3 = np.where(result_ghost3 >= 0.5)
        pacman_combined_locations = list(zip(*locations_left[::-1])) + list(zip(*locations_right[::-1])) + list(zip(*locations_up[::-1])) + list(zip(*locations_down[::-1])) + list(zip(*locations_closed[::-1]))
        ghost_position = list(zip(*location_ghost[::-1])) + list(zip(*location_ghost2[::-1]))  + list(zip(*location_ghost3[::-1]))

        return ghost_position, pacman_combined_locations, screen_capture
        
    def render_positions(self):
        ghost_position, pacman_combined_locations, screen_capture = self.get_character_positions()

        screen_capture = np.ascontiguousarray(screen_capture) # convert captured image to OpenCV compatability
        
        # Draw rectangles around matched locations using Matplotlib patches
        for loc in pacman_combined_locations:
            top_left = loc
            bottom_right = (top_left[0] + self.pacman_template_right.shape[1], top_left[1] + self.pacman_template_right.shape[0])
            # Create a rectangle patch and add it to the plot
            cv.rectangle(screen_capture, top_left, bottom_right, (255, 0, 0), 2)

        for loc in ghost_position:
            top_left = loc
            bottom_right = (top_left[0] + self.ghost_template.shape[1], top_left[1] + self.ghost_template.shape[0])
            # Create a rectangle patch and add it to the plot
            cv.rectangle(screen_capture, top_left, bottom_right, (0, 0, 255), 2)

        # cv.imshow('Test Render positions', screen_capture)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return screen_capture
    
    def calculate_distance (self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + ( pos1[1] - pos2[1])**2)
    
    # Reward for eating pellets
    def get_pellet_reward(self, current_pellet_count):
        if current_pellet_count < self.previous_pellet_count:
            reward = 30 
            self.previous_pellet_count = current_pellet_count
        else:
            reward = 0    
        return reward
    
    
    # Action that is called to do something in the game
    def step(self, action):
        action_map = {
            0: 'left',   # Move Left
            1: 'right',  # Move Right
            2: 'up',     # Move Up
            3: 'down',   # Move Down
            4: 'no_op'   # No operation (do nothing)
        }
        
        if action != 4:
            pydirectinput.press(action_map[action])
            
        current_pellet_count = self.read_pellet_count_from_file()
        pellet_reward = self.get_pellet_reward(current_pellet_count)
        
        # ghost_positions, pacman_positions, _ = self.get_character_positions()
        
        # if pacman_positions:
        #     pacman_pos = pacman_positions[0]
        # else:
        #     pacman_pos = (0, 0) # Default position if not detected
            
        # if ghost_positions:
        #     ghost_positions = ghost_positions[0]
        # else:
        #     ghost_positions = (0, 0)
            
        # ghost_penalty = 0
        # threshold_distance = 50
        # for ghost_pos in ghost_positions:
        #     distance = self.calculate_distance(pacman_pos, ghost_pos)
        #     if distance < threshold_distance:
        #         ghost_penalty -= 10
                
        current_lives = self.get_lives()
        life_penalty = 0
        # Penalize only when a life is lost (and only once per life loss)
        if current_lives < self.previous_lives:
            life_penalty -= 50
            self.previous_lives = current_lives # update previous lives 
            
        reward = pellet_reward + life_penalty
        
        # Penalize heavily if all lives are lost
        done = self.get_done()
        # end_game_penalty = 0
        # if done:
        #     end_game_penalty -= 500
        # else: 
        #     end_game_penalty -= 0

        # Get the next observation
        new_frame = self.get_observation()
        self.frame_stack.append(new_frame)
        stacked_observation = self.get_stacked_observation()
        
        return stacked_observation, reward, done, False, {}
    
    