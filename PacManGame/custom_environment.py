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
from gymnasium.utils.env_checker import check_env  # Import the environment checker
from collections import deque
import math


class PacMan(Env):
    def __init__(self):
        super().__init__()
        
        # Define spaces
        self.observation_space = Box(low=0, high=255, shape=(6,50,80), dtype=np.uint8)
        self.action_space = Discrete(4) # number of possible actions
        
        # Define capture locations
        self.cap = mss()
        self.game_location = {'top':50, 'left':-2280, 'width':1400, 'height':1300}# defines game viewing location
        self.lives_location = {'top':1070, 'left':-902, 'width':600, 'height':200} # defines lives location
        self.frame_stack = deque(maxlen=6) # stack frames to provide a sense of motion
        #self.score_location = {'top':380, 'left':-920, 'width':600, 'height':80} # defines score location
        #self.done_location = {'top':508, 'left':-1810, 'width':450, 'height':80} 
            
        # Define lives
        self.previous_lives = 2
        self.current_lives = self.previous_lives
        self.previous_score = 0
        self.time_alive = 0
        self.last_life = 2
        self.survival_reward_factor = 0.1
        
        # Define pellet count
        self.pellet_address = 0x7268 # ROM memory address
        self.file_path = "pellet_count.txt" # file to store value
        self.previous_pellet_count = self.read_pellet_count_from_file()
        

        # Define templates for tracking
        self.ghost_template = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\ghost_template.png', 0)
        self.ghost_template2 = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\ghost_template3.png', 0)
        self.ghost_template3 = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\ghost_template4.png', 0)
        self.pacman_life_template = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\pacman_life_icon.png', 0)
        self.pacman_template_left = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\pacman_template_left.png', 0)
        self.pacman_template_right = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\pacman_template_right.png', 0)
        self.pacman_template_up = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\pacman_template_up.png', 0)
        self.pacman_template_down = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\pacman_template_down.png', 0)
        self.pacman_template_closed = cv.imread('C:\\Users\\John Wesley\\Docs\\PacMan\\PacManGame\\Images\\pacman_template_closed.png', 0)
        
    # Observation of the state of the environment
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
        # Stack the frames in the deque and convert to the required shape
        return np.concatenate(list(self.frame_stack), axis=0)
    
    # Get number of lives left
    def get_lives(self):   
        # Capture the area where the lives are displayed
        lives_cap = np.array(self.cap.grab(self.lives_location))[:,:,:3]
        # Convert to grayscale
        lives_gray = cv.cvtColor(lives_cap, cv.COLOR_BGR2GRAY)
        # Perform template matching
        result = cv.matchTemplate(lives_gray, self.pacman_life_template, cv.TM_CCORR_NORMED)
        locations = np.where(result >= 0.8) # find areas that have values at or above threshold value
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
    
    # Get pellet count
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
        pydirectinput.click(x=-890, y=374) # Select game window
        pydirectinput.press('f1') # Start state 1 save
        # Reset pellet count
        self.previous_pellet_count = self.read_pellet_count_from_file()
        # Reset frame stack
        self.frame_stack.clear() # Delete all items from Deque
        # Update deque with reset state
        for _ in range(6):
            initial_frame = self.get_observation()
            self.frame_stack.append(initial_frame)
            
        return self.get_stacked_observation(), {}
    
    # Rendering method to see what the computer sees
    def render(self):
        frame = self.render_positions()
        cv.imshow('Game', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            self.close()
            
    # Closes rendering window        
    def close(self):
        cv.destroyAllWindows()
    
    # Find character locations on screen            
    def get_character_positions(self):
        # Capture the area where the lives are displayed
        screen_capture = np.array(self.cap.grab(self.game_location))[:,:,:3]
        cv.imwrite('game_capture.png', screen_capture)
        # Convert to grayscale
        gray_screen = cv.cvtColor(screen_capture, cv.COLOR_BGR2GRAY)
        
        # Match the templates to find Pac-Man and Ghosts
        result_left = cv.matchTemplate(gray_screen, self.pacman_template_left, cv.TM_CCOEFF_NORMED)
        result_right = cv.matchTemplate(gray_screen, self.pacman_template_right, cv.TM_CCOEFF_NORMED)
        result_up = cv.matchTemplate(gray_screen, self.pacman_template_up, cv.TM_CCOEFF_NORMED)
        result_down = cv.matchTemplate(gray_screen, self.pacman_template_down, cv.TM_CCOEFF_NORMED)
        result_closed = cv.matchTemplate(gray_screen, self.pacman_template_closed, cv.TM_CCOEFF_NORMED)
        result_ghost = cv.matchTemplate(gray_screen, self.ghost_template, cv.TM_CCOEFF_NORMED)
        result_ghost2 = cv.matchTemplate(gray_screen, self.ghost_template2, cv.TM_CCOEFF_NORMED)
        result_ghost3 = cv.matchTemplate(gray_screen, self.ghost_template3, cv.TM_CCOEFF_NORMED)
        
        # Locate pacman
        pacman_threshold = 0.6 # Adjust this value based on testing
        locations_left = np.where(result_left >= pacman_threshold)
        locations_right = np.where(result_right >= pacman_threshold)
        locations_up = np.where(result_up >= pacman_threshold)
        locations_down = np.where(result_down >= pacman_threshold)
        locations_closed = np.where(result_closed >= pacman_threshold)
        
        # Locate ghosts
        ghost_threshold = 0.5
        location_ghost = np.where(result_ghost >= ghost_threshold)
        location_ghost2 = np.where(result_ghost2 >= ghost_threshold)
        location_ghost3 = np.where(result_ghost3 >= ghost_threshold)
        
        # Pack locations
        pacman_combined_locations = list(zip(*locations_left[::-1])) + list(zip(*locations_right[::-1])) + list(zip(*locations_up[::-1])) + list(zip(*locations_down[::-1])) + list(zip(*locations_closed[::-1]))
        ghost_position = list(zip(*location_ghost[::-1])) + list(zip(*location_ghost2[::-1]))  + list(zip(*location_ghost3[::-1]))

        return ghost_position, pacman_combined_locations, screen_capture
 # Find ghosts by color
    def find_ghosts_by_color(self, image):
        # Convert image to HSV color space
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # Define range of colors to search
        ghost_colors = {
            'blinky': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},  # Red
            'pinky': {'lower': np.array([160, 100, 100]), 'upper': np.array([170, 255, 255])}, # Pink
            'inky': {'lower': np.array([85, 100, 100]), 'upper': np.array([95, 255, 255])},   # Cyan
            'clyde': {'lower': np.array([15, 100, 100]), 'upper': np.array([25, 255, 255])}
        }
        
        ghost_positions = {}
        total_ghosts = 0
        # Iterate over each ghost color and find position
        for ghost, color_range in ghost_colors.items():
            # Create mask for color
            mask = cv.inRange(hsv_image, color_range['lower'], color_range['upper'])
            # Find contours
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            positions = []
            # Iterate over each contour
            for contour in contours:
                # Calculate bounding box
                x, y, w, h = cv.boundingRect(contour)
                positions.append((x+w//2, y+h//2)) # Center of the ghost
            ghost_positions[ghost] = positions
            #print(f"{len(positions)} {ghost}(s) found")
            
            total_ghosts += len(positions)

        #print(f"{total_ghosts} total ghost(s) found")
        return ghost_positions
       
    # Method to see character detection    
    def render_positions(self):
        # Get character positions
        ghost_position, pacman_combined_locations, screen_capture = self.get_character_positions()

        screen_capture = np.ascontiguousarray(screen_capture) # convert captured image to OpenCV compatability
        
        # Draw rectangles around matched Pac-Man locations using OpenCV
        for loc in pacman_combined_locations:
            top_left = loc
            bottom_right = (top_left[0] + self.pacman_template_right.shape[1], top_left[1] + self.pacman_template_right.shape[0])
            # Create a rectangle patch and add it to the plot
            cv.rectangle(screen_capture, top_left, bottom_right, (255, 0, 0), 2)
            
        # Draw rectangles around matched Ghost locations using OpenCV
        for loc in ghost_position:
            top_left = loc
            bottom_right = (top_left[0] + self.ghost_template.shape[1], top_left[1] + self.ghost_template.shape[0])
            # Create a rectangle patch and add it to the plot
            cv.rectangle(screen_capture, top_left, bottom_right, (0, 0, 255), 2)

        # cv.imshow('Test Render positions', screen_capture)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return screen_capture
    
    def calculate_distance (self, pacman_pos, ghost_pos):
        # Unpack positions
        pacman_x, pacman_y = pacman_pos
        ghost_x, ghost_y = ghost_pos
        return math.sqrt((ghost_x - pacman_x) ** 2 + (ghost_y - pacman_y) ** 2)
    
    # Calculate reward for eating pellets
    def get_pellet_reward(self, current_pellet_count):
        if current_pellet_count < self.previous_pellet_count:
            reward = 30 
            self.previous_pellet_count = current_pellet_count
        else:
            reward = 0    
        return reward
    
    def ghost_avoidance_reward(self, screen_image):
        ghost_positions = self.find_ghosts_by_color(screen_image)
        _, pacman_combined_locations, _ = self.get_character_positions()
        if pacman_combined_locations:
            pacman_pos = pacman_combined_locations[0]
        else:
            pacman_pos = (0, 0)
        safe_distance = 260
        avoidance_reward = 0
        
        for ghost, posiitons in ghost_positions.items():
           for ghost_pos in posiitons:
               distance = self.calculate_distance(pacman_pos, ghost_pos)
               if distance > safe_distance:
                   avoidance_reward += 20 / len(ghost_positions)
               else:
                   avoidance_reward -= 5
        return avoidance_reward
    
        
    # def ghost_avoidance_reward(self):
    #     ghost_positions, pacman_combined_locations, _ = self.get_character_positions()
    #     if pacman_combined_locations:
    #         pacman_pos = pacman_combined_locations[0]
    #     else:
    #         pacman_pos = (0, 0)
    #     safe_distance = 260
    #     avoidance_reward = 0
        
    #     for ghost_index, ghost_pos in enumerate(ghost_positions):
    #         distance = self.calculate_distance(pacman_pos, ghost_pos)
    #         #print(f"Ghost {ghost_index + 1} Position: {ghost_pos}, Distance: {distance}")
    #         if distance > safe_distance:
    #             avoidance_reward += (distance - safe_distance)
    #         else:
    #             avoidance_reward -= (safe_distance - distance) * 2
    #     return avoidance_reward
    
    # Method that is called to do something in the game

    def step(self, action):
        action_map = {
            0: 'left',   # Move Left
            1: 'right',  # Move Right
            2: 'up',     # Move Up
            3: 'down',   # Move Down
        }
        
        pydirectinput.press(action_map[action])
        
        # Reward for eating pellets 
        # current_pellet_count = self.read_pellet_count_from_file()
        # pellet_reward = self.get_pellet_reward(current_pellet_count)
        # Reward for avoiding ghosts
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        avoidance_reward = self.ghost_avoidance_reward(raw)
        # Bonus reward for staying alive      
        current_lives = self.get_lives()
        # if current_lives < self.last_life:
        #     self.time_alive = 0
        #     self.last_life = current_lives
        # self.time_alive += 1
        # # survival_reward = self.survival_reward_factor * (1.1 ** self.time_alive)
        # survival_reward = self.time_alive * 1.01 
        # Penalize only when a life is lost (and only once per life loss)
        life_penalty = 0
        if current_lives < self.previous_lives:
            life_penalty -= 40
            self.previous_lives = current_lives # update previous lives 
           
        reward = avoidance_reward 
        
        done = self.get_done()
        
        # Get the next observation
        new_frame = self.get_observation()
        self.frame_stack.append(new_frame)
        stacked_observation = self.get_stacked_observation()
        
        return stacked_observation, reward, done, False, {}
    


# def main():
#     env = PacMan()
#     obs, _ = env.reset()
#     done = False
#     rewards = []
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, _, _ = env.step(action)
#         rewards.append(reward)
#         if done:
#             print(f"Total reward for episode is {sum(rewards)}")
#             rewards = []

# if __name__ == "__main__":
#     while True:
#         main()