# Import dependencies
from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env  # Change this import
from gymnasium.spaces import Box, Discrete  # Change this import
from gymnasium.utils import env_checker  # Import the environment checker

class PacMan(Env):
    def __init__(self):
        
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(1,1000,1030), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.previous_lives = 2
        self.cap = mss()
        self.game_location = {'top':50, 'left':-2280, 'width':2000, 'height':1300}
        self.done_location = {'top':520, 'left':-1800, 'width':450, 'height':70} 
        #self.done_location = {'top':508, 'left':-1810, 'width':450, 'height':80}     
        
             
    # Action that is called to do something in the game
    def step(self, action):
        
        # Action key - 0 = Move Left, 1 = Move Right, 2 = Move Up, 3 = Move Down, 4 = No op
        action_map = {
            0: 'left',   # Move Left
            1: 'right',  # Move Right
            2: 'up',     # Move Up
            3: 'down',   # Move Down
            4: 'no_op'   # No operation (do nothing)
        }
        
        if action != 4:
            pydirectinput.press(action_map[action])
        
        # Checking whether the game is done
        done = self.get_done()
         
        # Reward - we get a point for every frame we are alive
        reward = 1 if not done else -100

        # Determine if the game is over
        terminated = done  # Whether the game is successfully finished
        
        truncated = False  # Add logic here if you want to end the episode early
        
        # Get the next observation
        new_observation = self.get_observation()

        # Info dictionary
        info = {} # needed for stablebaselines what it expects
        
        return new_observation, reward, terminated, truncated, info

    
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
    
    # Restart the game
    def reset(self):
        pydirectinput.click(x=-890, y=374)
        time.sleep(0.1)
        pydirectinput.press('space')
        time.sleep(0.2)
        pydirectinput.press('enter')
        time.sleep(0.2)
        return self.get_observation()
    
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        # Get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        #Grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, (1030,1000))
        # Add channels first
        channel = np.reshape(resized, (1,1000,1030))
        return channel
    
    def get_lives(self):   
        # Capture the area where the lives are displayed
        lives_location = {'top':1070, 'left':-902, 'width':600, 'height':200}
        lives_cap = np.array(self.cap.grab(lives_location))[:,:,:3]
        # Convert to grayscale
        lives_gray = cv2.cvtColor(lives_cap, cv2.COLOR_BGR2GRAY)
        
        # Load pacman life icon template
        pacman_life_template = cv2.imread('pacman_life_icon.png', 0)
        
        # Perform template matching
        result = cv2.matchTemplate(lives_gray, pacman_life_template, cv2.TM_CCORR_NORMED)
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
    
    # Get the done text using OCR
    def get_done(self):
        
        # Get the number of lives left 
        num_lives = self.get_lives()
        if num_lives == 0: 
            done = True
        # # Get done screen
        # done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        # # Valid done text
        # done_strings = ['GAME', 'GAME OVER', 'GAMEOVER']
        
        # # Apply OCR
        # done = False
        # res = pytesseract.image_to_string(done_cap)[:4]
        # if res == 'PLAY':
        #     done = True
        return done

env = PacMan()
  
# Game loop
obs = env.reset()
done = False

while not done:
    env.render()  # Render the game screen
    action = env.action_space.sample()  # Sample random action
    obs, reward, done, truncated, info = env.step(action)  # Take the step
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Graceful exit if 'q' is pressed
        done = True