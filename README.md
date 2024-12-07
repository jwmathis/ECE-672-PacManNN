# Deep Q-Network for Chrome Dino and Pac-Man

This project explores the application of Deep Q-Network for pixelated games.

## Description

A custom environment was built for Chrome Dino and Pac-Man using OpenAI's Gym Retro framework, to enable
the agent to interact with the game and learn optimal strategies through reinforcement learning. This project
includes: 
* A Deep Q-Network (DQN) for decision-making.
* Custom reward systems tailored for each game.
* Preprocessing techniques, including frame stacking, grayscale conversion, and frame resizing.

The primary goal of this project is to develop an understanding and knowledge of DQNs, reinforcement learning, and to familiarization
with a framework for methods across different gaming environments.

## Getting Started

### Dependencies
To run this project, ensure you have the following dependencies installed:
Note: This project was created on Windows 11.
* Python: Version 3.8
* Gymnasium: Version 0.29.1
* PyTorch: Version 2.4.1+cu124
* TorchAudio: Version 0.19.1+cu124
* NumPy: Version 1.24.1
* OpenCV: Version 4.10
* Matplotlib: Version 3.7
* MSS: Version 9.0.2
* PyTesseract: Version 0.3.13
* PyDirectInput: Version 1.0.4
* Selenium: Version 4.25.0
* yaml: version 0.2.5
You can set up a virtual environment to manage these dependencies:
```
# Create a new virtual environment
python3.8 -m venv myenv
# Activate the environment
myenv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```
A requirements.txt file is provided in the repository for easy installation.

### Installing
Follow these steps to download, set up, and configure the project:

1. Clone the repository

Download the project files from GitHub:
```
git clone https://github.com/jwmathis/ECE672_ROM_NeuralNetwork.git
cd ECE672_ROM_NeuralNetwork
```

2. Install Dependencies

Ensure you have Python 3.8 or higher installed. Create a virtual environment and install the required packages:
```
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

3. Obtain Game ROMs
This project requires the Pac-Man ROM. Note that ROMs are copyrighted, so you must legally own the game to use 
its ROM. 

This project requires Chrome Dino. Chrome Dino can be accessed by navigating to Chrome://Dino in the Google Chrome Browser.

4. Configure Environment files
