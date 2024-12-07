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

* Python: Version 3.8 or higher
* Gymnasium: `pip install Gymnasium`
* Gymnasium-Retro: `pip install gymnasium[retro]`
* PyTorch
* NumPy
* OpenCV
* Matplotlib

You can set up a virtual environment to manage these dependencies:
```
# Create a new virtual environment
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```
A requirements.txt file is provided in the repository for easy installation.

### Installing
Follow these steps to download, set up, and configure the project:

1. Clone the repository

Download the project files from GitHub:
```
git clone https://github.com/your-username/your-repo-name.
cd your-repo-name
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



### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info


## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)