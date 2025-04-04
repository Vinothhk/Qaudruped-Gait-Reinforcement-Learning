# Qaudruped-Gait-Reinforcement-Learning

This project focuses on training and testing reinforcement learning (RL) models for a quadruped robot using the MuJoCo physics engine. The robot is designed to learn stable and efficient locomotion behaviors, such as walking, trotting, and running.

## Project Structure
```
quadruped_rl/
├── logs/                  # Training and evaluation logs (ignored by Git)
├── models/                # Saved RL models (ignored by Git)
├── mujoco-robot-env/      # Custom MuJoCo environment for the quadruped
├── quadruped_rl/          # Main project directory
│   ├── spot_env.py        # Environment implementation
│   ├── train_v2.py        # Training script
│   ├── test_v2.py         # Testing script
│   └── .gitignore         # Git ignore file
└── README.md              # Project documentation
```

## Requirements

- Python 3.10+
- MuJoCo physics engine
- Required Python libraries (install via `requirements.txt`):

  ```bash
  pip install -r requirements.txt
  ```

## Usage

### Training
To train the quadruped robot using PPO:
```bash
python3 spot_train.py
```

### Testing
To test a trained model:
```bash
python3 test_v2.py
```

## Key Features

- **Custom MuJoCo Environment**: Implements a quadruped robot with realistic physics.
- **Reinforcement Learning**: Uses Stable-Baselines3 for training RL models.
- **Reward Functions**: Includes stability, velocity, energy efficiency, and gait synchronization rewards.
- **Interactive Testing**: Allows real-time testing with rendering.

## File Descriptions

- `spot_env.py`: Defines the custom environment for the quadruped robot.
- `train_v2.py`: Script for training the RL model.
- `test_v2.py`: Script for testing the trained model.
- `.gitignore`: Specifies files and directories to ignore in version control.

## Displaying Simulation Video

You can visualize the robot's performance in TRAINING by watching the simulation video stored in the `media` folder.

### Example Video

Below is a sample video of the quadruped robot's gait simulation:

![Simulation Video](media/training.gif)


## Future Work

- Add more complex reward functions for advanced gaits.
- Implement domain randomization for robustness.
- Extend to uneven terrain simulations.


<!-- ## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```

Save this content as `README.md` in your project directory. Let me know if you need further customization! -->
