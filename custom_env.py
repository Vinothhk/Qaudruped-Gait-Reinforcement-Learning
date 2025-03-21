import gym
from gym import spaces
import mujoco.renderer
import numpy as np
import mujoco
import glfw
import mujoco_py

class QuadrupedEnv(gym.Env):
    def __init__(self, model_path):
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1] * self.model.nu),  # Normalized joint torque limits
            high=np.array([1] * self.model.nu),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv + 12,), dtype=np.float32
        )

        # Initialize rendering
        self.window = None
        self.context = None
        self.scene = None
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # Gait parameters
        self.time = 0
        self.gait_period = 0.5  # Time period for gait cycle
        self.last_foot_positions = self._get_foot_positions()

    def reset(self):
        # Reset the simulation to the initial state
        mujoco.mj_resetData(self.model, self.data)
        self.time = 0
        self.last_foot_positions = self._get_foot_positions()
        return self._get_obs()

    def step(self, action):
        # Apply control action
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation, reward, and done flag
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        info = {}

        # Increment time for gait cycle
        self.time += self.model.opt.timestep

        return obs, reward, done, info

    def _get_obs(self):
        # Observation includes:
        # - Joint positions (qpos)
        # - Joint velocities (qvel)
        # - Foot positions (relative to body)
        qpos = self.data.qpos
        qvel = self.data.qvel
        foot_positions = self._get_foot_positions()
        return np.concatenate([qpos, qvel, foot_positions])

    def _get_foot_positions(self):
        # Get the positions of all four feet relative to the body
        foot_names = ["FL", "FR", "HL", "HR"]
        foot_positions = []
        for foot_name in foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            foot_pos = self.data.geom_xpos[foot_id]
            body_pos = self.data.qpos[0:3]
            foot_positions.extend(foot_pos - body_pos)
        return np.array(foot_positions)

    def _get_reward(self):
        # Reward components:
        # 1. Stability: Penalize deviation from desired body orientation and height
        # 2. Gait efficiency: Reward smooth and periodic leg movements
        # 3. Forward velocity: Reward moving forward
        # 4. Energy efficiency: Penalize high joint torques

        # Stability
        body_orientation = self.data.qpos[3:7]  # Quaternion representing body orientation
        desired_orientation = np.array([1, 0, 0, 0])  # Upright orientation
        orientation_error = np.linalg.norm(body_orientation - desired_orientation)

        body_height = self.data.qpos[2]  # Z position of the body
        desired_height = 0.75  # Initial height
        height_error = abs(body_height - desired_height)

        stability_reward = -0.1 * orientation_error - 0.1 * height_error

        # Gait efficiency
        current_foot_positions = self._get_foot_positions()
        foot_movement = np.linalg.norm(current_foot_positions - self.last_foot_positions)
        self.last_foot_positions = current_foot_positions
        gait_reward = 0.1 * foot_movement

        # Forward velocity
        forward_velocity = self.data.qvel[0]  # X-axis velocity
        velocity_reward = 1.0 * forward_velocity

        # Energy efficiency
        joint_torques = np.abs(self.data.ctrl).sum()  # Sum of absolute joint torques
        energy_penalty = -0.01 * joint_torques

        # Total reward
        reward = stability_reward + gait_reward + velocity_reward + energy_penalty
        return reward

    def _get_done(self):
        # Termination conditions:
        # 1. Robot falls (body height too low)
        # 2. Simulation time exceeds a limit
        body_height = self.data.qpos[2]
        if body_height < 0.3:  # Robot falls
            return True
        if self.time > 10.0:  # Max simulation time
            return True
        return False

    # def render(self):
    #     if self.window is None:
    #         glfw.init()
    #         self.window = glfw.create_window(1200, 900, "Quadruped Robot", None, None)
    #         glfw.make_context_current(self.window)
    #         self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    #         # Initialize scene, camera, and options
    #         self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
    #         mujoco.mjv_defaultCamera(self.cam)
    #         mujoco.mjv_defaultOption(self.opt)

    #         # Set initial camera properties
    #         self.cam.distance = 2.5  # Move camera further back
    #         self.cam.elevation = -20  # Tilt downward
    #         self.cam.azimuth = 90  # Keep the camera behind the robot initially

    #     # Get robot position
    #     robot_position = self.data.qpos[:3]  # (x, y, z) position

    #     # Offset the camera to be behind the robot in the X direction
    #     camera_offset = np.array([-3.5, 0, 0.75])  # (backward, lateral, height)

    #     # Rotate the offset if the robot turns
    #     yaw_angle = np.arctan2(self.data.qvel[1], self.data.qvel[0])  # Get heading direction
    #     rotation_matrix = np.array([
    #         [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
    #         [np.sin(yaw_angle), np.cos(yaw_angle), 0],
    #         [0, 0, 1]
    #     ])
        
    #     camera_position = robot_position + rotation_matrix @ camera_offset  # Rotate offset around robot

    #     # Smoothly interpolate camera movement (prevents jitter)
    #     alpha = 0.1  # Adjust for smoothness (0 = no movement, 1 = instant movement)
    #     self.cam.lookat[:] = alpha * robot_position + (1 - alpha) * self.cam.lookat[:]
        
    #     # Set the new camera position and update the scene
    #     self.cam.distance = np.linalg.norm(camera_position - robot_position)  # Maintain consistent distance
    #     mujoco.mjv_updateScene(
    #         self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
    #     )

    #     # Render the scene
    #     viewport = mujoco.MjrRect(0, 0, 1200, 900)
    #     mujoco.mjr_render(viewport, self.scene, self.context)

    #     # Swap buffers and poll events
    #     glfw.swap_buffers(self.window)
    #     glfw.poll_events()

    #     # Close the window if ESC is pressed
    #     if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
    #         glfw.set_window_should_close(self.window, True)

    #     if glfw.window_should_close(self.window):
    #         glfw.terminate()
    #         self.window = None
    #         self.context = None
    #         self.scene = None  # Reset scene


    # def render(self):
    #     if self.window is None:
    #         glfw.init()
    #         self.window = glfw.create_window(1200, 900, "Quadruped Robot", None, None)
    #         glfw.make_context_current(self.window)
    #         self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    #         # Initialize scene, camera, and options
    #         self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
    #         mujoco.mjv_defaultCamera(self.cam)
    #         mujoco.mjv_defaultOption(self.opt)

    #         # Set initial camera position
    #         self.cam.lookat[:] = self.data.qpos[:3]  # Center on robot
    #         self.cam.distance = 2.0  # Move camera further back
    #         self.cam.elevation = -20  # Tilt slightly downward

    #     # Update camera position dynamically
    #     robot_position = self.data.qpos[:3]  # Get the robot's x, y, z position
    #     self.cam.lookat[:] = robot_position  # Keep the camera centered on the robot

    #     # Update the scene
    #     mujoco.mjv_updateScene(
    #         self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
    #     )

    #     # Render the scene
    #     viewport = mujoco.MjrRect(0, 0, 1200, 900)
    #     mujoco.mjr_render(viewport, self.scene, self.context)

    #     # Swap buffers and poll events
    #     glfw.swap_buffers(self.window)
    #     glfw.poll_events()

    #     # Close the window if ESC is pressed
    #     if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
    #         glfw.set_window_should_close(self.window, True)

    #     if glfw.window_should_close(self.window):
    #         glfw.terminate()
    #         self.window = None
    #         self.context = None
    #         self.scene = None  # Reset scene
    def render(self):
        if self.window is None:
            glfw.init()
            self.window = glfw.create_window(1200, 900, "Quadruped Robot", None, None)
            glfw.make_context_current(self.window)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            # Initialize scene, camera, and options
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            mujoco.mjv_defaultCamera(self.cam)
            mujoco.mjv_defaultOption(self.opt)

            # Set initial camera properties
            self.cam.distance = 2.5  # Set camera at a good distance
            self.cam.elevation = -20  # Slight downward tilt
            self.cam.azimuth = 90  # Default behind view

        # Get the robot’s position
        robot_position = self.data.qpos[:3]  # (x, y, z) coordinates

        # Camera offset to stay behind and above the robot
        camera_offset = np.array([-3.5, 0, 1.0])  # (back, side, height)

        # Get robot’s yaw (heading direction)
        yaw_angle = np.arctan2(self.data.qvel[1], self.data.qvel[0])  

        # Create a rotation matrix to align the camera with robot movement
        rotation_matrix = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])
        
        # Rotate the camera offset relative to the robot's direction
        camera_position = robot_position + rotation_matrix @ camera_offset  

        # Smoothly update camera position (avoiding sudden jumps)
        alpha = 0.2  # Adjust smoothness (0 = no motion, 1 = instant)
        self.cam.lookat[:] = alpha * robot_position + (1 - alpha) * self.cam.lookat[:]
        
        # Set the camera position
        self.cam.distance = np.linalg.norm(camera_position - robot_position)

        # Update the scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )

        # Render the scene
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjr_render(viewport, self.scene, self.context)

        # Swap buffers and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        # Close the window if ESC is pressed
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

        if glfw.window_should_close(self.window):
            glfw.terminate()
            self.window = None
            self.context = None
            self.scene = None  # Reset scene
