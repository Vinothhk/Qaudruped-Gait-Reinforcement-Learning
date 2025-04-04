import gym
from gym import spaces
import numpy as np
import mujoco
import glfw

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
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv + 12 + 4,), dtype=np.float32
        # )
        observation_size = self.model.nq + self.model.nv + 20  # nq + nv + 4 (quaternion) + 12 (foot positions) + 4 (contact states)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32
        )

        # Initialize rendering
        self.window = None
        self.context = None
        self.scene = None
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # Gait parameters
        self.time = 0
        self.gait_period = 0.5  # Time period for gait 
        
        # Foot names for gait control
        self.foot_names = ["FL", "FR", "HL", "HR"]

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

        # Apply gait pattern
        self._apply_gait()

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
        # - Body orientation (quaternion)
        # - Foot positions (relative to body)
        # - Contact states (binary)
        qpos = self.data.qpos
        qvel = self.data.qvel
        body_orientation = self.data.qpos[3:7]  # Quaternion
        foot_positions = self._get_foot_positions()
        contact_states = self._get_contact_states()
        return np.concatenate([qpos, qvel, body_orientation, foot_positions, contact_states])

    def _get_foot_positions(self):
        # Get the positions of all four feet relative to the body
        foot_positions = []
        for foot_name in self.foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            foot_pos = self.data.geom_xpos[foot_id]
            body_pos = self.data.qpos[0:3]
            foot_positions.extend(foot_pos - body_pos)
        return np.array(foot_positions)

    def _get_contact_states(self):
        # Check if each foot is in contact with the ground
        contact_states = []
        for foot_name in self.foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if foot_id == -1:
                raise ValueError(f"Foot '{foot_name}' not found in the model.")
            
            # Check if the foot is in contact
            in_contact = False
            for contact in self.data.contact:
                if contact.geom1 == foot_id or contact.geom2 == foot_id:
                    in_contact = True
                    break
            contact_states.append(in_contact)
        return np.array(contact_states, dtype=np.float32)

    def _apply_gait(self):
        # Generate foot trajectories for a trot gait
        time = self.time
        phase_offsets = [0, np.pi, np.pi, 0]  # FL, FR, HL, HR
        for i, phase_offset in enumerate(phase_offsets):
            foot_target = self._generate_foot_trajectory(time, phase_offset)
            self._set_foot_target(i, foot_target)

    def _generate_foot_trajectory(self, time, phase_offset):
        # Parameters
        stride_length = 0.2  # Length of the step
        step_height = 0.1    # Height of the step
        frequency = 1.0 / self.gait_period  # Frequency of the gait

        # Horizontal motion (forward and backward)
        x = stride_length * np.sin(2 * np.pi * frequency * time + phase_offset)

        # Vertical motion (up and down)
        z = step_height * np.abs(np.sin(2 * np.pi * frequency * time + phase_offset))

        return np.array([x, 0, z])

    def _set_foot_target(self, leg_index, foot_target):
        # Use inverse kinematics to set joint angles for the foot target
        # (This is a placeholder; you'll need to implement IK or use a solver)
        pass

    def _get_reward(self):
        # Stability
        stability_reward = self._get_stability_reward()

        # Gait efficiency
        gait_reward = self._get_gait_efficiency_reward()

        # Forward velocity
        velocity_reward = self._get_velocity_reward()

        # Energy efficiency
        energy_penalty = self._get_energy_penalty()

        # Sliding penalty
        sliding_penalty = self._get_sliding_penalty()

        # Total reward
        reward = stability_reward + gait_reward + velocity_reward + energy_penalty + sliding_penalty
        return reward

    def _get_stability_reward(self):
        body_orientation = self.data.qpos[3:7]
        desired_orientation = np.array([1, 0, 0, 0])  # Upright orientation
        orientation_error = np.linalg.norm(body_orientation - desired_orientation)

        body_height = self.data.qpos[2]
        desired_height = 0.75
        height_error = abs(body_height - desired_height)

        return -orientation_error - height_error

    def _get_gait_efficiency_reward(self):
        current_foot_positions = self._get_foot_positions()
        foot_movement = np.linalg.norm(current_foot_positions - self.last_foot_positions)
        self.last_foot_positions = current_foot_positions
        return 1.5 * foot_movement

    def _get_velocity_reward(self):
        forward_velocity = self.data.qvel[0]  # X-axis velocity
        return 1.0 * forward_velocity

    def _get_energy_penalty(self):
        joint_torques = np.abs(self.data.ctrl).sum()
        return -0.05 * joint_torques

    def _get_sliding_penalty(self):
        foot_velocities = self._get_foot_velocities()
        sliding_penalty = np.linalg.norm(foot_velocities)
        return sliding_penalty

    def _get_foot_velocities(self):
        # Get the velocities of all four feet
        foot_velocities = []
        for foot_name in self.foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if foot_id == -1:
                raise ValueError(f"Foot '{foot_name}' not found in the model.")
            
            # Get the body ID associated with the foot geom
            body_id = self.model.geom_bodyid[foot_id]
            
            # Get the linear velocity of the body (foot) in the world frame
            foot_vel = self.data.subtree_linvel[body_id]
            foot_velocities.extend(foot_vel)
        return np.array(foot_velocities)

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
            self.cam.distance = 2.5  # Move camera further back
            self.cam.elevation = -20  # Tilt downward
            self.cam.azimuth = 90  # Keep the camera behind the robot initially

        # Get robot position
        robot_position = self.data.qpos[:3]  # (x, y, z) position

        # Offset the camera to be behind the robot in the X direction
        camera_offset = np.array([-3.5, 0, 0.75])  # (backward, lateral, height)

        # Rotate the offset if the robot turns
        yaw_angle = np.arctan2(self.data.qvel[1], self.data.qvel[0])  # Get heading direction
        rotation_matrix = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])
        
        camera_position = robot_position + rotation_matrix @ camera_offset  # Rotate offset around robot

        # Smoothly interpolate camera movement (prevents jitter)
        alpha = 0.1  # Adjust for smoothness (0 = no movement, 1 = instant movement)
        self.cam.lookat[:] = alpha * robot_position + (1 - alpha) * self.cam.lookat[:]
        
        # Set the new camera position and update the scene
        self.cam.distance = np.linalg.norm(camera_position - robot_position)  # Maintain consistent distance
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