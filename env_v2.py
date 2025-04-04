import gym
from gym import spaces
import numpy as np
import mujoco
import glfw
from scipy.spatial.transform import Rotation

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
        self.foot_geom_ids = self._get_leg_geom_ids()
        self.leg_phases = np.array([0, np.pi, np.pi, 0])  # Trot gait
        self.prev_contact_states = np.zeros(4)
        self.contact_forces = np.zeros(4)
        self.swing_states = np.zeros(4, dtype=bool)
        
        # Update observation space size
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.model.nq + self.model.nv + 3 + 3 + 3 + 12 + 12 + 4 + 4 + 2 + 4,),
            dtype=np.float32
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
        mujoco.mj_resetData(self.model, self.data)
        self._add_noise_to_reset()
        self.time = 0
        self.leg_phases = np.array([0, np.pi, np.pi, 0])  # Reset phases
        self.prev_contact_states = np.zeros(4)
        self.contact_forces = np.zeros(4)
        return self._get_obs()

    def _get_obs(self):
        """Replace with this new version"""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Body info
        _, euler_angles = self._get_body_orientation()
        body_lin_vel = self.data.qvel[:3]
        body_ang_vel = self.data.qvel[3:6]
        
        # Foot info
        foot_positions = self._get_foot_positions().flatten()
        foot_velocities = self._get_foot_velocities().flatten()
        contact_states = self._get_contact_states()
        contact_forces = self._get_foot_contacts_force()
        
        # Gait info
        cop = self._get_center_of_pressure()
        phases = self.leg_phases.copy()
        
        return np.concatenate([
            qpos, qvel,
            euler_angles,
            body_lin_vel,
            body_ang_vel,
            foot_positions,
            foot_velocities,
            contact_states,
            contact_forces,
            cop,
            phases
        ])

    # def step(self, action):
    #     """Replace with this version"""
    #     # Apply action
    #     self.data.ctrl[:] = action
        
    #     # Update foot targets
    #     foot_targets = self._get_phase_based_foot_targets(self.time)
    #     for i in range(4):
    #         self._set_foot_target(i, foot_targets[i])
        
    #     # Step simulation
    #     mujoco.mj_step(self.model, self.data)
        
    #     # Update contact info
    #     self.prev_contact_states = self._get_contact_states()
    #     self.contact_forces = self._get_foot_contacts_force()
    #     self.swing_states = self._get_leg_swing_states()
    #     self._update_gait_phases()
        
    #     # Return step data
    #     obs = self._get_obs()
    #     reward = self._get_reward()
    #     done = self._get_done()
    #     info = {
    #         'contacts': self.prev_contact_states.copy(),
    #         'phases': self.leg_phases.copy()
    #     }
        
    #     self.time += self.model.opt.timestep
    #     return obs, reward, done, info
    def step(self, action):
        # Apply control action
        foot_targets = self._get_phase_based_foot_targets(self.time)
    
        # 2. Let policy modify targets (residual learning)
        foot_targets += 0.1 * action.reshape(4, 3)  # Small, learnable adjustments
        
        # 3. Apply IK control
        for i in range(4):
            self._set_foot_target(i, foot_targets[i])
        
        # 4. Step physics
        mujoco.mj_step(self.model, self.data)
        # Get observation, reward, and done flag
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        info = {}

        # Increment time for gait cycle
        self.time += self.model.opt.timestep

        return obs, reward, done, info
    
    def _get_foot_positions(self):
        # Get the positions of all four feet relative to the body
        foot_positions = []
        for foot_name in self.foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            foot_pos = self.data.geom_xpos[foot_id]
            body_pos = self.data.qpos[0:3]
            # foot_positions.extend(foot_pos - body_pos)
            foot_positions.append(foot_pos - body_pos)
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

    # def _set_foot_target(self, leg_index, foot_target):
    #     # Use inverse kinematics to set joint angles for the foot target
    #     # (This is a placeholder; you'll need to implement IK or use a solver)
    #     pass

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

        ##new rewards 
        # foot_clearance_reward = self._get_foot_clearance_reward()
        # gait_sync_reward = self._get_gait_sync_reward()
        # # smoothness_penalty = self._get_smoothness_penalty()
        # directionality_penalty = self._get_directionality_penalty()
        # Total reward
        reward = stability_reward + gait_reward + velocity_reward + energy_penalty + sliding_penalty
        
        # print(f"Stability: {stability_reward}, Gait: {gait_reward}, Velocity: {velocity_reward}, "
        # f"Energy: {energy_penalty}, Sliding: {sliding_penalty}, Clearance: {foot_clearance_reward}, "
        # f"Gait Sync: {gait_sync_reward}, Smoothness: none, Directionality: {directionality_penalty}, "
        # f"Total: {reward}")
        
        return reward

    def _get_stability_reward(self):
        body_orientation = self.data.qpos[3:7]
        desired_orientation = np.array([1, 0, 0, 0])  # Upright orientation
        orientation_error = np.linalg.norm(body_orientation - desired_orientation)

        body_height = self.data.qpos[2]
        desired_height = 0.52 #0.75 Old Desired height - Qpos - 2 : 0.46
        height_error = abs(body_height - desired_height)

        # orientation_reward = -orientation_error / 10.0  # Scale down
        # height_reward = -height_error / 0.1  # Scale down

        # return orientation_reward + height_reward

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

    # def _get_sliding_penalty(self):
    #     foot_velocities = self._get_foot_velocities()
    #     sliding_penalty = np.linalg.norm(foot_velocities)
    #     return -sliding_penalty

    def _get_sliding_penalty(self):
        """Returns penalty for foot sliding during contact."""
        foot_velocities = self._get_foot_velocities()  # (4, 3)
        foot_contacts = self._get_foot_contacts()      # (4,)
        # Compute sliding magnitude (L2 norm of XY velocity, ignore Z)
        sliding_speeds = np.linalg.norm(foot_velocities[:, :2], axis=1)  # (4,)
        
        # Only penalize sliding when foot is in contact
        sliding_penalty = np.sum(sliding_speeds * foot_contacts)
        
        return -sliding_penalty  # Negative because we want to minimize sliding

    # def _get_foot_velocities(self):
    #     # Get the velocities of all four feet
    #     foot_velocities = []
    #     for foot_name in self.foot_names:
    #         foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
    #         if foot_id == -1:
    #             raise ValueError(f"Foot '{foot_name}' not found in the model.")
            
    #         # Get the body ID associated with the foot geom
    #         body_id = self.model.geom_bodyid[foot_id]
            
    #         # Get the linear velocity of the body (foot) in the world frame
    #         foot_vel = self.data.subtree_linvel[body_id]
    #         foot_velocities.extend(foot_vel)
    #     return np.array(foot_velocities)

    def _get_foot_velocities(self):
        """Returns the linear velocity of all feet in world frame."""
        foot_velocities = []
        for foot_name in self.foot_names:  # ["FL", "FR", "HL", "HR"]
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            body_id = self.model.geom_bodyid[foot_id]
            foot_vel = self.data.cvel[body_id][3:6]  # Linear velocity components (3D)
            foot_velocities.append(foot_vel)
        return np.array(foot_velocities)  # Shape: (4, 3)


    def _get_foot_contacts(self):
        """Returns boolean array indicating foot-ground contact."""
        contacts = []
        for foot_name in self.foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            in_contact = False
            for contact in self.data.contact:
                if contact.geom1 == foot_id or contact.geom2 == foot_id:
                    in_contact = True
                    break
            contacts.append(in_contact)
        return np.array(contacts)  # Shape: (4,)

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

    ##Penalties
    def _get_smoothness_penalty(self):
        joint_accelerations = np.abs(self.data.qacc).sum()
        return -0.01 * joint_accelerations  # Penalize high accelerations

    def _get_foot_clearance_reward(self):
        foot_positions = self._get_foot_positions()
        clearance = foot_positions[:, 2]  # Z-axis positions
        return np.sum(clearance > 0.05)  # Reward feet that lift above 5 cm
    
    def _get_directionality_penalty(self):
        lateral_velocity = np.abs(self.data.qvel[1])  # Y-axis velocity
        backward_velocity = -min(0, self.data.qvel[0])  # Negative X-axis velocity
        return -0.1 * (lateral_velocity + backward_velocity)
    
    def _get_gait_sync_reward(self):
        # Reward for maintaining proper phase offsets between legs
        phase_offsets = [0, np.pi, np.pi, 0]  # Example for trot gait
        current_phases = self._get_foot_phases()  # Compute foot phases
        phase_errors = np.abs(current_phases - phase_offsets)
        return -0.01 * np.sum(phase_errors)  # Penalize deviations from desired phases

    def _get_foot_phases(self):
        """
        Compute the phase of each foot in the gait cycle.
        The phase is normalized to [0, 2π], where 0 represents the start of the cycle.
        """
        foot_positions = self._get_foot_positions()  # Get current foot positions
        foot_velocities = self._get_foot_velocities()  # Get current foot velocities

        # Initialize an array to store the phases
        foot_phases = []

        for i in range(len(self.foot_names)):  # Iterate over each foot
            # Use the Z-axis position (vertical motion) to determine the phase
            z_position = foot_positions[i, 2]  # Z-axis position of the foot
            z_velocity = foot_velocities[i, 2]  # Z-axis velocity of the foot

            # Determine the phase based on the foot's vertical motion
            if z_velocity > 0:  # Foot is moving upward (swing phase)
                phase = np.arctan2(z_velocity, z_position)
            else:  # Foot is moving downward (stance phase)
                phase = np.arctan2(-z_velocity, -z_position)

            # Normalize the phase to [0, 2π]
            phase = (phase + 2 * np.pi) % (2 * np.pi)
            foot_phases.append(phase)

        return np.array(foot_phases)

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

    def _get_leg_joint_indices(self, leg_index):
        """Returns joint indices for a specific leg (0:FL, 1:FR, 2:HL, 3:HR)"""
        base_idx = leg_index * 3  # 3 joints per leg
        return [base_idx, base_idx + 1, base_idx + 2]
    
    def _get_leg_geom_ids(self):
        """Returns geom IDs for all feet in order [FL, FR, HL, HR]"""
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) 
                for name in ["FL", "FR", "HL", "HR"]]
    
    def _get_joint_ranges(self):
        """Returns joint limits as (lower, upper) for all joints"""
        return self.model.jnt_range.copy()
    
    def _get_default_joint_angles(self):
        """Returns default joint angles from the keyframe"""
        return self.model.key_qpos[7:]  # Skip free joint (pos+quat)
    
    def _leg_ik(self, leg_index, foot_target):
        """Robust inverse kinematics with numerical safeguards"""
        # Leg lengths from your XML
        L1 = 0.1108  # Upper leg length (hip to knee)
        L2 = 0.32    # Lower leg length (knee to foot)
        
        x, y, z = foot_target
        
        # 1. Hip abduction (sideways movement)
        hx_angle = np.arctan2(y, np.sqrt(x**2 + z**2 + 1e-10))  # Add small epsilon
        
        # 2. Calculate distance with numerical safeguards
        d_squared = x**2 + z**2 - y**2
        d_squared = max(d_squared, 0)  # Ensure non-negative
        d = np.sqrt(d_squared)
        
        # 3. Knee angle with cosine clipping
        cos_kn = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        cos_kn = np.clip(cos_kn, -1, 1)  # Strict clipping
        
        # Handle edge cases where leg is fully extended
        if abs(cos_kn - 1) < 1e-6:
            kn_angle = 0
        elif abs(cos_kn + 1) < 1e-6:
            kn_angle = np.pi
        else:
            kn_angle = np.pi - np.arccos(cos_kn)
        
        # 4. Hip flexion with fallback for singularities
        try:
            hy_angle = np.arctan2(x, z) + np.arcsin(L1 * np.sin(np.pi - kn_angle) / (d + 1e-10))
        except:
            hy_angle = np.arctan2(x, z)  # Fallback
        
        return np.array([hx_angle, hy_angle, kn_angle])
    
    def _set_foot_target(self, leg_index, foot_target):
        # """Sets joint angles to reach target foot position (relative to body)"""
        # # Get hip position in world frame
        # hip_pos = self._get_hip_positions()[leg_index]
        # body_pos = self.data.qpos[0:3]
        
        # # Convert to hip-relative coordinates
        # hip_rel_target = foot_target - (hip_pos - body_pos)
        pass
        # # Compute IK
        # target_angles = self._leg_ik(leg_index, hip_rel_target)
        
        # # Get joint indices for this leg
        # joint_indices = self._get_leg_joint_indices(leg_index)
        
        # # Apply with PD control
        # kp = 0.5
        # kd = 0.05
        # for i, j in enumerate(joint_indices):
        #     error = target_angles[i] - self.data.qpos[7 + j]  # 7 = free joint dims
        #     self.data.ctrl[j] = kp * error - kd * self.data.qvel[6 + j]  # 6 = free joint vel dims
    
    def _get_hip_positions(self):
        """Returns world-frame positions of all hip joints (updated for current MuJoCo API)"""
        hip_positions = []
        for hip_name in ["fl_hip", "fr_hip", "hl_hip", "hr_hip"]:
            hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hip_name)
            hip_positions.append(self.data.xpos[hip_id])  # Changed from body_xpos to xpos
        return np.array(hip_positions)
    
    def _get_body_orientation(self):
        """Returns body orientation as rotation matrix and Euler angles"""
        quat = self.data.qpos[3:7]
        rot = Rotation.from_quat(quat[[1, 2, 3, 0]])  # MuJoCo to scipy convention
        return rot.as_matrix(), rot.as_euler('xyz')
    
    def _get_foot_contacts_force(self):
        """Returns contact forces for all feet (normal force magnitude)"""
        contact_forces = np.zeros(4)
        foot_geom_ids = self._get_leg_geom_ids()
        
        for i, geom_id in enumerate(foot_geom_ids):
            for contact in self.data.contact:
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    # Convert contact force to world frame
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, 0, force)
                    contact_forces[i] = force[0]  # Normal force
                    break
        return contact_forces
    
    def _get_ground_reaction_forces(self):
        """Returns GRF vectors for all feet in world frame"""
        grf = np.zeros((4, 3))
        foot_geom_ids = self._get_leg_geom_ids()
        
        for i, geom_id in enumerate(foot_geom_ids):
            for contact in self.data.contact:
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    # Convert contact force to world frame
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, 0, force)
                    grf[i] = force[:3]  # Force vector
                    break
        return grf
    
    def _add_noise_to_reset(self):
        """Simplified version using only numpy"""
        # Joint and position noise (same as before)
        noise_scale = 0.1
        self.data.qpos[7:] += noise_scale * np.random.randn(self.model.nq - 7)
        self.data.qpos[:2] += 0.05 * np.random.randn(2)
        self.data.qpos[2] += 0.01 * np.random.randn()
        
        # Orientation noise - small random axis-angle perturbation
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = 0.05 * np.random.randn()
        noise_quat = np.zeros(4)
        noise_quat[0] = np.cos(angle/2)  # w
        noise_quat[1:] = np.sin(angle/2) * axis  # x,y,z
        
        # Combine with original quaternion (MuJoCo quat multiplication)
        original_quat = self.data.qpos[3:7]
        w = original_quat[0]*noise_quat[0] - np.dot(original_quat[1:], noise_quat[1:])
        xyz = original_quat[0]*noise_quat[1:] + noise_quat[0]*original_quat[1:] + np.cross(original_quat[1:], noise_quat[1:])
        self.data.qpos[3:7] = np.array([w, *xyz])
        
        # Velocity noise
        self.data.qvel[:] += 0.01 * np.random.randn(self.model.nv)

    def _get_phase_based_foot_targets(self, time):
        """Returns foot targets for all legs based on gait phase"""
        targets = []
        for i, phase_offset in enumerate([0, np.pi, np.pi, 0]):  # Trot gait
            targets.append(self._generate_foot_trajectory(time, phase_offset))
        return np.array(targets)
    
    def _get_foot_phase(self, foot_pos, foot_vel):
        """Estimates phase of foot in gait cycle (0-2π)"""
        # Normalize vertical position and velocity
        z_norm = (foot_pos[2] + 0.32) / 0.1  # Approximate range
        vz_norm = foot_vel[2] / 0.5          # Approximate max velocity
        
        # Simple phase estimation based on vertical motion
        phase = np.arctan2(vz_norm, z_norm) % (2 * np.pi)
        return phase
    
    def _get_leg_swing_states(self):
        """Returns boolean array indicating which legs are in swing phase (not in contact)"""
        contact_states = np.zeros(4, dtype=bool)
        
        # Get all foot geom IDs
        foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) 
                        for name in ["FL", "FR", "HL", "HR"]]
        
        # Check contacts for each foot
        for i, geom_id in enumerate(foot_geom_ids):
            # Check if geom exists
            if geom_id == -1:
                continue
                
            # Check all active contacts
            for contact_idx in range(self.data.ncon):
                contact = self.data.contact[contact_idx]
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    contact_states[i] = True
                    break
        
        return ~contact_states  # Return swing states (opposite of contact states)
    
    def _get_center_of_pressure(self):
        """Calculates center of pressure from ground reaction forces"""
        grf = self._get_ground_reaction_forces()
        foot_positions = self._get_foot_positions()
        
        # Only consider feet in contact
        contact_states = self._get_contact_states()
        if not np.any(contact_states):
            return np.zeros(2)  # No contact
        
        normal_forces = np.linalg.norm(grf, axis=1)
        total_force = np.sum(normal_forces * contact_states)
        
        if total_force < 1e-6:  # Avoid division by zero
            return np.zeros(2)
        
        # Weighted average of foot positions
        cop_x = np.sum(foot_positions[:, 0] * normal_forces * contact_states) / total_force
        cop_y = np.sum(foot_positions[:, 1] * normal_forces * contact_states) / total_force
        
        return np.array([cop_x, cop_y])
    
    def _get_body_orientation(self):
        """Returns body orientation as Euler angles"""
        quat = self.data.qpos[3:7]
        rot = Rotation.from_quat(quat[[1, 2, 3, 0]])  # MuJoCo to scipy convention
        return rot.as_matrix(), rot.as_euler('xyz')

    def _get_foot_contacts_force(self):
        """Returns normal contact forces for all feet"""
        forces = np.zeros(4)
        for i, geom_id in enumerate(self.foot_geom_ids):
            for contact in self.data.contact:
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, 0, force)
                    forces[i] = force[0]  # Normal force
                    break
        return forces

    def _update_gait_phases(self):
        """Updates leg phases based on contact events"""
        for i in range(4):
            if not self.prev_contact_states[i] and self._get_contact_states()[i]:
                self.leg_phases[i] = 0  # Reset phase on new contact
            self.leg_phases[i] += 2*np.pi*self.model.opt.timestep/self.gait_period
            self.leg_phases[i] %= 2*np.pi