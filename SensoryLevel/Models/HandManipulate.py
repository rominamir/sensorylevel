import os
import numpy as np
from gym import utils, error
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 5.0,
}

class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hand0_slider_3tendon_ball.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[8]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class HandEnvRot0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 # xml_file='hand0_3tendon_ball_rot.xml',
                 # xml_file='hand0_3DOF_ball_rot.xml',
                 xml_file='hand_3DOF_ball_rot_v0.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[13]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[13]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # sensor = self.sim.data.sensordata.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

   # def get_sensor_sensordata(self):
    #    return self.data.sensordata
    

class HandEnvRot1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 # xml_file='hand0_3tendon_ball_rot.xml',
                 # xml_file='hand0_3DOF_ball_rot.xml',
                 xml_file='hand_3DOF_ball_rot_v1.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[13]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[13]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # sensor = self.sim.data.sensordata.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class HandEnvRot2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 # xml_file='hand0_3tendon_ball_rot.xml',
                 # xml_file='hand0_3DOF_ball_rot.xml',
                 xml_file='hand_3DOF_ball_rot_v2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[13]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[13]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # sensor = self.sim.data.sensordata.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)



class HandEnvRot0ver2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hand0_3tendon_ball_rot-v0.xml',
                 # xml_file='hand0_3DOF_ball_rot.xml',
                 # xml_file='hand_3DOF_ball_rot_v2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[10]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[10]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # sensor = self.sim.data.sensordata.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class HandEnvRot1ver2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hand0_3tendon_ball_rot-v1.xml',
                 # xml_file='hand0_3DOF_ball_rot.xml',
                 # xml_file='hand_3DOF_ball_rot_v2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[10]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[10]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # sensor = self.sim.data.sensordata.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class HandEnvRot2ver2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hand0_3tendon_ball_rot-v2.xml',
                 # xml_file='hand0_3DOF_ball_rot.xml',
                 # xml_file='hand_3DOF_ball_rot_v2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[10]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[10]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # sensor = self.sim.data.sensordata.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        # observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)



'''
class HandEnvRotSensory(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hand0_3tendon_ball_rot.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True,
                 touch='sensordata'):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        self.touch = touch

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[11]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[11]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        touch_values = self.sim.data.sensordata.copy()
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity, touch_values)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_sensor_sensordata(self):
        return self.data.sensordata

'''

'''
def quat_from_angle_and_axis(axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    #quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(axis)
    return quat
'''
'''
class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 reward_type,
                 target_position,
                 target_position_range,

                 xml_file='hand0_slider_3tendon_ball.xml',
                 # target_rotation,
                 initial_qpos=None,
                 randomize_initial_position=True,
                 # randomize_initial_rotation=True,
                 distance_threshold=0.01,
                 # rotation_threshold=0.1,
                 n_substeps=20,
                 relative_control=False,
                 # ignore_z_target_rotation=False
                 ):

        utils.EzPickle.__init__(**locals())

        self.target_position = target_position
        #self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.reward_type = reward_type
        #self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]
        #self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        #self.rotation_threshold = rotation_threshold
        #self.ignore_z_target_rotation = ignore_z_target_rotation

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)



    def _get_achieved_goal(self):
        # Object position and rotation.
        object_qpos = self.sim.data.get_joint_qpos('object:joint')
        #assert object_qpos.shape == (7,)
        return object_qpos


    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
       # d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        #if self.target_rotation != 'ignore':
            #quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            # if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                #euler_a = rotations.quat2euler(quat_a)
                #euler_b = rotations.quat2euler(quat_b)
                #euler_a[2] = euler_b[2]
                #quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            #quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            #angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            #d_rot = angle_diff
        #assert d_pos.shape == d_rot.shape
        return d_pos #, d_rot

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return (success - 1.)
        else:
            d_pos = self._goal_distance(achieved_goal, goal)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).
            return -(10. * d_pos )

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        #achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        #achieved_both = achieved_pos * achieved_rot
        return achieved_pos

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None


        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        goal = target_pos
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

'''

'''
class HandBlockEnv(hand1Env, utils.EzPickle):
    def __init__(self, target_position='random', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, reward_type)
        ManipulateEnv.__init__(self,
                               xml_file='hand0_slider_3tendon_ball.xml', target_position=target_position,
                                #target_rotation=target_rotation,
                                target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                                reward_type=reward_type)
'''
'''
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

'''


'''
class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hand0_slider_3tendon_ball.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())
        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    #def control_cost(self, action):
     #   control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
     #  return control_cost

  def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        #reward = standing_cost - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info
  
  '''

# C:\Users\Romin\AppData\Roaming\Python\Python37\site-packages\gym\envs>