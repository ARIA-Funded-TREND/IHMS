# # # coding=utf-8
# # # Copyright 2020 The Google Research Authors.
# # #
# # # Licensed under the Apache License, Version 2.0 (the "License");
# # # you may not use this file except in compliance with the License.
# # # You may obtain a copy of the License at
# # #
# # #     http://www.apache.org/licenses/LICENSE-2.0
# # #
# # # Unless required by applicable law or agreed to in writing, software
# # # distributed under the License is distributed on an "AS IS" BASIS,
# # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # # See the License for the specific language governing permissions and
# # # limitations under the License.

# # """A simple locomotion task and termination condition."""

# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
# # import numpy as np

# # import os
# # import inspect
# # currentdir = os.path.dirname(os.path.abspath(
# #   inspect.getfile(inspect.currentframe())))
# # parentdir = os.path.dirname(os.path.dirname(currentdir))
# # os.sys.path.insert(0, parentdir)


# # class MoveForwardTask(object):
# #   """move forward task."""

# #   def __init__(
# #       self,
# #       z_constrain=False,
# #       move_forward_coeff=1,
# #       other_direction_penalty=0,
# #       z_penalty=0,
# #       orientation_penalty=1,
# #       time_step_s=0.01,
# #       num_action_repeat=10,
# #       height_fall_coeff=0.3,
# #       alive_reward=0.1,
# #       fall_reward=0,
# #       target_vel=None,
# #       check_contact=False,
# #       target_vel_dir=(1, 0),
# #       subgoal_reward=None
# #       # init_orientation=None,
# #   ):
# #     """Initializes the task."""
# #     self._draw_ref_model_alpha = 1.
# #     # self.energy_weight = -0.01
# #     self.energy_weight = -0.005
# #     self.move_forward_coeff = move_forward_coeff
# #     self._ref_model = -1
# #     self._alive_reward = alive_reward
# #     self.fall_reward = fall_reward
# #     self._time_step = time_step_s
# #     self.num_action_repeat = num_action_repeat
# #     self.z_constrain = z_constrain
# #     self.other_direction_penalty = other_direction_penalty
# #     self.z_penalty = z_penalty
# #     self.init_orientation = np.array([0, 0, 0, 1])
# #     self.orientation_penalty = orientation_penalty
# #     self.height_fall_coeff = height_fall_coeff
# #     self.target_vel = target_vel
# #     self.check_contact = check_contact
# #     # return
# #     self.target_vel_dir = np.array(target_vel_dir)
# #     self.subgoal_reward = subgoal_reward

# #   def __call__(self, env):
# #     return self.reward(env)

# #   def reset(self, env):
# #     """Resets the internal state of the task."""
# #     self._env = env
# #     self.last_base_pos = env.robot.GetBasePosition()
# #     self.current_base_pos = self.last_base_pos

# #     if self.subgoal_reward is not None:
# #       self.subgoal_trackers = np.ones(
# #         len(env._env_randomizers[-1].subgoal_ids),
# #         dtype=np.uint8
# #       )

# #   def update(self, env):
# #     """Updates the internal state of the task."""
# #     self.last_base_pos = self.current_base_pos
# #     self.current_base_pos = env.robot.GetBasePosition()

# #   def done(self, env):
# #     """Checks if the episode is over."""
# #     del env
# #     env = self._env
# #     pyb = env._pybullet_client

# #     root_pos_sim, _ = pyb.getBasePositionAndOrientation(
# #       env.robot.quadruped)

# #     rot_quat = env.robot.GetBaseOrientation()
# #     rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

# #     rot_fall = rot_mat[-1] < 0.6
# #     height_fall = root_pos_sim[2] < self.height_fall_coeff
# #     if self.z_constrain:
# #       height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

# #     contact_done = False

# #     if self.check_contact:
# #       contacts = env._pybullet_client.getContactPoints(
# #         bodyA=env.robot.quadruped)
# #       for contact in contacts:
# #         if contact[2] is env._world_dict["ground"] or \
# #             ("terrain" in env._world_dict and
# #              contact[2] is env._world_dict["terrain"]):
# #           if contact[3] not in env.robot._foot_link_ids:
# #             contact_done = True
# #             break

# #       # contacts = env._pybullet_client.getContactPoints(bodyA=env.robot.quadruped)
# #       for contact in contacts:
# #         if contact[2] is not env._world_dict["ground"]:
# #           contact_done = True
# #           break
# #       speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
# #         (self._time_step * self.num_action_repeat)

# #       contact_done = contact_done and np.linalg.norm(speed) <= 0.05
# #     done = height_fall or rot_fall or contact_done
# #     return done

# #   def reward(self, env):
# #     """Get the reward without side effects."""
# #     del env

# #     env = self._env
# #     # energy_reward = np.abs(
# #     #     np.dot(env.robot.GetMotorTorques(),
# #     #            env.robot.GetMotorVelocities())) * self._time_step

# #     energy_reward = np.dot(
# #       env.robot.GetMotorTorques(),
# #       env.robot.GetMotorTorques()
# #     ) * self._time_step

# #     move_forward_reward = self._calc_reward_root_velocity()
# #     alive_reward = self._alive_reward
# #     orientation_reward = self._calc_reward_rotation()

# #     reward = move_forward_reward * self.move_forward_coeff + \
# #       energy_reward * self.energy_weight - \
# #       self.orientation_penalty * orientation_reward + \
# #       alive_reward
# #     # print("Rew:{:.4f} Move Rew:{:.4f}, Ori Rew:{:.4f}, Eng Rew:{:.4f}".format(
# #     #   reward, move_forward_reward, orientation_reward, energy_reward))
# #     # print(move_forward_reward)
# #     done = self.done(env)
# #     if done:
# #       reward += self.fall_reward

# #     if self.subgoal_reward is not None:
# #       # self.last_base_pos = env.robot.GetBasePosition()
# #       # print(self.current_base_pos)
# #       dis = env._env_randomizers[-1].subgoal_centers - \
# #         self.current_base_pos[:2]
# #       dis = np.linalg.norm(dis, axis=1)

# #       contacted_ones = np.where(
# #         (dis < env._env_randomizers[-1].radius) * self.subgoal_trackers
# #       )[0]

# #       for contacted_idx in contacted_ones:
# #         self.subgoal_trackers[contacted_idx] = 0
# #         reward += self.subgoal_reward

# #         env.pybullet_client.changeVisualShape(
# #           env._env_randomizers[-1].subgoal_ids[contacted_idx],
# #           -1,
# #           rgbaColor=(1, 0.2, 0.2, 0)
# #         )
# #         # env.pybullet_client.removeBody(contacted_idx)

# #     return reward

# #   def _get_pybullet_client(self):
# #     """Get bullet client from the environment"""
# #     return self._env._pybullet_client

# #   def _calc_reward_root_velocity(self):
# #     """Get the root velocity reward."""
# #     env = self._env
# #     robot = env.robot
# #     sim_model = robot.quadruped

# #     pyb = self._get_pybullet_client()

# #     root_vel_sim, _ = pyb.getBaseVelocity(sim_model)
# #     root_vel_sim = np.array(root_vel_sim)

# #     x_speed = (self.current_base_pos[0] - self.last_base_pos[0]
# #                ) / (self._time_step * self.num_action_repeat)
# #     y_speed = (self.current_base_pos[1] - self.last_base_pos[1]
# #                ) / (self._time_step * self.num_action_repeat)
# #     z_speed = (self.current_base_pos[2] - self.last_base_pos[2]
# #                ) / (self._time_step * self.num_action_repeat)

# #     xy_speed = np.array([x_speed, y_speed])

# #     along_speed = np.dot(xy_speed, self.target_vel_dir)
# #     per_speed = xy_speed - along_speed * self.target_vel_dir

# #     along_speed = np.clip(
# #       along_speed, a_min=None, a_max=self.target_vel
# #     )
# #     along_reward = self.target_vel ** 2 - (
# #       along_speed - self.target_vel
# #     ) ** 2

# #     forward_reward = along_reward - \
# #       self.other_direction_penalty * (np.linalg.norm(per_speed) ** 2) - \
# #       self.z_penalty * (z_speed ** 2)

# #     return forward_reward

# #   def _calc_reward_rotation(self):
# #     env = self._env
# #     pyb = self._get_pybullet_client()

# #     rot_quat = env.robot.GetBaseOrientation()

# #     if self.init_orientation is None:
# #       return 0
# #     # Norm of displacement vector
# #     rot_reward = np.sum(
# #       (self.init_orientation - np.array(rot_quat)) ** 2)  # * self.num_action_repeat
# #     return rot_reward



# # coding=utf-8
# # Copyright 2020 The Google Research Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """A simple locomotion task and termination condition."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import numpy as np

# import os
# import inspect
# currentdir = os.path.dirname(os.path.abspath(
#   inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)


# class MoveForwardTask(object):
#   """move forward task."""

#   def __init__(
#       self,
#       z_constrain=False,
#       move_forward_coeff=1,
#       other_direction_penalty=0,
#       z_penalty=0,
#       orientation_penalty=0,
#       time_step_s=0.01,
#       num_action_repeat=10,
#       height_fall_coeff=0.3,
#       alive_reward=0.1,
#       fall_reward=0,
#       target_vel=None,
#       check_contact=False,
#       target_vel_dir=(1, 0),
#       subgoal_reward=None
#       # init_orientation=None,
#   ):
#     """Initializes the task."""
#     self._draw_ref_model_alpha = 1.
#     # self.energy_weight = -0.01
#     self.energy_weight = -0.005
#     self.move_forward_coeff = move_forward_coeff
#     self._ref_model = -1
#     self._alive_reward = alive_reward
#     self.fall_reward = fall_reward
#     self._time_step = time_step_s
#     self.num_action_repeat = num_action_repeat
#     self.z_constrain = z_constrain
#     self.other_direction_penalty = other_direction_penalty
#     self.z_penalty = z_penalty
#     self.init_orientation = np.array([0, 0, 0, 1])
#     self.orientation_penalty = orientation_penalty
#     self.height_fall_coeff = height_fall_coeff
#     self.target_vel = target_vel
#     self.check_contact = check_contact
#     # return
#     self.target_vel_dir = np.array(target_vel_dir)
#     self.subgoal_reward = subgoal_reward
#     print("Move_Forward_Task Init")

#   def __call__(self, env):
#     return self.reward(env)

#   def reset(self, env):
#     """Resets the internal state of the task."""
#     self._env = env
#     self.last_base_pos = env.robot.GetBasePosition()
#     self.current_base_pos = self.last_base_pos

#     if self.subgoal_reward is not None:
#       self.subgoal_trackers = np.ones(
#         len(env._env_randomizers[-1].subgoal_ids),
#         dtype=np.uint8
#       )

#   def update(self, env):
#     """Updates the internal state of the task."""
#     self.last_base_pos = self.current_base_pos
#     self.current_base_pos = env.robot.GetBasePosition()


#     ######################################################### DONE LAST USED ################################################

#   # def done(self, env):
#   #   """Checks if the episode is over."""
#   #   del env
#   #   env = self._env
#   #   pyb = env._pybullet_client

#   #   root_pos_sim, _ = pyb.getBasePositionAndOrientation(
#   #     env.robot.quadruped)

#   #   rot_quat = env.robot.GetBaseOrientation()
#   #   rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

#   #   rot_fall = rot_mat[-1] < 0.6
#   #   height_fall = root_pos_sim[2] < self.height_fall_coeff
#   #   if self.z_constrain:
#   #     height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

#   #   contact_done = False

#   #   # 1. Pull the speed calculation OUTSIDE the contact loop so we always know how fast it's moving
#   #   speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
#   #     (self._time_step * self.num_action_repeat)
#   #   speed_norm = np.linalg.norm(speed)

#   #   if self.check_contact:
#   #     contacts = env._pybullet_client.getContactPoints(
#   #       bodyA=env.robot.quadruped)
#   #     for contact in contacts:
#   #       if contact[2] is env._world_dict["ground"] or \
#   #           ("terrain" in env._world_dict and
#   #            contact[2] is env._world_dict["terrain"]):
#   #         if contact[3] not in env.robot._foot_link_ids:
#   #           contact_done = True
#   #           break

#   #     for contact in contacts:
#   #       if contact[2] is not env._world_dict["ground"]:
#   #         contact_done = True
#   #         break

#   #     contact_done = contact_done and speed_norm <= 0.05

#   #   # =========================================================================
#   #   # NEW: 100 STEPS STUCK CONDITION
#   #   # =========================================================================
#   #   stuck_done = False
    
#   #   # Initialize a step counter if it doesn't exist yet
#   #   if not hasattr(self, 'stuck_step_counter'):
#   #       self.stuck_step_counter = 0

#   #   # If speed is virtually zero, increment the counter. Otherwise, reset it to 0.
#   #   if speed_norm <= 0.05:
#   #       self.stuck_step_counter += 1
#   #   else:
#   #       self.stuck_step_counter = 0

#   #   # If it has been stuck for 100 continuous steps, end the episode
#   #   if self.stuck_step_counter >= 100:
#   #       stuck_done = True
#   #       self.stuck_step_counter = 0 # Reset for the next episode

#   #   # =========================================================================
#   #   # NEW: DETECT ALL-BLACK IMAGE CONDITION
#   #   # =========================================================================
#   #   vision_done = False
#   #   try:
#   #       obs_dict = env.get_observation()
#   #       vision_key = next((k for k in obs_dict.keys() if 'depth' in k or 'rgb' in k or 'vision' in k), None)
        
#   #       if vision_key is not None:
#   #           image_data = np.array(obs_dict[vision_key])
#   #           if np.max(image_data) < 0.01:
#   #               vision_done = True
#   #   except Exception as e:
#   #       pass

#   #   # =========================================================================

#   #   done = height_fall or rot_fall or contact_done or stuck_done or vision_done
#   #   return done


#     ###########################################################################################################################


    
#   # def done(self, env):
#   #   """Checks if the episode is over."""
#   #   del env
#   #   env = self._env
#   #   pyb = env._pybullet_client

#   #   root_pos_sim, _ = pyb.getBasePositionAndOrientation(
#   #     env.robot.quadruped)

#   #   rot_quat = env.robot.GetBaseOrientation()
#   #   rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

#   #   rot_fall = rot_mat[-1] < 0.6
#   #   height_fall = root_pos_sim[2] < self.height_fall_coeff
#   #   if self.z_constrain:
#   #     height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

#   #   contact_done = False

#   #   if self.check_contact:
#   #     contacts = env._pybullet_client.getContactPoints(
#   #       bodyA=env.robot.quadruped)
#   #     for contact in contacts:
#   #       if contact[2] is env._world_dict["ground"] or \
#   #           ("terrain" in env._world_dict and
#   #            contact[2] is env._world_dict["terrain"]):
#   #         if contact[3] not in env.robot._foot_link_ids:
#   #           contact_done = True
#   #           break

#   #     # contacts = env._pybullet_client.getContactPoints(bodyA=env.robot.quadruped)
#   #     for contact in contacts:
#   #       if contact[2] is not env._world_dict["ground"]:
#   #         contact_done = True
#   #         break
#   #     speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
#   #       (self._time_step * self.num_action_repeat)

#   #     contact_done = contact_done and np.linalg.norm(speed) <= 0.05
#   #   done = height_fall or rot_fall or contact_done
#   #   return done

#   # def reward(self, env):
#   #   """Get the reward without side effects."""
#   #   del env

#   #   env = self._env
#   #   # energy_reward = np.abs(
#   #   #     np.dot(env.robot.GetMotorTorques(),
#   #   #            env.robot.GetMotorVelocities())) * self._time_step

#   #   energy_reward = np.dot(
#   #     env.robot.GetMotorTorques(),
#   #     env.robot.GetMotorTorques()
#   #   ) * self._time_step

#   #   move_forward_reward = self._calc_reward_root_velocity()
#   #   alive_reward = self._alive_reward
#   #   orientation_reward = self._calc_reward_rotation()

#   #   reward = move_forward_reward * self.move_forward_coeff + \
#   #     energy_reward * self.energy_weight - \
#   #     self.orientation_penalty * orientation_reward + \
#   #     alive_reward
#   #   # print("Rew:{:.4f} Move Rew:{:.4f}, Ori Rew:{:.4f}, Eng Rew:{:.4f}".format(
#   #   #   reward, move_forward_reward, orientation_reward, energy_reward))
#   #   # print(move_forward_reward)
#   #   done = self.done(env)
#   #   if done:
#   #     reward += self.fall_reward

#   #   if self.subgoal_reward is not None:
#   #     # self.last_base_pos = env.robot.GetBasePosition()
#   #     # print(self.current_base_pos)
#   #     dis = env._env_randomizers[-1].subgoal_centers - \
#   #       self.current_base_pos[:2]
#   #     dis = np.linalg.norm(dis, axis=1)

#   #     contacted_ones = np.where(
#   #       (dis < env._env_randomizers[-1].radius) * self.subgoal_trackers
#   #     )[0]

#   #     for contacted_idx in contacted_ones:
#   #       self.subgoal_trackers[contacted_idx] = 0
#   #       reward += self.subgoal_reward

#   #       env.pybullet_client.changeVisualShape(
#   #         env._env_randomizers[-1].subgoal_ids[contacted_idx],
#   #         -1,
#   #         rgbaColor=(1, 0.2, 0.2, 0)
#   #       )
#   #       # env.pybullet_client.removeBody(contacted_idx)

#   #   return reward


#     ###################################################################################################################################


#   def done(self, env):
#     """Checks if the episode is over."""
#     del env
#     env = self._env
#     pyb = env._pybullet_client

#     root_pos_sim, _ = pyb.getBasePositionAndOrientation(
#       env.robot.quadruped)

#     rot_quat = env.robot.GetBaseOrientation()
#     rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

#     rot_fall = rot_mat[-1] < 0.6
#     height_fall = root_pos_sim[2] < self.height_fall_coeff
#     if self.z_constrain:
#       height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

#     contact_done = False

#     # Calculate speed norm for the stuck condition
#     speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
#       (self._time_step * self.num_action_repeat)
#     speed_norm = np.linalg.norm(speed)

#     if self.check_contact:
#       contacts = env._pybullet_client.getContactPoints(
#         bodyA=env.robot.quadruped)
#       for contact in contacts:
#         if contact[2] is env._world_dict["ground"] or \
#             ("terrain" in env._world_dict and
#              contact[2] is env._world_dict["terrain"]):
#           if contact[3] not in env.robot._foot_link_ids:
#             contact_done = True
#             break

#       for contact in contacts:
#         if contact[2] is not env._world_dict["ground"]:
#           contact_done = True
#           break

#       contact_done = contact_done and speed_norm <= 0.05

#     # =========================================================================
#     # 100 STEPS STUCK CONDITION
#     # =========================================================================
#     stuck_done = False
#     if not hasattr(self, 'stuck_step_counter'):
#         self.stuck_step_counter = 0

#     if speed_norm <= 0.05:
#         self.stuck_step_counter += 1
#     else:
#         self.stuck_step_counter = 0

#     if self.stuck_step_counter >= 100:
#         stuck_done = True
#         self.stuck_step_counter = 0
#     # =========================================================================
#     # DETECT ALL-BLACK / BLIND VISION CONDITION (PRECISE SLICING)
#     # =========================================================================
#     # =========================================================================
#     # DETECT ALL-BLACK / BLIND VISION CONDITION (100 CONSECUTIVE FRAMES)
#     # =========================================================================
#     vision_done = False
    
#     # Initialize the blind counter if it doesn't exist yet
#     if not hasattr(self, 'blind_step_counter'):
#         self.blind_step_counter = 0

#     try:
#         if hasattr(env, 'depth_frames') and len(env.depth_frames) > 0:
#             latest_frame = np.array(env.depth_frames[-1])
#             mean_val = np.mean(latest_frame)
            
#             # If the frame matches your blind fingerprint, increment
#             if 0.45 < mean_val < 0.65:
#                 self.blind_step_counter += 1
#             else:
#                 # If the robot sees something normal, instantly reset the counter
#                 self.blind_step_counter = 0
            
#             # If it has been blind for 100 continuous frames, end the episode
#             if self.blind_step_counter >= 100:
#                 vision_done = True
#                 self.blind_step_counter = 0 # Reset for the next episode
                
#     except Exception as e:
#         pass

#     # =========================================================================

#     done = height_fall or rot_fall or contact_done or stuck_done or vision_done
#     return done

#   def reward(self, env):
#     """Get the reward without side effects."""
#     del env
#     env = self._env

#     energy_reward = np.dot(
#       env.robot.GetMotorTorques(),
#       env.robot.GetMotorTorques()
#     ) * self._time_step

#     move_forward_reward = self._calc_reward_root_velocity()
#     alive_reward = self._alive_reward
#     orientation_reward = self._calc_reward_rotation()

#     reward = move_forward_reward * self.move_forward_coeff + \
#       energy_reward * self.energy_weight - \
#       self.orientation_penalty * orientation_reward + \
#       alive_reward

#     done = self.done(env)
#     if done:
#       reward += 2*self.fall_reward        # ADDED BY TALHA

#     if self.subgoal_reward is not None:
#       dis = env._env_randomizers[-1].subgoal_centers - \
#         self.current_base_pos[:2]
#       dis = np.linalg.norm(dis, axis=1)

#       contacted_ones = np.where(
#         (dis < env._env_randomizers[-1].radius) * self.subgoal_trackers
#       )[0]

#       for contacted_idx in contacted_ones:
#         self.subgoal_trackers[contacted_idx] = 0
#         reward += self.subgoal_reward

#         env.pybullet_client.changeVisualShape(
#           env._env_randomizers[-1].subgoal_ids[contacted_idx],
#           -1,
#           rgbaColor=(1, 0.2, 0.2, 0)
#         )

#     # =========================================================================
#     # APPLY ALL-BLACK IMAGE PENALTY (-5) USING PRECISE SLICING
#     # =========================================================================
#     try:
#         obs = env.get_observation()
#         image_data = None
        
#         if isinstance(obs, dict):
#             vision_key = next((k for k in obs.keys() if any(w in k.lower() for w in ['depth', 'rgb', 'vision', 'image'])), None)
#             if vision_key is not None:
#                 image_data = np.array(obs[vision_key])
#         elif isinstance(obs, np.ndarray):
#             split_idx = None
#             current_layer = env
#             while current_layer is not None:
#                 for attr in ['state_input_shape', 'state_shape']:
#                     if hasattr(current_layer, attr):
#                         val = getattr(current_layer, attr)
#                         split_idx = val[0] if isinstance(val, (tuple, list, getattr(np, 'ndarray', ()))) else val
#                         break
#                 if split_idx is not None:
#                     break
#                 current_layer = getattr(current_layer, 'env', None)

#             if split_idx is not None:
#                 image_data = obs[..., split_idx:]
#             else:
#                 if obs.shape[-1] > 100:
#                     image_data = obs[..., -1024:]

#         if image_data is not None:
#             mean_val = np.mean(image_data)
#             if -1.80 < mean_val < -1.65:
#                 reward -= 5.0
#     except Exception as e:
#         pass
#     # =========================================================================

#     return reward


# ################################################# REWARD LAST USED BY ME ################################################################

    
#   # def reward(self, env):
#   #   """Get the reward without side effects."""
#   #   del env

#   #   env = self._env
#   #   # energy_reward = np.abs(
#   #   #     np.dot(env.robot.GetMotorTorques(),
#   #   #            env.robot.GetMotorVelocities())) * self._time_step

#   #   energy_reward = np.dot(
#   #     env.robot.GetMotorTorques(),
#   #     env.robot.GetMotorTorques()
#   #   ) * self._time_step

#   #   move_forward_reward = self._calc_reward_root_velocity()
#   #   alive_reward = self._alive_reward
#   #   orientation_reward = self._calc_reward_rotation()

#   #   reward = move_forward_reward * self.move_forward_coeff + \
#   #     energy_reward * self.energy_weight - \
#   #     self.orientation_penalty * orientation_reward + \
#   #     alive_reward
#   #   # print("Rew:{:.4f} Move Rew:{:.4f}, Ori Rew:{:.4f}, Eng Rew:{:.4f}".format(
#   #   #    reward, move_forward_reward, orientation_reward, energy_reward))
#   #   # print(move_forward_reward)
#   #   done = self.done(env)
#   #   if done:
#   #     reward += self.fall_reward

#   #   if self.subgoal_reward is not None:
#   #     # self.last_base_pos = env.robot.GetBasePosition()
#   #     # print(self.current_base_pos)
#   #     dis = env._env_randomizers[-1].subgoal_centers - \
#   #       self.current_base_pos[:2]
#   #     dis = np.linalg.norm(dis, axis=1)

#   #     contacted_ones = np.where(
#   #       (dis < env._env_randomizers[-1].radius) * self.subgoal_trackers
#   #     )[0]

#   #     for contacted_idx in contacted_ones:
#   #       self.subgoal_trackers[contacted_idx] = 0
#   #       reward += self.subgoal_reward

#   #       env.pybullet_client.changeVisualShape(
#   #         env._env_randomizers[-1].subgoal_ids[contacted_idx],
#   #         -1,
#   #         rgbaColor=(1, 0.2, 0.2, 0)
#   #       )
#   #       # env.pybullet_client.removeBody(contacted_idx)

#   #   # =========================================================================
#   #   # DETECT ALL-BLACK IMAGE PENALTY
#   #   # =========================================================================
#   #   try:
#   #       # Fetch the raw observation dictionary from the gym environment
#   #       obs_dict = env.get_observation()
        
#   #       # Look for the vision key (common keys in vision4leg are 'depth', 'rgb', or 'vision')
#   #       # We search dynamically so your script won't crash if the key changes in JSON
#   #       vision_key = next((k for k in obs_dict.keys() if 'depth' in k or 'rgb' in k or 'vision' in k), None)
        
#   #       if vision_key is not None:
#   #           image_data = np.array(obs_dict[vision_key])
            
#   #           # Check if all pixels are completely black (or practically zero due to floating point noise)
#   #           if np.max(image_data) < 0.01:
#   #               reward -= 5.0
#   #   except Exception as e:
#   #       # Prevent the training process from crashing entirely if an observation wrapper alters the dict
#   #       pass
#   #   # =========================================================================

#   #   return reward

# #######################################################################################################################
    

#   def _get_pybullet_client(self):
#     """Get bullet client from the environment"""
#     return self._env._pybullet_client

#   def _calc_reward_root_velocity(self):
#     """Get the root velocity reward."""
#     env = self._env
#     robot = env.robot
#     sim_model = robot.quadruped

#     pyb = self._get_pybullet_client()

#     root_vel_sim, _ = pyb.getBaseVelocity(sim_model)
#     root_vel_sim = np.array(root_vel_sim)

#     x_speed = (self.current_base_pos[0] - self.last_base_pos[0]
#                ) / (self._time_step * self.num_action_repeat)
#     y_speed = (self.current_base_pos[1] - self.last_base_pos[1]
#                ) / (self._time_step * self.num_action_repeat)
#     z_speed = (self.current_base_pos[2] - self.last_base_pos[2]
#                ) / (self._time_step * self.num_action_repeat)

#     xy_speed = np.array([x_speed, y_speed])

#     along_speed = np.dot(xy_speed, self.target_vel_dir)
#     per_speed = xy_speed - along_speed * self.target_vel_dir

#     along_speed = np.clip(
#       along_speed, a_min=None, a_max=self.target_vel
#     )
#     along_reward = self.target_vel ** 2 - (
#       along_speed - self.target_vel
#     ) ** 2

#     forward_reward = along_reward - \
#       self.other_direction_penalty * (np.linalg.norm(per_speed) ** 2) - \
#       self.z_penalty * (z_speed ** 2)

#     return forward_reward

#   def _calc_reward_rotation(self):
#     env = self._env
#     pyb = self._get_pybullet_client()

#     rot_quat = env.robot.GetBaseOrientation()

#     if self.init_orientation is None:
#       return 0
#     # Norm of displacement vector
#     rot_reward = np.sum(
#       (self.init_orientation - np.array(rot_quat)) ** 2)  # * self.num_action_repeat
#     return rot_reward






















##################################################################################################################################
#### JUST COMMENT AS IS ##################################################
##########################################################################








# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
  inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class MoveForwardTask(object):
  """move forward task."""

  def __init__(
      self,
      z_constrain=False,
      move_forward_coeff=1,
      other_direction_penalty=0,
      z_penalty=0,
      orientation_penalty=0,
      time_step_s=0.01,
      num_action_repeat=10,
      height_fall_coeff=0.3,
      alive_reward=0.1,
      fall_reward=0,
      target_vel=None,
      check_contact=False,
      target_vel_dir=(1, 0),
      subgoal_reward=None
      # init_orientation=None,
  ):
    """Initializes the task."""
    self._draw_ref_model_alpha = 1.
    # self.energy_weight = -0.01
    self.energy_weight = -0.005
    self.move_forward_coeff = move_forward_coeff
    self._ref_model = -1
    self._alive_reward = alive_reward
    self.fall_reward = fall_reward
    self._time_step = time_step_s
    self.num_action_repeat = num_action_repeat
    self.z_constrain = z_constrain
    self.other_direction_penalty = other_direction_penalty
    self.z_penalty = z_penalty
    self.init_orientation = np.array([0, 0, 0, 1])
    self.orientation_penalty = orientation_penalty
    self.height_fall_coeff = height_fall_coeff
    self.target_vel = target_vel
    self.check_contact = check_contact
    # return
    self.target_vel_dir = np.array(target_vel_dir)
    self.subgoal_reward = subgoal_reward

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

    if self.subgoal_reward is not None:
      self.subgoal_trackers = np.ones(
        len(env._env_randomizers[-1].subgoal_ids),
        dtype=np.uint8
      )

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over."""
    del env
    env = self._env
    pyb = env._pybullet_client

    root_pos_sim, _ = pyb.getBasePositionAndOrientation(
      env.robot.quadruped)

    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

    rot_fall = rot_mat[-1] < 0.6
    height_fall = root_pos_sim[2] < self.height_fall_coeff
    if self.z_constrain:
      height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

    contact_done = False

    if self.check_contact:
      contacts = env._pybullet_client.getContactPoints(
        bodyA=env.robot.quadruped)
      for contact in contacts:
        if contact[2] is env._world_dict["ground"] or \
            ("terrain" in env._world_dict and
             contact[2] is env._world_dict["terrain"]):
          if contact[3] not in env.robot._foot_link_ids:
            contact_done = True
            break

      # contacts = env._pybullet_client.getContactPoints(bodyA=env.robot.quadruped)
      for contact in contacts:
        if contact[2] is not env._world_dict["ground"]:
          contact_done = True
          break
      speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
        (self._time_step * self.num_action_repeat)

      # contact_done = contact_done and np.linalg.norm(speed) <= 0.05
      contact_done = contact_done and np.linalg.norm(speed) <= 0.2  ############################################## ADDED THIS FOR RL
    done = height_fall or rot_fall or contact_done
    return done

  def reward(self, env):
    """Get the reward without side effects."""
    del env

    env = self._env
    # energy_reward = np.abs(
    #     np.dot(env.robot.GetMotorTorques(),
    #            env.robot.GetMotorVelocities())) * self._time_step

    energy_reward = np.dot(
      env.robot.GetMotorTorques(),
      env.robot.GetMotorTorques()
    ) * self._time_step

    move_forward_reward = self._calc_reward_root_velocity()
    alive_reward = self._alive_reward
    orientation_reward = self._calc_reward_rotation()

    reward = move_forward_reward * self.move_forward_coeff + \
      energy_reward * self.energy_weight - \
      self.orientation_penalty * orientation_reward + \
      alive_reward
    # print("Rew:{:.4f} Move Rew:{:.4f}, Ori Rew:{:.4f}, Eng Rew:{:.4f}".format(
    #   reward, move_forward_reward, orientation_reward, energy_reward))
    # print(move_forward_reward)
    done = self.done(env)
    if done:
      reward += self.fall_reward

    if self.subgoal_reward is not None:
      # self.last_base_pos = env.robot.GetBasePosition()
      # print(self.current_base_pos)
      dis = env._env_randomizers[-1].subgoal_centers - \
        self.current_base_pos[:2]
      dis = np.linalg.norm(dis, axis=1)

      contacted_ones = np.where(
        (dis < env._env_randomizers[-1].radius) * self.subgoal_trackers
      )[0]

      for contacted_idx in contacted_ones:
        self.subgoal_trackers[contacted_idx] = 0
        reward += self.subgoal_reward

        env.pybullet_client.changeVisualShape(
          env._env_randomizers[-1].subgoal_ids[contacted_idx],
          -1,
          rgbaColor=(1, 0.2, 0.2, 0)
        )
        # env.pybullet_client.removeBody(contacted_idx)

    return reward

  def _get_pybullet_client(self):
    """Get bullet client from the environment"""
    return self._env._pybullet_client

  def _calc_reward_root_velocity(self):
    """Get the root velocity reward."""
    env = self._env
    robot = env.robot
    sim_model = robot.quadruped

    pyb = self._get_pybullet_client()

    root_vel_sim, _ = pyb.getBaseVelocity(sim_model)
    root_vel_sim = np.array(root_vel_sim)

    x_speed = (self.current_base_pos[0] - self.last_base_pos[0]
               ) / (self._time_step * self.num_action_repeat)
    y_speed = (self.current_base_pos[1] - self.last_base_pos[1]
               ) / (self._time_step * self.num_action_repeat)
    z_speed = (self.current_base_pos[2] - self.last_base_pos[2]
               ) / (self._time_step * self.num_action_repeat)

    xy_speed = np.array([x_speed, y_speed])

    along_speed = np.dot(xy_speed, self.target_vel_dir)
    per_speed = xy_speed - along_speed * self.target_vel_dir

    along_speed = np.clip(
      along_speed, a_min=None, a_max=self.target_vel
    )
    along_reward = self.target_vel ** 2 - (
      along_speed - self.target_vel
    ) ** 2

    forward_reward = along_reward - \
      self.other_direction_penalty * (np.linalg.norm(per_speed) ** 2) - \
      self.z_penalty * (z_speed ** 2)

    return forward_reward

  def _calc_reward_rotation(self):
    env = self._env
    pyb = self._get_pybullet_client()

    rot_quat = env.robot.GetBaseOrientation()

    if self.init_orientation is None:
      return 0
    # Norm of displacement vector
    rot_reward = np.sum(
      (self.init_orientation - np.array(rot_quat)) ** 2)  # * self.num_action_repeat
    return rot_reward
