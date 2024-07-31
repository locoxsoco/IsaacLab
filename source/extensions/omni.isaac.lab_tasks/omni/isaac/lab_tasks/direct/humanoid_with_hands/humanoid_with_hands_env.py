# TODO: Check observation space
import numpy as np
import torch
import os
from collections.abc import Sequence

from gym import spaces

from omni.isaac.lab_assets.humanoid_with_hands import HUMANOID_WITH_HANDS_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_angle_axis, quat_rotate, quat_mul
from omni.isaac.lab_tasks.utils.amp.motion_lib_mpl import MotionLibMPL
from omni.isaac.lab_tasks.utils.amp.motion_lib_ip import MotionLibIP

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@configclass
class HumanoidWithHandsEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = (1.0 / 120.0) * decimation * 200
    action_scale = 1.0  # [N]
    num_actions = 54
    num_observations = 199
    num_states = 0 # This is useful for asymmetric actor-critic and defines the observation space for the critic.

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = HUMANOID_WITH_HANDS_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    upper_body_dof_name = "upper_body"
    lower_body_dof_name = "lower_body"
    right_hand_dof_name = "right_hand"
    left_hand_dof_name = "left_hand"
    key_body_names = ["right_palm", "left_palm", "right_foot", "left_foot"]
    contact_body_names = ["right_foot", "left_foot"]
    mocap_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 32, 33, 34, 58, 59, 60, 65, 66, 67, 9, 35, 61, 68, 10, 11, 12, 36, 37, 38, 62, 63, 64, 69, 70, 71, 13, 17, 24, 28, 39, 43, 50, 54, 14, 18, 21, 25, 29, 40, 44, 47, 51, 55, 15, 19, 22, 26, 30, 41, 45, 48, 52, 56, 16, 20, 23, 27, 31, 42, 46, 49, 53, 57]
    DOF_OFFSETS_MPL = [0,  3,  6,  9, 12, 15, 18,
                       19, 20 , 21, 22,
                       25, 28, 31, 34,
                       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    contact_bodies = ["right_foot", "left_foot"]
    termination_height = 0.15

    # reward scales
    rew_scale_speed = 0.7
    rew_scale_heading = 0.3

    # motion file
    motion_file = "Running_AMP_MPL.npy"

    # hands interactive prior files
    right_hand_ip_file = "right_hand_ip.npy"
    left_hand_ip_file = "left_hand_ip.npy"

    num_amp_obs_steps = 8
    num_pmp4setsip_obs_per_step = 199+ 75 +75

class HumanoidWithHandsEnv(DirectRLEnv):
    cfg: HumanoidWithHandsEnvCfg

    def __init__(self, cfg: HumanoidWithHandsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._upper_body_dof_idx, _ = self.humanoid_with_hands.find_joints(self.cfg.upper_body_dof_name)
        # self._lower_body_dof_idx, _ = self.humanoid_with_hands.find_joints(self.cfg.lower_body_dof_name)
        # self._right_hand_dof_idx, _ = self.humanoid_with_hands.find_joints(self.cfg.right_hand_dof_name)
        # self._left_hand_dof_idx, _ = self.humanoid_with_hands.find_joints(self.cfg.left_hand_dof_name)
        self._key_body_ids, _ = self.humanoid_with_hands.find_bodies(self.cfg.key_body_names)
        self._key_body_ids = to_torch(self._key_body_ids, device=self.device, dtype=torch.long)
        self._contact_body_ids, _ = self.humanoid_with_hands.find_bodies(self.cfg.contact_body_names)
        # self._contact_body_ids = to_torch(self._contact_body_ids, device=self.device, dtype=torch.long)
        self._actuator_joints_ids = self.humanoid_with_hands.actuators["body"].joint_indices
        self.action_scale = self.cfg.action_scale

        self._humanoid_root_states = self.humanoid_with_hands.data.default_root_state
        self.root_pos = self.humanoid_with_hands.data.default_root_state[:, 0:3]
        self.root_rot = self.humanoid_with_hands.data.default_root_state[:, 3:7]
        self.root_vel = self.humanoid_with_hands.data.default_root_state[:, 7:10]
        self._rigid_body_pos = self.humanoid_with_hands.data.body_pos_w
        self.joint_limits = self.humanoid_with_hands.data.joint_limits
        self.joint_pos = self.humanoid_with_hands.data.joint_pos
        self.joint_vel = self.humanoid_with_hands.data.joint_vel
        self.target_pos = torch.tensor([25.0, 0.0], device=self.device)
        self.DOF_OFFSETS_MPL = self.cfg.DOF_OFFSETS_MPL

        self._build_pd_action_offset_scale()

        motion_file = self.cfg.motion_file
        motion_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./" + motion_file,
        )
        self._load_motion(motion_file_path)

        right_hand_ip_file = self.cfg.right_hand_ip_file
        right_hand_ip_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./" + right_hand_ip_file,
        )
        left_hand_ip_file = self.cfg.left_hand_ip_file
        left_hand_ip_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./" + left_hand_ip_file,
        )
        self._load_hands_interactive_prior(right_hand_ip_file_path, left_hand_ip_file_path)

        self._local_root_obs = False
        self._termination_height = 0.15

        self.mocap_indices = self.cfg.mocap_indices
        
        self.simulation_cfg = self.cfg.sim

        self._num_pmp4setsip_obs_steps = self.cfg.num_amp_obs_steps
        self._num_pmp4setsip_obs_per_step = self.cfg.num_pmp4setsip_obs_per_step

        self.num_pmp4setsip_obs = (
            self._num_pmp4setsip_obs_steps * self._num_pmp4setsip_obs_per_step
        )

        self._pmp4setsip_obs_space = spaces.Box(
            np.ones(self.num_pmp4setsip_obs) * -np.Inf,
            np.ones(self.num_pmp4setsip_obs) * np.Inf,
        )
        
        self._pmp4setsip_obs_buf = torch.zeros(
            (
                self.num_envs,
                self._num_pmp4setsip_obs_steps,
                self._num_pmp4setsip_obs_per_step,
            ),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_pmp4setsip_obs_buf = self._pmp4setsip_obs_buf[:, 0]
        self._hist_pmp4setsip_obs_buf = self._pmp4setsip_obs_buf[:, 1:]

        self._curr_pmp4setsip_obs_buf[:, 199+59:199+62] = 100.0
        self._curr_pmp4setsip_obs_buf[:, 199+75+59:199+75+62] = 100.0

        self._hist_pmp4setsip_obs_buf[:, :, 199+59:199+62] = 1000.0
        self._hist_pmp4setsip_obs_buf[:, :, 199+75+59:199+75+62] = 1000.0

        self._pmp4setsip_obs_demo_buf = None
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLibMPL(
            motion_file=motion_file,
            # TODO: should call humanoid_num_dof
            # num_dofs=self.humanoid_num_dof,
            num_dofs=72,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
        )
        return

    def _load_hands_interactive_prior(self, right_hand_ip_file, left_hand_ip_file):
        self._motion_right_hand_ip = MotionLibIP(
            motion_file=right_hand_ip_file,
            device=self.device,
        )
        self._motion_left_hand_ip = MotionLibIP(
            motion_file=left_hand_ip_file,
            device=self.device,
        )
        return

    def _setup_scene(self):
        self.humanoid_with_hands = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["humanoid_with_hands"] = self.humanoid_with_hands
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _get_observations(self) -> dict:
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids]
        obs = build_pmp4setsip_observations(
            self._humanoid_root_states,
            self.joint_pos,
            self.joint_vel,
            key_body_pos,
            self._local_root_obs,
        )
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
     total_reward = compute_rewards(
         self.cfg.rew_scale_speed,
         self.cfg.rew_scale_heading,
         self.root_pos,
         self.root_rot,
         self.root_vel,
         self.target_pos,
     )
     return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # # TODO: Test if necessary -> It was necessary
        self._rigid_body_pos = self.humanoid_with_hands.data.body_pos_w
        # self.joint_pos = self.humanoid_with_hands.data.joint_pos
        # self.joint_vel = self.humanoid_with_hands.data.joint_vel

        reset_buf, self._terminate_buf[:] = compute_humanoid_reset(
            self.episode_length_buf,
            self._contact_body_ids,
            self._rigid_body_pos,
            self.max_episode_length,
            self._termination_height,
        )
        return reset_buf, self._terminate_buf
    
    def _set_env_state(
        self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 0:3] += self.scene.env_origins[env_ids]
        self._humanoid_root_states[env_ids, 2] += 0.30
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self.joint_pos[env_ids] = dof_pos
        self.joint_vel[env_ids] = dof_vel

        self.humanoid_with_hands.write_root_pose_to_sim(self._humanoid_root_states[env_ids, :7], env_ids)
        self.humanoid_with_hands.write_root_velocity_to_sim(self._humanoid_root_states[env_ids, 7:], env_ids)
        self.humanoid_with_hands.write_joint_state_to_sim(dof_pos, dof_vel, self.mocap_indices, env_ids)
        return
    
    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        # motion_right_hand_ip_ids = self._motion_right_hand_ip.sample_motions(num_envs)
        # motion_left_hand_ip_ids = self._motion_left_hand_ip.sample_motions(num_envs)

        motion_times = self._motion_lib.sample_time(motion_ids)
        # motion_right_hand_ip_times = self._motion_lib.sample_time(motion_ids)
        # motion_left_hand_ip_times = self._motion_lib.sample_time(motion_ids)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )

        # print(f'root_pos[133]: {root_pos[133]}')

        # # Here should update position of assets interacting with object too with motion_ids
        # asset_pos, asset_rot = self.get_bouldering_wall_state(motion_ids)

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        # self._reset_ref_env_ids = env_ids
        # self._reset_ref_motion_ids = motion_ids
        # self._reset_ref_motion_times = motion_times

        # self._reset_ref_right_hand_ip_motion_ids = motion_right_hand_ip_ids
        # self._reset_ref_right_hand_ip_motion_times = motion_right_hand_ip_times

        # self._reset_ref_left_hand_ip_motion_ids = motion_left_hand_ip_ids
        # self._reset_ref_left_hand_ip_motion_times = motion_left_hand_ip_times
        return
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        self._terminate_buf[env_ids] = 0
        if env_ids is None:
            env_ids = self.humanoid_with_hands._ALL_INDICES
        super()._reset_idx(env_ids)
        # print(f'env_ids: {env_ids}')

        self._reset_ref_state_init(env_ids)
    
    def _build_pd_action_offset_scale(self):
        num_joints = len(self.DOF_OFFSETS_MPL) - 1
        
        lim_low = self.joint_limits[0,:,0].cpu().numpy()
        lim_high = self.joint_limits[0,:,1].cpu().numpy()

        for j in range(num_joints):
            dof_offset = self.DOF_OFFSETS_MPL[j]
            dof_size = self.DOF_OFFSETS_MPL[j + 1] - self.DOF_OFFSETS_MPL[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset[self._actuator_joints_ids] + self._pd_action_scale[self._actuator_joints_ids] * action
        return pd_tar

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        self.cur_targets = self._action_to_pd_targets(self.actions)
    
    def _apply_action(self) -> None:
        # print(f'cur_targets: {self.cur_targets[0]}')
        self.humanoid_with_hands.set_joint_effort_target(self.cur_targets, joint_ids=self._actuator_joints_ids)
    
    @property
    def pmp4setsip_observation_space(self):
        return self._pmp4setsip_obs_space
    
    def fetch_pmp4setsip_obs_demo(self, num_samples):
        return self.task.fetch_pmp4setsip_obs_demo(num_samples)

    def fetch_pmp4setsip_obs_demo(self, num_samples):
        # TODO: Check if dt should be (1/120) or (1/60) => In isaacgymenvs is 0.0332 (1/30)
        # Now dt is set as (1/60)
        dt = self.simulation_cfg.dt * self.simulation_cfg.render_interval
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._pmp4setsip_obs_demo_buf is None:
            self._build_pmp4setsip_obs_demo_buf(num_samples)
        else:
            assert self._pmp4setsip_obs_demo_buf.shape[0] == num_samples

        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_pmp4setsip_obs_steps]
        )
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_pmp4setsip_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        pmp4setsip_obs_demo = build_pmp4setsip_observations(
            root_states, dof_pos, dof_vel, key_pos, self._local_root_obs
        )

        right_motion_ids = self._motion_right_hand_ip.sample_motions(num_samples)
        right_motion_times0 = self._motion_right_hand_ip.sample_time(right_motion_ids)
        right_motion_ids = np.tile(
            np.expand_dims(right_motion_ids, axis=-1),
            [1, self._num_pmp4setsip_obs_steps],
        )
        right_motion_times = np.expand_dims(right_motion_times0, axis=-1)
        right_time_steps = -dt * np.arange(0, self._num_pmp4setsip_obs_steps)
        right_motion_times = right_motion_times + right_time_steps

        right_motion_ids = right_motion_ids.flatten()
        right_motion_times = right_motion_times.flatten()
        right_state_and_actions = (
            self._motion_right_hand_ip.get_motion_state_and_actions(
                right_motion_ids, right_motion_times
            )
        )

        left_motion_ids = self._motion_left_hand_ip.sample_motions(num_samples)
        left_motion_times0 = self._motion_left_hand_ip.sample_time(left_motion_ids)
        left_motion_ids = np.tile(
            np.expand_dims(left_motion_ids, axis=-1),
            [1, self._num_pmp4setsip_obs_steps],
        )
        left_motion_times = np.expand_dims(left_motion_times0, axis=-1)
        left_time_steps = -dt * np.arange(0, self._num_pmp4setsip_obs_steps)
        left_motion_times = left_motion_times + left_time_steps

        left_motion_ids = left_motion_ids.flatten()
        left_motion_times = left_motion_times.flatten()
        left_state_and_actions = self._motion_left_hand_ip.get_motion_state_and_actions(
            left_motion_ids, left_motion_times
        )

        obs = torch.cat(
            [pmp4setsip_obs_demo, right_state_and_actions, left_state_and_actions],
            dim=-1,
        )

        self._pmp4setsip_obs_demo_buf[:] = obs.view(self._pmp4setsip_obs_demo_buf.shape)

        pmp4setsip_obs_demo_flat = self._pmp4setsip_obs_demo_buf.view(
            -1, self.num_pmp4setsip_obs
        )
        return pmp4setsip_obs_demo_flat
    
    def _build_pmp4setsip_obs_demo_buf(self, num_samples):
        self._pmp4setsip_obs_demo_buf = torch.zeros(
            (
                num_samples,
                self._num_pmp4setsip_obs_steps,
                self._num_pmp4setsip_obs_per_step,
            ),
            device=self.device,
            dtype=torch.float,
        )
        return
    
    def _update_hist_pmp4setsip_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._pmp4setsip_obs_buf.shape[1] - 1)):
                self._pmp4setsip_obs_buf[:, i + 1] = self._pmp4setsip_obs_buf[:, i]
        else:
            for i in reversed(range(self._pmp4setsip_obs_buf.shape[1] - 1)):
                self._pmp4setsip_obs_buf[env_ids, i + 1] = self._pmp4setsip_obs_buf[
                    env_ids, i
                ]
        return

    def _compute_pmp4setsip_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_pmp4setsip_obs_buf[:, 0:199] = build_pmp4setsip_observations(
                self._humanoid_root_states,
                self.humanoid_with_hands.data.joint_pos,
                self.humanoid_with_hands.data.joint_vel,
                key_body_pos,
                self._local_root_obs,
            )
            self._curr_pmp4setsip_obs_buf[:, 199:199+75] = 10000.0
            self._curr_pmp4setsip_obs_buf[:, 199+75:199+75+75] = 10000.0
        else:
            self._curr_pmp4setsip_obs_buf[env_ids, 0:199] = (
                build_pmp4setsip_observations(
                    self._humanoid_root_states[env_ids],
                    self.humanoid_with_hands.data.joint_pos[env_ids],
                    self.humanoid_with_hands.data.joint_vel[env_ids],
                    key_body_pos[env_ids],
                    self._local_root_obs,
                )
            )
            self._curr_pmp4setsip_obs_buf[:, 199:199+75] = 10000.0
            self._curr_pmp4setsip_obs_buf[:, 199+75:199+75+75] = 10000.0
        return
    
    def post_physics_step(self):
        self.extras["terminate"] = self._terminate_buf
        self._update_hist_pmp4setsip_obs()
        self._compute_pmp4setsip_observations()

        pmp4setsip_obs_flat = self._pmp4setsip_obs_buf.view(
            -1, self.num_pmp4setsip_obs
        )
        self.extras["pmp4setsip_obs"] = pmp4setsip_obs_flat

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def calc_heading(
    q: torch.Tensor
):
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(
    q: torch.Tensor
):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

@torch.jit.script
def calc_heading_quat_inv(
    q: torch.Tensor
):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

@torch.jit.script
def normalize_angle(
    x: torch.Tensor,
):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def exp_map_to_angle_axis(
    exp_map: torch.Tensor,
):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = angle > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis

@torch.jit.script
def exp_map_to_quat(
    exp_map: torch.Tensor,
):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

@torch.jit.script
def quat_to_tan_norm(
    q: torch.Tensor,
):
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

@torch.jit.script
def dof_to_obs_mpl(
    pose: torch.Tensor,
):
    dof_obs_size = 102
    # dof_offsets = [abdomen_xyz, neck_xyz, right_shoulder_xyz, right_elbow_y
    # right_wrist_puf, right_thumb_ampd, right_index_ampd, right_middle_mpd,
    # right_ring_ampd, right_pinky_ampd, left_shoulder_xyz, left_elbow_y,
    # left_wrist_puf, left_thumb_ampd, left_index_ampd, left_middle_mpd,
    # left_ring_ampd, left_pinky_ampd, right_hip_xyz, right_knee_x,
    # right_ankle_xyz, left_hip_xyz, left_knee_x, left_ankle_xyz]
    # body_ids = [1, 2, 3, 4,
    # 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
    # 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    # 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
    # 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    # 51, 52, 53, 54]
    dof_offsets = [0,  3,  6,  9, 10,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   25, 26, 27, 28, 29, 30, 31, 32, 35, 36,
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                   51, 52, 53, 54, 55, 56, 57, 58, 61, 62,
                   65, 68, 69, 72]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(
            dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def build_pmp4setsip_observations(
    root_states: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
    local_root_obs: bool,
):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs_mpl(dof_pos)

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_rewards(
    rew_scale_speed: float,
    rew_scale_heading: float,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_vel: torch.Tensor,
    tar_pos: torch.Tensor,
):
    speed_err_scale = 0.25
    tar_speed = 4.0

    tar_vel = tar_pos[..., 0:2]
    tar_vel = torch.nn.functional.normalize(tar_vel, dim=-1)
    
    char_speed = torch.sum(tar_vel * root_vel[..., 0:2], dim=-1)
    char_speed_err = tar_speed - char_speed
    speed_reward = torch.exp(-speed_err_scale * (char_speed_err * char_speed_err))
    speed_mask = speed_reward <= 0
    speed_reward[speed_mask] = 0

    heading_rot = calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_vel[..., 0:2] * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = rew_scale_speed * speed_reward + rew_scale_heading * facing_reward
    # print(f'speed_reward[0]: {speed_reward[0]} facing_reward[0]: {facing_reward[0]} reward[0]: {reward[0]}')

    return reward

@torch.jit.script
def compute_humanoid_reset(
    episode_length_buf: torch.Tensor,
    contact_body_ids: list[int],
    rigid_body_pos: torch.Tensor,
    max_episode_length: float,
    termination_height: float,
):
    terminated = episode_length_buf >= max_episode_length - 1

    body_height = rigid_body_pos[..., 2]
    fall_height = body_height < termination_height
    fall_height[:, contact_body_ids] = False
    fall_height = torch.any(fall_height, dim=-1)

    has_fallen = fall_height

    # first timestep can sometimes still have nonzero contact forces
    # so only check after first couple of steps
    has_fallen *= (episode_length_buf > 1)
    # reset = torch.where(has_fallen, torch.ones_like(terminated), terminated)

    return has_fallen, terminated
