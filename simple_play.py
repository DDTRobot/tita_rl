from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO

import cv2
import os

from isaacgym import gymapi
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from configs import *
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR

from PIL import Image as im



from configs.tita_flat_config import TitaFlatCfg, TitaFlatCfgPPO
from configs.tita_rough_config import TitaRoughCfg, TitaRoughCfgPPO
from configs.tita_single_stand_config import TitaSingleStandCfg, TitaSingleStandCfgPPO
from envs.no_constrains_legged_robot import Tita
from envs import *
from export_policy_as_onnx  import *
import argparse

from utils import webviewer



def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def play_on_constraint_policy_runner(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =1# min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.domain_rand.push_robots = False
    #env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.control.use_filter = True
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticRMA = actor_critic_class(env.cfg.env.n_proprio,
                                                      env.cfg.env.n_scan,
                                                      env.num_obs,
                                                      env.cfg.env.n_priv_latent,
                                                      env.cfg.env.history_len,
                                                      env.num_actions,
                                                      **policy_cfg_dict)
    # print(policy)
    # model_dict = torch.load(os.path.join(ROOT_DIR, 'model_4000_phase2_hip.pt'))
    # model_dict = torch.load(os.path.join(ROOT_DIR, 'tita_examptita_rl/logs/tita_constraint/Apr06_22-05-10_test_barlowtwins_feetcontactle_10000.pt'))
    model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/tita_constraint/Apr08_17-57-31_test_barlowtwins_feetcontact/model_10000.pt'))
    policy.load_state_dict(model_dict['model_state_dict'])
    policy = policy.to(env.device)
    if EXPORT_POLICY:
        policy.save_torch_jit_policy('model.pt',env.device)
    if WEB_VIEWER:
        web_viewer = webviewer.WebViewer()
        web_viewer.setup(env)


    # set rgba camera sensor for debug and doudle check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    video_duration = 40
    num_frames = int(video_duration / env.dt)

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0


    for i in range(num_frames):

        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        env.commands[:,0] = 0
        env.commands[:,1] = 0
        env.commands[:,2] = 0
        env.commands[:,3] = 0
        actions = policy.act_teacher(obs)

        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)
        if WEB_VIEWER:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        #print(env.foot_positions[:,0,2])
        # print(env.contact_forces[:, env.feet_indices, :])
        # print(env.root_states[:,2])
def play_no_constraint_policy_runner(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        export_policy_as_onnx(args)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)




if __name__ == '__main__':
    EXPORT_POLICY = False#True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    WEB_VIEWER = True
    # Register tasks
    task_registry.register("tita_flat", Tita, TitaFlatCfg(), TitaFlatCfgPPO())
    task_registry.register("tita_rough", Tita, TitaRoughCfg(), TitaRoughCfgPPO())
    task_registry.register("tita_constraint", LeggedRobot, TitaConstraintRoughCfg(), TitaConstraintRoughCfgPPO())
    task_registry.register("single_stand", SingleStand, TitaSingleStandCfg(), TitaSingleStandCfgPPO())

    args = get_args()
    args.task = "single_stand"
    # Get the selected task name

    task_name = args.task
    args.headless = True
    # Run the corresponding function based on the selected task
    if task_name in ["tita_flat", "tita_rough"]:
        play_no_constraint_policy_runner(args)
    # elif task_name == "tita_constraint":
    else:
        play_on_constraint_policy_runner(args)
