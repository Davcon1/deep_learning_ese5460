import argparse
import sys
import pathlib
import os 
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

parser = argparse.ArgumentParser(description="Collect trajectories.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g., 'entity/project/run_id' or 'entity/project/run_id/model_xxx.pt')")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--rollout_length", type=int, default=250)
parser.add_argument("--output", type=str, default="trajectories.pt")
parser.add_argument("--use_noise", action="store_true", default=False)
parser.add_argument("--motion_file", type=str, default = None, help="Path to motion file.")  # Add this
parser.add_argument("--motion_name", type=str, default=None, help="Name of motion artifact to use (e.g., 'walk3_subject1')")
parser.add_argument("--min_traj_length", type=int, default=150)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from rsl_rl.modules import ActorCritic
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks  # noqa: F401

def parse_observations(obs):
    '''
    Used to convert the raw observations into organized observations that we can use
    '''
    parsed = {
        'target_joint_pos': obs[..., 0:29],
        'target_joint_vel': obs[..., 29:58],  
        'pose_error_pos': obs[..., 58:61],
        'pose_error_ori': obs[..., 61:67],
        'base_lin_vel': obs[..., 67:70],
        'base_ang_vel': obs[..., 70:73],
        'joint_pos': obs[..., 73:102],
        'joint_vel': obs[..., 102:131],
        'last_action': obs[..., 131:160]
    }
    
    return parsed

def load_policy(model_path, obs_dim, action_dim, device):
    model_params = torch.load(model_path, map_location=device, weights_only=False)
    model_state = model_params["model_state_dict"]
    
    # infer what the critic dimensionality is
    critic_input_dim = model_state['critic.0.weight'].shape[1]
    
    print(f"Actor obs dim: {obs_dim}")
    print(f"Critic obs dim: {critic_input_dim}")
    print(f"Action dim: {action_dim}")

    # define the actor policy before loading in the saved checkpoint
    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=critic_input_dim,
        num_actions=action_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)

    policy.load_state_dict(model_state)
    policy.eval()
    
    return policy

def load_policy_with_normalizer(checkpoint_path, env, agent_cfg, device):
    '''
    ensure that the normalizer is used with the PPO.
    '''
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=device)
    
    return policy, ppo_runner

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def generate_trajectories(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg.scene.num_envs = args_cli.num_envs

    checkpoint_path = args_cli.checkpoint
    motion_file_path = args_cli.motion_file
        
    if args_cli.wandb_path:
        import wandb

        print(f"Loading from wandb: {args_cli.wandb_path}")
        
        api = wandb.Api()
        run_path = args_cli.wandb_path
        
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        
        wandb_run = api.run(run_path)
        
        # download checkpoint if not provided locally
        if checkpoint_path is None:
            files = [file.name for file in wandb_run.files() if "model" in file.name]
            
            if "model" in args_cli.wandb_path:
                file = args_cli.wandb_path.split("/")[-1]
            else:
                file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            
            os.makedirs("./logs/rsl_rl/temp", exist_ok=True)
            wandb_file = wandb_run.file(str(file))
            wandb_file.download("./logs/rsl_rl/temp", replace=True)
            checkpoint_path = f"./logs/rsl_rl/temp/{file}"
            print(f:"Checkpoint downloaded to: {checkpoint_path}")

    # download the wandb
    if motion_file_path is None:
        if hasattr(args_cli, 'motion_name') and args_cli.motion_name:
            import wandb
            
            # build the full artifact path
            artifact_name = f"pattheboy135/csv_to_npz/{args_cli.motion_name}"
            if ":" not in artifact_name:
                artifact_name += ":latest"
            
            print(f"Downloading artifact: {artifact_name}")
            api = wandb.Api()
            art = api.artifact(artifact_name, type='motions')
            motion_file_path = str(pathlib.Path(art.download()) / "motion.npz")
            print(f"Motion file downloaded to: {motion_file_path}")
        else:
            raise ValueError("--motion_name required when no --motion_file provided")
            
    if checkpoint_path is None:
        raise ValueError("Checkpoint required: provide --checkpoint or --wandb_path")
    if motion_file_path is None:
        raise ValueError("Motion file required: provide --motion_file or --motion_name")
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
        env_cfg.commands.motion.motion_file = motion_file_path
        print(f"Using motion file: {motion_file_path}")
        
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap first
    env = RslRlVecEnvWrapper(env)

    # then get dimensions from the wrapper
    obs_dim = env.num_obs
    action_dim = env.num_actions
    num_envs = env.num_envs

    # load the policy with the normalizer and ppo
    policy, ppo_runner = load_policy_with_normalizer(
        checkpoint_path, env, agent_cfg, device
    )

    # initialize a list for saving all the trajectories
    collected_states = []

    num_rollouts = 0
    min_trajectory_length = args_cli.min_traj_length
    while num_rollouts < args_cli.num_rollouts:
        obs, _ = env.get_observations()
        env_trajectories = [[] for _ in range(num_envs)]
        env_failed = [False] * num_envs
        first_step = True

        with torch.inference_mode():
            for step in range(args_cli.rollout_length):
                # get the action from the observation
                action = policy(obs)

                # find the parsed observation
                parsed_obs = parse_observations(obs)
                for env_idx in range(num_envs):
                    if not env_failed[env_idx]:
                        # save the environment trajectories
                        env_trajectories[env_idx].append({
                            "obs": obs[env_idx].cpu().clone(),
                            "target_action": action[env_idx].cpu().clone(),
                            "target_joint_pos": parsed_obs['target_joint_pos'][env_idx].cpu().clone(),
                            "target_joint_vel": parsed_obs['target_joint_vel'][env_idx].cpu().clone(),
                            "pose_error_pos": parsed_obs['pose_error_pos'][env_idx].cpu().clone(),
                            "pose_error_ori": parsed_obs['pose_error_ori'][env_idx].cpu().clone(),
                            "base_lin_vel": parsed_obs['base_lin_vel'][env_idx].cpu().clone(),
                            "base_ang_vel": parsed_obs['base_ang_vel'][env_idx].cpu().clone(),
                            "joint_pos": parsed_obs['joint_pos'][env_idx].cpu().clone(),
                            "joint_vel": parsed_obs['joint_vel'][env_idx].cpu().clone(),
                            "last_action": parsed_obs['last_action'][env_idx].cpu().clone(),
                        })
                
                obs, _, dones, _ = env.step(action)

                if first_step:
                    first_step = False
                    for env_idx in range(num_envs):
                        if dones[env_idx]:
                            env_trajectories[env_idx] = [] 
                    continue

                for env_idx in range(num_envs):
                    if dones[env_idx] and not env_failed[env_idx]:
                        env_failed[env_idx] = True
                        trajectory_length = len(env_trajectories[env_idx])
                        if trajectory_length >= min_trajectory_length:
                            collected_states.extend(env_trajectories[env_idx])
                            num_rollouts += 1
                        print(trajectory_length)

        for env_idx in range(num_envs):
            if not env_failed[env_idx]:
                trajectory_length = len(env_trajectories[env_idx])
                if trajectory_length >= min_trajectory_length:
                    collected_states.extend(env_trajectories[env_idx])
                    num_rollouts += 1
                print(trajectory_length)

        print(f"Current rollouts at {num_rollouts}")

    save_dataset(collected_states, args_cli.output)
    env.close()

def save_dataset(dataset, save_path):
    ''' 
    saved the collected dataset into the save path
    '''    
    # Stack all data
    processed = {
        # Raw data
        'obs': torch.stack([d['obs'] for d in dataset]),
        'target_action': torch.stack([d['target_action'] for d in dataset]),
        
        # Parsed observation components
        'target_joint_pos': torch.stack([d['target_joint_pos'] for d in dataset]),
        'target_joint_vel': torch.stack([d['target_joint_vel'] for d in dataset]),
        'pose_error_pos': torch.stack([d['pose_error_pos'] for d in dataset]),
        'pose_error_ori': torch.stack([d['pose_error_ori'] for d in dataset]),
        'base_lin_vel': torch.stack([d['base_lin_vel'] for d in dataset]),
        'base_ang_vel': torch.stack([d['base_ang_vel'] for d in dataset]),
        'joint_pos': torch.stack([d['joint_pos'] for d in dataset]),
        'joint_vel': torch.stack([d['joint_vel'] for d in dataset]),
        'last_action': torch.stack([d['last_action'] for d in dataset]),
    }
    
    torch.save(processed, save_path)

if __name__ == "__main__":
    generate_trajectories()
    simulation_app.close()