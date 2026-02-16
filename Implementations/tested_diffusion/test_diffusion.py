import argparse
import pathlib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vae'))
sys.path.append(os.path.dirname(__file__))

from model import VAE
from diffusion_model import DiffusionTransformer
from scheduler import Diffusion_Scheduler
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test diffusion model in simulation with video recording.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE checkpoint")
parser.add_argument("--diffusion_path", type=str, required=True, help="Path to diffusion checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--episode_length", type=int, default=500)
parser.add_argument("--motion_name", type=str, required=True)
parser.add_argument("--denoising_steps", type=int, default=20, help="Number of denoising steps at inference")
parser.add_argument("--history_length", type=int, default=4)
parser.add_argument("--prediction_horizon", type=int, default=16)
parser.add_argument("--video_dir", type=str, default="./videos", help="Directory to save videos")
parser.add_argument("--video_length_threshold", type=int, default=100, help="Min steps to save video")

# Add Isaac Lab launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from collections import deque
import numpy as np

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_observations(obs):
    return {
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


class DiffusionController:
    def __init__(self, vae, diffusion_model, scheduler, history_length, prediction_horizon):
        self.vae = vae
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        
        # State and latent history buffers
        self.state_history = None
        self.latent_history = None
        self.num_envs = None
        
    def reset(self, num_envs):
        '''
        used to reset the buffers
        '''
        self.num_envs = num_envs
        self.state_history = []
        self.latent_history = []
    
    def get_current_state(self, parsed_obs):
        '''
        find the current state in each environment
        '''
        state = torch.cat([
            parsed_obs['joint_pos'],
            parsed_obs['joint_vel'],
            parsed_obs['base_lin_vel'],
            parsed_obs['base_ang_vel']
        ], dim=-1)
        return state
    
    def get_vae_latent(self, parsed_obs):
        '''
        find what the latent vector is by passing through the vae
        '''
        joint_states = torch.cat([
            parsed_obs['target_joint_pos'],
            parsed_obs['target_joint_vel']
        ], dim=-1)
        
        anchor_pose = torch.cat([
            parsed_obs['base_lin_vel'],
            parsed_obs['base_ang_vel']
        ], dim=-1)
        
        conditioning = torch.cat([
            parsed_obs['joint_pos'],
            parsed_obs['joint_vel'],
            parsed_obs['pose_error_pos'],
            parsed_obs['pose_error_ori'],
            parsed_obs['last_action']
        ], dim=-1)
        
        with torch.no_grad():
            mean, logvar, z, action = self.vae(joint_states, anchor_pose, conditioning)
        
        return z, action
    
    def predict_trajectory(self):
        '''
        predict what the trajectory should be like given the history
        '''
        
        history_states = torch.stack(self.state_history, dim=1)
        history_latents = torch.stack(self.latent_history, dim=1)
        
        # randomly generate future states and future latents that we'll pass into denoising steps
        future_states = torch.randn(
            self.num_envs, self.prediction_horizon, 64, 
            device=device
        )
        future_latents = torch.randn(
            self.num_envs, self.prediction_horizon, 32, 
            device=device
        )
        
        # add the historical states and the future states together (history states includes the current state)
        full_states = torch.cat([history_states, future_states], dim=1)
        full_latents = torch.cat([history_latents, future_latents], dim=1)
        
        total_timesteps = self.history_length + 1 + self.prediction_horizon
        
        # for each of the different steps of the scheduler
        for k in reversed(range(self.scheduler.steps)):
            # fill with the current timestep and the latent timesteps (since we assume that we are denoising from the same timestep for each future timestep)
            state_timesteps = torch.full(
                (self.num_envs, total_timesteps), 
                k, 
                dtype=torch.long, 
                device=device
            )
            latent_timesteps = torch.full(
                (self.num_envs, total_timesteps), 
                k, 
                dtype=torch.long, 
                device=device
            )

            # combine the state and latent timesteps generated
            diffusion_timesteps = torch.zeros(
                self.num_envs, total_timesteps * 2, 
                dtype=torch.long, 
                device=device
            )
            diffusion_timesteps[:, 0::2] = state_timesteps
            diffusion_timesteps[:, 1::2] = latent_timesteps

            with torch.no_grad():
                pred_states, pred_latents = self.diffusion_model(
                    full_states, full_latents, diffusion_timesteps
                )
            
            # generate the schedule states. Tested both with inpainting and without inpainting
            # Found that generating without inpainting actually worked better. Would explore more as to 
            # why that might be the case
            full_states = self.scheduler.step_per_token(
                pred_states, 
                state_timesteps, 
                full_states,
                inpainting_mask=None,
                clean_data=None
            )
            full_latents = self.scheduler.step_per_token(
                pred_latents, 
                latent_timesteps, 
                full_latents,
                inpainting_mask=None,
                clean_data=None
            )
        
        current_latent = full_latents[:, self.history_length]
        
        return current_latent
    
    def step(self, obs):
        '''
        used predict_trajectory to take a single action.
        '''
        parsed_obs = parse_observations(obs)
        
        # get the current state and subsequently the latent from the parsed observation
        current_state = self.get_current_state(parsed_obs)
        z_encoder, vae_action = self.get_vae_latent(parsed_obs)

        # save the current state and latent to state and latent history
        self.state_history.append(current_state)
        self.latent_history.append(z_encoder)

        if len(self.state_history) > self.history_length + 1:
            self.state_history.pop(0)
            self.latent_history.pop(0)
        
        if len(self.state_history) == self.history_length + 1:
            z_diffusion = self.predict_trajectory()
            
            conditioning = torch.cat([
                parsed_obs['joint_pos'],
                parsed_obs['joint_vel'],
                parsed_obs['pose_error_pos'],
                parsed_obs['pose_error_ori'],
                parsed_obs['last_action']
            ], dim=-1).to(device)
            
            # predict the next action using the 
            with torch.no_grad():
                action_diffusion = self.vae.decoder(z_diffusion, conditioning)
            
            return action_diffusion
        
        #if not enough history yet, then return the VAE action instead.
        return vae_action


class VideoRecorder:
    '''
    video recording module
    '''
    
    def __init__(self, video_dir, fps=30):
        self.video_dir = video_dir
        self.fps = fps
        self.frames = []
        os.makedirs(video_dir, exist_ok=True)
        
    def reset(self):
        self.frames = []
    
    def add_frame(self, frame):
        if frame is not None:
            # to handle different frame formats
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            self.frames.append(frame)
    
    def save(self, filename):
        if len(self.frames) == 0:
            print(f"No frames to save for {filename}")
            return False
            
        filepath = os.path.join(self.video_dir, filename)
        
        try:
            import imageio
            imageio.mimsave(filepath, self.frames, fps=self.fps)
            print(f"Saved video: {filepath} ({len(self.frames)} frames)")
            return True
        except Exception as e:
            print(f"Failed to save video: {e}")
            # Try alternative method with ffmpeg
            try:
                import subprocess
                import tempfile
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, frame in enumerate(self.frames):
                        frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                        imageio.imwrite(frame_path, frame)
                    
                    cmd = [
                        'ffmpeg', '-y', '-framerate', str(self.fps),
                        '-i', os.path.join(tmpdir, 'frame_%06d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        filepath
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"Saved video (ffmpeg fallback): {filepath}")
                    return True
            except Exception as e2:
                print(f"FFmpeg fallback also failed: {e2}")
                return False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def test_diffusion_policy(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    
    # setup environment
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # configure viewer for video recording
    env_cfg.viewer.resolution = (640, 480)
    # Adjust camera position for better view (tune these as needed)
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    
    # Load the motion file
    if args_cli.motion_name:
        import wandb
        api = wandb.Api()
        artifact_name = f"pattheboy135/csv_to_npz/{args_cli.motion_name}:latest"
        art = api.artifact(artifact_name, type='motions')
        motion_file_path = str(pathlib.Path(art.download()) / "motion.npz")
        
        if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'motion'):
            env_cfg.commands.motion.motion_file = motion_file_path
            print(f"Using motion: {args_cli.motion_name}")
    
    # Create environment with rgb_array render mode for video capture
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    unwrapped_env = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    num_envs = env.num_envs
    
    # Create video recorder
    video_recorder = VideoRecorder(args_cli.video_dir, fps=30)
    
    # Load the VAE and the checkpoint in
    vae = VAE(64, 29, 32, 96).to(device)
    vae_checkpoint = torch.load(args_cli.vae_path)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # Load the diffusion model
    diffusion_model = DiffusionTransformer(
        embedding_dims=512,
        history_length=args_cli.history_length,
        prediction_horizon=args_cli.prediction_horizon,
        state_dim=64,
        latent_dim=32,
        attention_heads=8,
        num_layers=6
    ).to(device)
    diffusion_checkpoint = torch.load(args_cli.diffusion_path)
    diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
    diffusion_model.eval()
    
    # Load the diffusion scheduler
    scheduler = Diffusion_Scheduler(
        beta_begin=1e-4,
        beta_end=0.02,
        steps=args_cli.denoising_steps,
        cosine=False
    )
    print(f"Scheduler initialized ({args_cli.denoising_steps} denoising steps)")
    
    # Create the controller for the diffusion model
    controller = DiffusionController(
        vae, diffusion_model, scheduler,
        args_cli.history_length, args_cli.prediction_horizon
    )
        
    # running episodes for saving
    episode_lengths = []
    videos_saved = 0
    
    for episode in range(args_cli.num_episodes):
        obs, _ = env.get_observations()
        controller.reset(num_envs)
        video_recorder.reset()
        
        episode_step = 0
        
        for step in range(args_cli.episode_length):
            # get the action from controller
            action = controller.step(obs)
            
            # step in the environment
            obs, _, dones, _ = env.step(action)
            
            # render and capture the frame
            try:
                frame = unwrapped_env.render()
                video_recorder.add_frame(frame)
            except Exception as e:
                if step == 0:
                    print(f"render failed: {e}")
            
            episode_step += 1
            
            if step % 100 == 0:
                print(f"episode {episode+1}, Step {step}/{args_cli.episode_length}")
            
            if dones.any():
                print(f"episode {episode+1} terminated at step {episode_step}")
                break
        
        episode_lengths.append(episode_step)
        
        # save video if episode was long enough
        if episode_step >= args_cli.video_length_threshold:
            video_filename = f"episode_{episode+1:03d}_steps_{episode_step}.mp4"
            if video_recorder.save(video_filename):
                videos_saved += 1
        
        print(f"episode {episode+1}/{args_cli.num_episodes} completed: {episode_step} steps")

    # summary
    print("\n" + "="*60)
    print(f"Episodes completed: {args_cli.num_episodes}")
    print(f"Videos saved: {videos_saved}")
    print(f"Average episode length: {sum(episode_lengths)/len(episode_lengths):.1f}")
    print(f"Min length: {min(episode_lengths)}")
    print(f"Max length: {max(episode_lengths)}")
    print(f"Episodes ≥ {args_cli.video_length_threshold} steps: {sum(1 for l in episode_lengths if l >= args_cli.video_length_threshold)}")
    
    if videos_saved > 0:
        print(f"Videos saved to: {os.path.abspath(args_cli.video_dir)}/")
    
    env.close()


if __name__ == "__main__":
    test_diffusion_policy()
    simulation_app.close()