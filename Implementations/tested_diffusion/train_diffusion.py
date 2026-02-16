import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusion_dataset import DiffusionDataset
from diffusion_model import DiffusionTransformer
from scheduler import Diffusion_Scheduler
from tqdm import tqdm
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def diffusion_training(
    model,
    scheduler,
    filepaths,
    num_epochs = 1000,
    batch_size = 3000, 
    lr = 1e-4,
    weight_decay=1e-3,
    history_len=4,
    prediction_horizon=8,
    use_wandb=True
):
    if use_wandb:
        wandb.init(project="diffusion-training", config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "history_len": history_len,
            "prediction_horizon": prediction_horizon,
            "scheduler_steps": scheduler.steps
        })
        
    model = model.to(device)
    dataset = DiffusionDataset(filepaths, history_len, prediction_horizon)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=4)
    optimizer = optim.AdamW(model.parameters(), lr = lr , weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    num_timesteps = history_len + 1 + prediction_horizon

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0
        total_state_loss = 0
        total_latent_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            states = batch[0].to(device)
            latents = batch[1].to(device)
            
            B = states.shape[0]
            seq_len = 2 * num_timesteps 
            
            # Sample the states for each of the diffusion samples
            state_steps = torch.randint(0, scheduler.steps, (B, num_timesteps), device=device)
            latent_steps = torch.randint(0, scheduler.steps, (B, num_timesteps), device=device)
            noisy_states, _ = scheduler.add_noise_per_token(states, state_steps)
            noisy_latents, _ = scheduler.add_noise_per_token(latents, latent_steps)
            diffusion_steps = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            diffusion_steps[:, 0::2] = state_steps
            diffusion_steps[:, 1::2] = latent_steps
            
            # Predict what the clean trajectory looks like
            pred_states, pred_latents = model(noisy_states, noisy_latents, diffusion_steps)
            
            # What the actual trajectory was versus the reconstructed trajectory
            state_loss = nn.functional.mse_loss(pred_states, states)
            latent_loss = nn.functional.mse_loss(pred_latents, latents)
            loss = state_loss + latent_loss
            
            # Find the backwards
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_latent_loss += latent_loss.item()
        avg_loss = total_loss / len(dataloader)
        avg_state_loss = total_state_loss / len(dataloader)
        avg_latent_loss = total_latent_loss / len(dataloader)

        lr_scheduler.step()
        
        n = len(dataloader)
        print(f"Epoch {epoch}: Loss={total_loss/n:.4f} "
              f"(State={total_state_loss/n:.4f}, Latent={total_latent_loss/n:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/diffusion_checkpoint_{epoch+1}.pt')

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "loss": avg_loss,
                "state_loss": avg_state_loss,
                "latent_loss": avg_latent_loss,
                "lr": lr_scheduler.get_last_lr()[0]
            })
    
    if use_wandb:
        wandb.finish()

    return model

if __name__ == '__main__':
    model = DiffusionTransformer()
    scheduler = Diffusion_Scheduler(beta_begin=1e-4, beta_end = 0.02, steps = 200, cosine = False)
    trajectory_files = [
        "data/iter4_walk1_subject2_fixed.pt",
        "data/iter4_walk1_subject5_fixed.pt",
        "data/iter4_walk2_subject3_fixed.pt",
        "data/iter4_run1_subject2_fixed.pt",
        "data/iter4_run2_subject1_fixed.pt",
        "data/iter4_run2_subject4_fixed.pt"
    ]
    model = diffusion_training(model, scheduler, filepaths = trajectory_files)