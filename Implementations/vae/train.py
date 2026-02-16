import torch
from model import VAE
from torch.utils.data import Dataset, DataLoader
import os
import wandb
from tqdm import tqdm
import argparse

'''
File used for training VAE model
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.load(data)

        # reference joint states
        self.reference_joint_states = torch.cat([
            self.data['target_joint_pos'],
            self.data['target_joint_vel'] 
        ], dim=-1)

        # save anchor poses
        self.anchor_pose = torch.cat([
            self.data['base_lin_vel'], 
            self.data['base_ang_vel'],
        ], dim=-1)

        # save conditioning variables
        self.conditioning = torch.cat([
            self.data['joint_pos'],
            self.data['joint_vel'],
            self.data['pose_error_pos'],
            self.data['pose_error_ori'],
            self.data['last_action'],
        ], dim=-1)

        self.actions = self.data['target_action']
    
    def __len__(self):
        return self.conditioning.shape[0]
    
    def __getitem__(self, index):
        # generates all the needed data for running the training.
        return self.reference_joint_states[index], self.anchor_pose[index], self.conditioning[index], self.actions[index]


def train_vae(model, epochs, dataloader, output_dir, dataset, lr=5e-4, beta=0.01, iteration = 0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()

        # define loss parameters for logging
        accumulated_loss = 0
        accumulated_recon_loss = 0
        accumulated_kl_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in progress_bar:
            joint_states = batch[0].to(device)
            anchor_pose = batch[1].to(device)
            conditioning = batch[2].to(device)
            actions = batch[3].to(device)

            # generate the required outputs from the model for training
            mean, logvar, z, output = model.forward(joint_states, anchor_pose, conditioning)

            #Find model loss
            loss, recon_loss, kl_loss = model.loss(output, actions, mean, logvar, beta)

            #Calculate backwards
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Save information for logging
            accumulated_loss += loss.item()
            accumulated_recon_loss += recon_loss.item()
            accumulated_kl_loss += kl_loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = accumulated_loss / len(dataloader)
        avg_recon_loss = accumulated_recon_loss / len(dataloader)
        avg_kl_loss = accumulated_kl_loss / len(dataloader)

        wandb.log({
            "epoch": epoch,
            "loss/total": avg_loss,
            "loss/reconstruction": avg_recon_loss,
            "loss/kl_divergence": avg_kl_loss,
        })

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            checkpoint_path = os.path.join(output_dir, f"vae_model_{epoch+1}_iter{iteration}.pth")
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(output_dir, "vae_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'condition_dim': dataset.conditioning.shape[-1],
        'action_dim': dataset.actions.shape[-1],
        'latent_dim': 32,
    }, final_path)
    print(f"\nFinal model saved to {final_path}")

    return model

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--data_path", type=str, default="../datasets/combined_dataset.pt")
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--iter", type=int, default=0)
    args = parser.parse_args()

    dataset = trajectoryDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dims = dataset.reference_joint_states.shape[-1] + dataset.anchor_pose.shape[-1]
    output_dims = dataset.actions.shape[-1]
    conditioning_dims = dataset.conditioning.shape[-1]
    latent_dims = 32

    wandb.init(
        project="vae-motion-training",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "beta": args.beta,
            "latent_dim": latent_dims,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "conditioning_dims": conditioning_dims,
            "dataset_size": len(dataset),
        }
    )

    model = VAE(input_dims, output_dims, latent_dims, conditioning_dims).to(device)
    wandb.watch(model, log="all", log_freq=100)
    model = train_vae(model, args.epochs, dataloader, args.output_dir, dataset, lr = args.lr, beta=args.beta, iteration = args.iter)
    wandb.finish()