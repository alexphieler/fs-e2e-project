import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from networks import ConvVAE, flattenFuncVision
from buffers import ReplayBuffer, DictReplayBuffer, load_replay_buffer, save_replay_buffer
from tqdm.rich import tqdm, trange
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib
from torch.utils.tensorboard import SummaryWriter

# --- 1. CONFIGURATION ---
IMAGE_CHANNELS = 3
IMAGE_DIM = 128  # Assuming (128, 128, 3) input
BETA = 1.0                   # Beta-VAE factor (scales KL loss)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Example Usage ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_size = 128
kl_tolerance = 0.5
device = DEVICE
# import vit
model = ConvVAE(z_size=z_size, kl_tolerance=kl_tolerance).to(device)

# 2. Define optimizer
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3. Dummy training loop (replace with your data loader)
# Assuming 'dataset' is a DataLoader yielding batches of
# (batch_size, 3, 128, 128) tensors, normalized to [0, 1].
# Note: PyTorch expects channels first: (N, C, H, W)

# dummy_dataset = [torch.rand(16, IMAGE_CHANNELS, IMAGE_DIM, IMAGE_DIM) for _ in range(10)]
# 
def resize_tensor(image_tensor, new_size=(128, 128)):
    """
    Resizes a PyTorch tensor (B, C, H, W) or (C, H, W) to (C, new_size[0], new_size[1]).
    Uses 'antialias=True' for better quality downsampling.
    """
    # Add a batch dimension if it's missing (C, H, W) -> (1, C, H, W)
    was_unbatched = False
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        was_unbatched = True

    # Perform the resize
    resized_tensor = F.interpolate(
        image_tensor,
        size=new_size,
        mode='bilinear', # 'bilinear' is good for resizing
        align_corners=False,
        antialias=True # Use antialiasing for better quality
    )
    
    # Remove the batch dimension if we added it
    if was_unbatched:
        resized_tensor = resized_tensor.squeeze(0)

    return resized_tensor


writer = SummaryWriter("runs/vae_training")

rb = load_replay_buffer("data_state_actor_noisy.pkl")
device = 'cuda'
batch_size = 16

model.train()
batch_idx = 0

MAX_BETA = 1.0  # The final beta value you want to reach (1.0 is a good default)
BETA_ANNEAL_EPOCHS = 700 # Number of epochs to ramp up beta from 0 to MAX_BETA


print(rb.size())

for epoch in tqdm(range(10000)):
    bufferSamples = rb.sample(batch_size)
    observations = bufferSamples[0]
    # bufferActions = bufferSamples[1].to(device)
    # obsflat = flattenFunc(observations)
    vision, sensors = flattenFuncVision(observations)
    vision = vision.reshape(-1, 3, vision.shape[-2], vision.shape[-1])
    # print(vision.shape)
    # vision = resize_tensor(vision, new_size=(256,256))
    vision = resize_tensor(vision, new_size=(224,224))
    vision = vision / 255.0
    # print(vision.shape)
    vision = vision.to(device)
    data = vision

    # data = data.to(DEVICE)
    
    # Forward pass
    recon_batch, mu, logvar = model(data)


    # Calculate loss
    if epoch < BETA_ANNEAL_EPOCHS:
        # Linearly ramp up beta from 0 to MAX_BETA
        current_beta = MAX_BETA * (epoch / BETA_ANNEAL_EPOCHS)
    else:
        # After annealing, keep beta at its max value
        current_beta = MAX_BETA

    # Use simple MSE reconstruction loss for now (can swap for full VAE loss)
    loss = F.mse_loss(recon_batch, data)
    # loss = F.l1_loss(recon_batch, data)
    # loss, r_loss, kl_loss = model.loss_function(recon_batch, data, mu, logvar)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("Loss/train", loss.item(), epoch)
    
    if batch_idx % 2 == 0:
        # print(f"Epoch {epoch} [Batch {batch_idx}]: "
        #         f"Total Loss: {loss.item():.4f}, "
        #         f"Recon Loss: {recon_loss.item():.4f}, "
        #         f"KL Loss: {kl_loss.item():.4f}")
        print(f"Epoch {epoch} [Batch {batch_idx}]: "
                f"Total Loss: {loss.item():.6f}, ")

    batch_idx += 1

writer.close()

# 4. GET THE ENCODER FOR YOUR RL POLICY
# After training, save the state dict:
# torch.save(model.state_dict(), 'spatial_vae.pth')
torch.save(model, "networks/vae")

matplotlib.use('Agg')

def visualize_reconstruction(model, image_samples, device, num_images=8):
    """
    Visualizes original vs. reconstructed images from the VAE.
    
    Args:
        model (nn.Module): The trained VAE model.
        data_loader (DataLoader): DataLoader to fetch a batch from.
        device (torch.device): The device the model and data are on.
        num_images (int): Number of image pairs to display.
    """
    
    print("Generating reconstructions...")
    # Set model to evaluation mode
    model.eval()
    
    # # Get one batch of test data
    # try:
    #     originals = next(iter(data_loader))
    # except StopIteration:
    #     print("DataLoader is empty.")
    #     return
    originals=image_samples

    # Move data to the correct device
    originals = originals.to(device)
    
    # We only need num_images
    originals = originals[:num_images]
    
    # Get reconstructions
    with torch.no_grad():
        recons, _, _ = model(originals)
        # recons = model(originals)
        # recons, mu, logvar = model(data)

    # Move images back to CPU for plotting
    originals = originals.cpu()
    recons = recons.cpu()
    
    # --- Create a comparison grid ---
    # We'll interleave the images: [original_1, recon_1, original_2, recon_2, ...]
    
    # 1. Stack them: (num_images, 2, C, H, W)
    #    (The '2' dimension holds the [original, recon] pair)
    comparison = torch.stack([originals, recons], dim=1)
    
    # 2. Flatten them into a single list: (num_images * 2, C, H, W)
    comparison = comparison.view(-1, *originals.shape[1:])

    # 3. Create the grid
    #    nrow=2 makes it display in [original, recon] columns
    grid = vutils.make_grid(
        comparison,
        nrow=2,             # Display as [Original, Recon]
        padding=2,          # Padding between images
        normalize=True      # Adjust pixel values to [0, 1] for display
    )
    
    # --- Plot the grid (convert colors and rotate 90° right) ---
    plt.figure(figsize=(num_images, 4)) # Adjust size as needed
    # Convert (C, H, W) -> (H, W, C), move to CPU and to numpy
    img = grid.permute(1, 2, 0).cpu().numpy()
    # If this is a 3-channel image in BGR order (e.g., from OpenCV), convert to RGB
    if img.shape[-1] == 3:
        img = img[..., ::-1]
    # Rotate 90 degrees right (clockwise) for display
    img_rot = np.rot90(img, k=-1)
    plt.imshow(img_rot)
    plt.title('Originals (Top Row) vs. Reconstructions (Bottom Row)')
    plt.axis('off')
    plt.savefig('vae.png')
    plt.close()



# bufferSamples = rb.sample(2)
observations = bufferSamples[0]
bufferActions = bufferSamples[1].to(device)
# obsflat = flattenFunc(observations)
vision, sensors = flattenFuncVision(observations)
vision = vision.reshape(-1, 3, vision.shape[-2], vision.shape[-1])
# print(vision.shape)
# vision = resize_tensor(vision, new_size=(256,256))
vision = resize_tensor(vision, new_size=(224,224))
vision = vision / 255.0
vision = vision.to(device)

visualize_reconstruction(model, vision, "cuda")