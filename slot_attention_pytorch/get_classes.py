import torch
from model import SlotAttentionAutoEncoder
from dataset import CustomDataset

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt  # For plotting

# Import your model and dataset definitions.
from model import SlotAttentionAutoEncoder  
from dataset import CustomDataset    

class SlotClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(SlotClassifier, self).__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, slots):
        # slots: expected shape [batch_size, num_slots, hidden_dim]
        logits = self.linear(slots)  # If slots has extra spatial dims, linear will be applied to the last dimension.
        probs = F.softmax(logits, dim=-1)  # Softmax over the class dimension
        return probs

def main():
    # Set device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define model hyperparameters
    resolution = (32, 32)
    num_slots = 7
    num_iterations = 3
    hid_dim = 64
    hidden_dim = 64
    num_classes = 10  

    # Initialize the model
    model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, hid_dim)
    
    # Load the checkpoint
    checkpoint_path = 'ckpts/model_220.ckpt'  # Update path if necessary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model.to(device)
    model.eval()

    classifier = SlotClassifier(hidden_dim, num_classes)
    classifier.to(device)
    # If you have a trained classifier head, load its state_dict here.
    classifier.eval()

    # Ensure any registered buffer (like 'grid') is on the right device.
    if hasattr(model, 'grid'):
        model.grid = model.grid.to(device)
    
    # Load an example image from your dataset
    dataset = CustomDataset("1000_nopick_pixel_states.npy")
    sample = dataset[0]  # Get the first image sample
    image = sample['image'].unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    image = image.to(device)
    print("Image device:", image.device)

    # Forward pass: obtain slot representations
    with torch.no_grad():
        # The model's forward returns several outputs; we extract slots.
        _, _, _, slots = model(image)

        print(slots.shape)

        # class_probs = classifier(slots)
        # print("Class probabilities shape:", class_probs.shape)  # Likely [7, 8, 8, 10]
    
    # ------------------- Plotting Code ------------------- #
    # Convert the image to a NumPy array for plotting.
    # Assuming the image tensor shape is [1, C, H, W] (C could be 1 or 3)
    # img_np = image[0].cpu().permute(1, 2, 0).numpy()
    # if img_np.shape[2] == 1:
    #     img_np = img_np.squeeze(axis=2)

    # # Convert class probabilities to NumPy.
    # # Here, class_probs is [num_slots, H, W, num_classes]
    # slot_probs = class_probs.cpu().numpy()  # Shape: [7, 8, 8, 10]
    # num_slots = slot_probs.shape[0]
    # num_classes = slot_probs.shape[-1]

    # # Create subplots: one for the image and one per slot.
    # fig, axes = plt.subplots(1, num_slots + 1, figsize=(4 * (num_slots + 1), 4))

    # # Plot the input image.
    # axes[0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
    # axes[0].set_title("Input Image")
    # axes[0].axis("off")

    # # Plot the class probability distribution for each slot.
    # # We average the spatial dimensions (axis 0 and 1 of each slot) to get a single probability vector per slot.
    # for i in range(num_slots):
    #     # Average over the spatial dimensions (8,8) -> result is shape [num_classes]
    #     slot_avg = slot_probs[i].mean(axis=(0, 1))
    #     axes[i + 1].bar(np.arange(num_classes), slot_avg)
    #     axes[i + 1].set_title(f"Slot {i} Class Probabilities")
    #     axes[i + 1].set_xlabel("Class")
    #     axes[i + 1].set_ylabel("Probability")
    #     axes[i + 1].set_ylim(0, 1)  # Probabilities are between 0 and 1

    # plt.tight_layout()
    # plt.show()
    # ------------------------------------------------------ #

if __name__ == '__main__':
    main()
