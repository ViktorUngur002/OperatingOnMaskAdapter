import pickle
import torch

# Load the .pkl file
with open('mask2former_w/model_final_7e47bf.pkl', 'rb') as f:
    data = pickle.load(f)

# Save the data as a .pth file
torch.save(data, 'mask2former_w/mask2former.pth')

print("Conversion from .pkl to .pth completed!")