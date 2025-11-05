import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load model
checkpoint = torch.load("final_model.pth", map_location="cpu")
if 'xyz' in checkpoint:
    state = checkpoint
else:
    state = checkpoint['model_state']

# Extract Gaussian positions and colors
positions = state['_xyz'].cpu().numpy()
features_dc = state['_features_dc'].cpu().numpy()
opacities = torch.sigmoid(state['_opacity']).cpu().numpy()

# Convert SH to RGB (DC component only for simplicity)
colors = features_dc[:, 0, :] * 0.28209479177387814  # SH constant
colors = np.clip(colors, 0, 1)

print(f"Total Gaussians: {len(positions):,}")
print(f"Position range:")
print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
print(f"Mean opacity: {opacities.mean():.3f}")

# Plot visible Gaussians (opacity > 0.1)
visible = opacities.squeeze() > 0.1
vis_pos = positions[visible]
vis_colors = colors[visible]

print(f"Visible Gaussians (opacity > 0.1): {len(vis_pos):,}")

# 3D scatter plot
fig = plt.figure(figsize=(15, 5))

# View 1: XY plane
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(vis_pos[:, 0], vis_pos[:, 1], vis_pos[:, 2],
            c=vis_colors, s=1, alpha=0.5)
ax1.set_title("3D View 1")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# View 2: Different angle
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(vis_pos[:, 0], vis_pos[:, 1], vis_pos[:, 2],
            c=vis_colors, s=1, alpha=0.5)
ax2.view_init(elev=20, azim=45)
ax2.set_title("3D View 2")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# View 3: Top-down
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(vis_pos[:, 0], vis_pos[:, 1], vis_pos[:, 2],
            c=vis_colors, s=1, alpha=0.5)
ax3.view_init(elev=90, azim=0)
ax3.set_title("Top-Down View")
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.savefig("./output/quick_view.png", dpi=150)
print("\nSaved visualization to ./output/quick_view.png")
plt.show()