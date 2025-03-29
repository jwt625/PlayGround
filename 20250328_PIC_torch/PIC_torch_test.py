#%%
import torch
import torch.nn as nn
import torch.optim as optim

# Assume that the target function T(x, y) is defined by the user.
# Here we provide a dummy implementation for demonstration.
def target_function(x, y):
    # Replace this with your complicated function.
    # For instance, this dummy function returns sin(pi*x) * cos(pi*y)
    return torch.sin(4*torch.pi * x) * torch.cos(torch.pi * y)

# Define the neural network with two hidden layers.
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network, loss function (MSE for least squares), and optimizer.
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Generate training data.
# For example, generate 1000 random (x, y) pairs in the interval [0,1].
num_samples = 1000
inputs = torch.rand(num_samples, 2)
x_vals = inputs[:, 0]
y_vals = inputs[:, 1]
# Compute the target values using the target function.
targets = target_function(x_vals, y_vals).unsqueeze(1)  # shape: (num_samples, 1)

# Training loop.
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()           # Reset gradients
    outputs = net(inputs)           # Forward pass
    loss = criterion(outputs, targets)  # Compute loss (least squares)
    loss.backward()                 # Backpropagation
    optimizer.step()                # Update parameters
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')






#%%
import numpy as np
import matplotlib.pyplot as plt

# Create a grid of (x,y) values.
grid_points = 100
x_grid = np.linspace(0, 1, grid_points)
y_grid = np.linspace(0, 1, grid_points)
X, Y = np.meshgrid(x_grid, y_grid)
XY = np.column_stack((X.ravel(), Y.ravel()))
XY_tensor = torch.tensor(XY, dtype=torch.float32)

# Evaluate the network and target function on the grid.
with torch.no_grad():
    # Neural network prediction reshaped to grid
    pred = net(XY_tensor).cpu().numpy().reshape(X.shape)
    
    # Compute the target function on the grid.
    # Convert grid values to tensors.
    x_vals = torch.tensor(X.ravel(), dtype=torch.float32)
    y_vals = torch.tensor(Y.ravel(), dtype=torch.float32)
    target_vals = target_function(x_vals, y_vals).cpu().numpy().reshape(X.shape)

# Plot the target function and the neural network fit.
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot the target function.
contour1 = ax[0].contourf(X, Y, target_vals, 50)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title('Target Function T(x,y)')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

# Plot the neural network fit.
contour2 = ax[1].contourf(X, Y, pred, 50)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title('Neural Network Fit')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

plt.tight_layout()
plt.show()

# Evaluate the difference between the neural network fit and the target function.
difference = pred - target_vals

# Create a new figure window for the difference plot.
plt.figure(figsize=(6, 5))
diff_contour = plt.contourf(X, Y, difference, 50)
plt.title('Difference: Neural Network Fit - Target Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(diff_contour)
plt.tight_layout()
plt.show()






#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for name, param in net.named_parameters():
    if param.grad is not None:
        writer.add_histogram(f'{name}_values', param.data, epoch)
        writer.add_histogram(f'{name}_grads', param.grad, epoch)


#%%
# Add TensorBoard visualization
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/network_structure')

# Create a dummy input tensor
dummy_input = torch.randn(1, 2)  # batch_size=1, input_size=2
writer.add_graph(net, dummy_input)
writer.close()
















#%%

import torch
import torch.nn as nn
import torch.optim as optim

# Custom module to generate fc2 parameters using simple math functions.
class HyperFC2(nn.Module):
    def __init__(self, hidden_size, latent_dim=10):
        """
        hidden_size: the input/output size of the fc2 layer
        latent_dim: dimension of the latent vector z used for parameter generation
        """
        super(HyperFC2, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        # A latent vector that will be optimized.
        self.z = nn.Parameter(torch.randn(1, latent_dim))
        
        # Instead of a full MLP, we use simple math functions (sin, cos, sqrt)
        # and then a linear projection to produce the flattened weight and bias.
        # After applying sin, cos, and sqrt, we get 3 * latent_dim features.
        # The final projection outputs (hidden_size*hidden_size + hidden_size) elements.
        self.proj = nn.Linear(latent_dim * 3, hidden_size * hidden_size + hidden_size)
        
    def forward(self, x):
        """
        x: input tensor with shape [batch_size, hidden_size]
        Returns: transformed x with shape [batch_size, hidden_size]
        """
        # Use differentiable math functions on the latent vector z.
        a = torch.sin(self.z)  # shape [1, latent_dim]
        b = torch.cos(self.z)  # shape [1, latent_dim]
        # Use sqrt on the absolute value (adding a small epsilon to avoid issues with zero).
        c = torch.sqrt(torch.abs(self.z) + 1e-6)  # shape [1, latent_dim]
        
        # Concatenate the features along the feature dimension.
        features = torch.cat([a, b, c], dim=1)  # shape [1, latent_dim * 3]
        
        # Project the concatenated features to generate parameters.
        hyper_out = self.proj(features)  # shape [1, hidden_size*hidden_size + hidden_size]
        
        # Split the output into weight and bias.
        weight_flat = hyper_out[:, :self.hidden_size * self.hidden_size]
        bias = hyper_out[:, self.hidden_size * self.hidden_size:]
        
        # Reshape weight into a matrix of shape [hidden_size, hidden_size].
        weight = weight_flat.view(self.hidden_size, self.hidden_size)
        
        # Compute the linear transformation: x @ weight^T + bias.
        out = x @ weight.t() + bias
        return out

# Define the main network. We'll use the HyperFC2 module in place of the regular fc2 layer.
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=20, output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = HyperFC2(hidden_size)  # Our custom hypernetwork layer.
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        self.act1 = nn.ReLU()
        # fc2 includes its own computation; we then add another activation after fc3.
        self.act3 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

# Example usage: Initialize network, define loss and optimizer.
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Create some dummy training data.
num_samples = 1000
inputs = torch.rand(num_samples, 2)

# Dummy target function (for demonstration).
def target_function(x, y):
    return torch.sin(4 * torch.pi * x) * torch.cos(torch.pi * y)
x_vals = inputs[:, 0]
y_vals = inputs[:, 1]
targets = target_function(x_vals, y_vals).unsqueeze(1)

# Forward pass.
outputs = net(inputs)
loss = criterion(outputs, targets)
loss.backward()

# Print some diagnostics.
print("Loss:", loss.item())
print("Gradient for latent vector z in fc2:", net.fc2.z.grad)

# %%
