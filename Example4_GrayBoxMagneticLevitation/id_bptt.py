# Identification of levitator gray-box parameters by ADAM BPTT.
# Simone Pirrera
# Oct 17, 2024

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io

# Define the recurrent network model
class LevitatorModel(nn.Module):
    def __init__(self):
        super(LevitatorModel, self).__init__()
        self.km = nn.Parameter(torch.tensor(0.5))  # Initialize a_1 parameter
        self.k0 = nn.Parameter(torch.tensor(0.5))  # Initialize b_1 parameter
        self.mass = torch.tensor(24.197e-3)  # Initialize c_1 parameter for y_{t-2}
    def forward(self, y_prev, y_prev2, u_t2):
        y_t = 0.001*(-self.k0 - self.km * (u_t2**2) + 9.81*self.mass*(y_prev2**2))+2*y_prev-y_prev2
        return y_t    
    def printParams(self):
        print(f'Km={self.km.double():.6f}, k0={self.k0.double():.6f}')

# Generate some sample data
torch.manual_seed(0)
# Load data from .mat file
mat_data = scipy.io.loadmat('levitat_data.mat')

# Assuming the .mat file has variables 'u' and 'y'
u_data = torch.tensor(mat_data['u'].squeeze(), dtype=torch.float32)
y_data = torch.tensor(mat_data['y'].squeeze(), dtype=torch.float32)

u_data = u_data[0:200]
y_data = y_data[0:200]

# Initialize the model, loss function, and optimizer
model = LevitatorModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 6000
for epoch in range(num_epochs):
    
    y_sim = torch.zeros(200);
    y_sim[0] = torch.tensor(y_data[0])
    y_sim[1] = torch.tensor(y_data[1])
    
    # Simulation
    for t in range(2,len(u_data)):
        y_sim[t] = model(y_sim[t-1].detach(), y_sim[t-2].detach(), u_data[t-2])
        
    # Compute the loss
    loss = sum((y_data - y_sim)**2)
        
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

print("Training complete.")

model.printParams()

# Result with 10.000 epochs
# Km=0.000999, k0=-0.173830


# Result with 6.000 epochs
# Km=0.000746, k0=-0.143092