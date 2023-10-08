import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Lenet import LeNet
# Define LeNet and its architecture
# ... (LeNet model definition)

# Define data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create LeNet model instance
model = LeNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification problems
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_data, batch_targets in train_loader:
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss for each epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'lenet_epoch.pth')

print('Training finished.')
