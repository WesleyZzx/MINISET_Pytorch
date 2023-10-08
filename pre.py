import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Lenet import LeNet  # Assuming LeNet is your model class

# Load the trained model
model = LeNet()  # Assuming LeNet is your model class
model.load_state_dict(torch.load('lenet_epoch.pth'))  # Path to your saved model params
model.eval()

# Load and preprocess the image you want to predict
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match LeNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

# Load and preprocess the image
image_path = 'pre5.jpg'  # Replace with your image path
image = preprocess_image(image_path)

# Make prediction
with torch.no_grad():
    output = model(image)

# Get the predicted classes for each sample
predicted_classes = torch.argmax(output, dim=1)

# Get the predicted probabilities for each sample
predicted_probabilities = torch.softmax(output, dim=1)

# Display the images, predictions, and probabilities
plt.figure(figsize=(12, 6))
for i in range(predicted_classes.shape[0]):
    plt.subplot(1, predicted_classes.shape[0], i + 1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_classes[i].item()}\n'
              f'Probability: {predicted_probabilities[i, predicted_classes[i]].item():.4f}')
plt.show()
