import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn


n_classes = 2  # Change this according to the number of classes in your dataset

# Load the trained model
model = models.resnet18(pretrained=False)  # Instantiate the ResNet18 model
model.fc = nn.Linear(512, n_classes)  # Update the fully connected layer for the correct number of classes
model.load_state_dict(torch.load("C:\\Users\\vinay\\Desktop\\Github\\Stop Sign Project\\model.pt"))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the expected input size of the model
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
])

# Function to predict stop sign from an image file
def predict_stop_sign(image_path):
    # Open the image
    image = Image.open(image_path)
    # Apply the transformation
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        # Assuming class 1 represents a stop sign
        if predicted.item() == 0:
            return "Stop sign detected!"
        else:
            return "No stop sign detected."

# Test the model on sample images
image_paths = ['C:\\Users\\vinay\\Desktop\\Github\\Stop Sign Project\\t4.jpg']
for image_path in image_paths:
    prediction = predict_stop_sign(image_path)
    print(f"{image_path}: {prediction}")