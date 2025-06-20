import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import gradio as gr
import os

# Device (CPU for compatibility with Hugging Face Spaces)
device = torch.device("cpu")

# Transform for training and uploaded images
transform = transforms.Compose([
    transforms.Resize((6, 6)),
    transforms.ToTensor()
])

# Define a convolution block
def conv(ic, oc):
    ks=3
    return nn.Sequential(
        nn.Conv2d(ic, oc, stride=2, kernel_size=ks, padding=ks//2),
        nn.BatchNorm2d(oc)
    )

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv(1, 8),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            conv(8, 16),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            conv(16, 10),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    batch_size = 36
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # Keep it light for HF Spaces
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# Load or train model
model_path = "mnist_cnn.pt"
if os.path.exists(model_path):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model = train_model()
    torch.save(model.state_dict(), model_path)

# Prediction function
def predict(img):
    if isinstance(img, Image.Image):
        img = img.convert("L")
    else:
        return "Invalid image"
    x = transform(img).unsqueeze(0).to(device)  # Shape: [1,1,8,8]
    model.eval()
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
    return f"Predicted digit: {pred}"

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs="text",
    title="MNIST Digit Classifier (6x6 CNN)",
    description="Upload or draw a digit to classify it using a lightweight CNN trained on MNIST resized to 8×8."
)

if __name__ == "__main__":
    demo.launch()
