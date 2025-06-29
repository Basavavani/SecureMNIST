import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import crypten
import crypten.nn as cnn

# Step 1: Initialize CrypTen
crypten.init()

# Step 2: Load and transform MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_data = MNIST(root='./data', train=True, download=True, transform=transform)

# Step 3: Preprocess the data
images, labels = [], []
for i in range(100):  # take 100 samples for demo
    img, label = train_data[i]
    images.append(img.view(-1))  # Flatten 28x28 to 784
    labels.append(label)

X = torch.stack(images)  # shape: [100, 784]
y = torch.tensor(labels)

# Step 4: Encrypt data using CrypTen
enc_X = crypten.cryptensor(X)
enc_y = crypten.cryptensor(y)

# Step 5: Build encrypted neural network
model = cnn.Sequential(
    cnn.Linear(784, 128),
    cnn.ReLU(),
    cnn.Linear(128, 10)
)
model.encrypt()

# Step 6: Make predictions
output = model(enc_X)
predicted = output.argmax(dim=1)
correct = predicted.eq(enc_y).sum()
accuracy = correct.get_plain_text().item() / len(y)

# Step 7: Print accuracy
print(f"Accuracy on encrypted data: {accuracy * 100:.2f}%")
