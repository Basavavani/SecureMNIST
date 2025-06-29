import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

print("🔒 Starting Secure Federated MNIST Training...")

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset and simulate data splitting between 2 clients
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Split 60K samples into 2 halves
client1_data, client2_data = torch.utils.data.random_split(dataset, [30000, 30000])
loader1 = torch.utils.data.DataLoader(client1_data, batch_size=64, shuffle=True)
loader2 = torch.utils.data.DataLoader(client2_data, batch_size=64, shuffle=True)

# Local training function
def train(model, data_loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    return model

# Train model on client 1
print("🏋️ Training on Client 1...")
model_c1 = Net()
optimizer_c1 = optim.SGD(model_c1.parameters(), lr=0.01)
model_c1 = train(model_c1, loader1, optimizer_c1)

# Train model on client 2
print("🏋️ Training on Client 2...")
model_c2 = Net()
optimizer_c2 = optim.SGD(model_c2.parameters(), lr=0.01)
model_c2 = train(model_c2, loader2, optimizer_c2)

# Federated Averaging
print("🔗 Performing Federated Averaging...")
global_model = Net()
with torch.no_grad():
    for g_param, c1_param, c2_param in zip(global_model.parameters(), model_c1.parameters(), model_c2.parameters()):
        g_param.data.copy_((c1_param.data + c2_param.data) / 2)

print("✅ Federated Averaging Done!")
print("🎉 Secure Federated Training Completed.")
# 📊 Evaluate the global model on test data
print("🔍 Evaluating Global Model...")

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

global_model.eval()
correct = 0
total = 0
loss_total = 0
loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
        outputs = global_model(images)
        loss = loss_fn(outputs, labels)
        loss_total += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
avg_loss = loss_total / len(test_loader)

print(f"✅ Accuracy: {accuracy:.2f}%")
print(f"📉 Average Loss: {avg_loss:.4f}")

# 💾 Save the global model
torch.save(global_model.state_dict(), "global_model.pth")
print("💾 Global model saved as 'global_model.pth'")


