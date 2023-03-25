import torch
import torch.nn as nn  # neural network modules: nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  #  Optimization : SGD, Adam, etc.
import torch.nn.functional as F  # Activation functions: relu, sigmoid etc.
from torchvision import datasets, transforms
from torch.utils.data import DataLoader # Dataset managment and creates mini batches

# Create Network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # fully connected layers : fc1 ...
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784  # number of features 28x28
num_classes = 10  # final freatures
hidden_size = 128
momentum = 0.9  # used in optimizer
learning_rate = 0.01
batch_size = 64  # number of samples used in each iteration
num_epochs = 2  # trained twice on whole dataset


# Load Data
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
)

train_dataset = datasets.MNIST(
    root="data/", transform=transforms, train=True, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(
    root="data/", transform=transforms, train=False, download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize neural network
model = NN(
    input_size=input_size,
    num_classes=num_classes,
    hidden_size=hidden_size,
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # reshaping
        data = data.reshape(data.shap[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent and adam step
        optimizer.step() 

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

# Evaluation
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Accuracy on train data")
    else:
        print("Accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Accuray {num_correct}/ {num_samples} = {float(num_correct)/float(num_samples) * 100 :.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
