# Library import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Constants definition
batch_size = 100
epochs = 25
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating dataloaders
# ToTensor() - Converts a Image (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.MNIST('.data/', train=True, download=True, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST('.data/', train=False, download=True, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Statistics printing
x_batch, y_batch = iter(train_loader).next()
print("Training set: {} samples - Max value: {} - Min value: {}".format(len(train_loader.dataset),
                                                                        x_batch.max(), x_batch.min()))
x_batch, y_batch = iter(test_loader).next()
print("Test set: {} samples - Max value: {} - Min value: {}".format(len(test_loader.dataset),
                                                                    x_batch.max(), x_batch.min()))
print("Example batch shape: {}".format(x_batch.shape))


# Creating our Neural Network - Fully Connected
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.relu1(self.linear1(x))
        out = self.relu2(self.linear2(out))
        out = self.relu3(self.linear3(out))
        out = self.classifier(out)
        return out


# Instantiating the network and printing its architecture
net = Net().to(device)
print(net)

# Training hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)

# Start training
print("\n---- Start Training ----")
best_accuracy = -1
for epoch in range(epochs):

    # TRAIN THE NETWORK
    train_loss, train_correct = 0, 0
    net.train()
    for inputs, targets in train_loader:
        # data is a list of [inputs, labels]
        inputs, targets = inputs.to(device), targets.to(device)
        # care! net expect a 784 size vector and our dataset provide 1x28x28 (channels, height, width) -> Reshape!
        inputs = inputs.view(inputs.size(0), -1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, pred = outputs.max(1)  # get the index of the max log-probability
        train_correct += pred.eq(targets).sum().item()

        # print statistics
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    # TEST NETWORK
    net.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # care! net expect a 784 size vector and our dataset provide 1x28x28 (channels, height, width) -> Reshape!
            inputs = inputs.view(inputs.size(0), -1)
            outputs = net(inputs)
            test_loss += criterion(outputs, targets)
            _, pred = outputs.max(1)  # get the index of the max log-probability
            correct += pred.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print("[Epoch {}] Train Loss: {:.6f} - Test Loss: {:.6f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%".format(
        epoch + 1, train_loss, test_loss, 100. * train_correct / len(train_loader.dataset), test_accuracy
    ))

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

print("Finished Training")
print("Best Test accuracy: {:.2f}%".format(best_accuracy))
