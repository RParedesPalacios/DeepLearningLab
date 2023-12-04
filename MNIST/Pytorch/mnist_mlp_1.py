import torch
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print("Torch version: ", torch. __version__)

####################################################################
# Set Device
####################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


####################################################################
# Prepare Data
####################################################################

train_set = torchvision.datasets.MNIST('.data/', train=True, download=True)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST('.data/', train=False, download=True)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

print("Train images: ", train_set)
print("Image: ", train_set[0][0])
print("Label: ", train_set[0][1])
print("Label one hot: ", F.one_hot(torch.tensor(train_set[0][1]), num_classes=10))


####################################################################
# Dataset Class
####################################################################

class MNIST_dataset(Dataset):

    def __init__(self, data, partition = "train"):

        print("\nLoading MNIST ", partition, " Dataset...")
        self.data = data
        self.partition = partition
        print("\tTotal Len.: ", len(self.data), "\n", 50*"-")

    def __len__(self):
        return len(self.data)
    
    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):

        # Image
        image = self.data[idx][0]
        # PIL Image to torch tensor
        image_tensor = self.from_pil_to_tensor(image)
        # care! net expect a 784 size vector and our dataset 
        # provide 1x28x28 (channels, height, width) -> Reshape!
        image_tensor = image_tensor.view(-1)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}

train_dataset = MNIST_dataset(train_set, partition="train")
test_dataset = MNIST_dataset(test_set, partition="test")


####################################################################
# DataLoader Class
####################################################################

batch_size = 100
num_workers = multiprocessing.cpu_count()-1
print("Num workers", num_workers)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)


####################################################################
# Neural Network Class
####################################################################

# Creating our Neural Network - Fully Connected
class Net(nn.Module):
    def __init__(self, num_classes):
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
num_classes = 10
net = Net(num_classes)
print(net)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(net))

####################################################################
# Training settings
####################################################################

# Training hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)
epochs = 25


####################################################################
# Training
####################################################################

# Load model in GPU
net.to(device)

print("\n---- Start Training ----")
best_accuracy = -1
best_epoch = 0
for epoch in range(epochs):


    # TRAIN NETWORK
    train_loss, train_correct = 0, 0
    net.train()
    with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
        for batch in tepoch:
            
            # Returned values of Dataset Class
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Calculate gradients
            loss.backward()

            # Update gradients
            optimizer.step()

            # one hot -> labels
            labels = torch.argmax(labels, dim=1)
            pred = torch.argmax(outputs, dim=1)
            train_correct += pred.eq(labels).sum().item()

            # print statistics
            train_loss += loss.item()

    train_loss /= len(train_dataloader.dataset)

    # TEST NETWORK
    test_loss, test_correct = 0, 0
    net.eval()
    with torch.no_grad():
      with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
          for batch in tepoch:

            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # Forward
            outputs = net(images)
            test_loss += criterion(outputs, labels)

            # one hot -> labels
            labels = torch.argmax(labels, dim=1)
            pred = torch.argmax(outputs, dim=1)

            test_correct += pred.eq(labels).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * test_correct / len(test_dataloader.dataset)

    print("[Epoch {}] Train Loss: {:.6f} - Test Loss: {:.6f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%".format(
        epoch + 1, train_loss, test_loss, 100. * train_correct / len(train_dataloader.dataset), test_accuracy
    ))

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch

        # Save best weights
        torch.save(net.state_dict(), "best_model.pt")

print("\nBEST TEST ACCURACY: ", best_accuracy, " in epoch ", best_epoch)



####################################################################
# Load best weights
####################################################################

# Load best weights
net.load_state_dict(torch.load("best_model.pt"))

test_loss, test_correct = 0, 0
net.eval()
with torch.no_grad():
    with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
        for batch in tepoch:

            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # Forward
            outputs = net(images)
            test_loss += criterion(outputs, labels)

            # one hot -> labels
            labels = torch.argmax(labels, dim=1)
            pred = torch.argmax(outputs, dim=1)

            test_correct += pred.eq(labels).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * test_correct / len(test_dataloader.dataset)
print("Final best acc: ", test_accuracy)
