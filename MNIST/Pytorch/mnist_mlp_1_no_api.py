import gzip
import torch
import struct
import numpy as np
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


# Dowload Data
#!wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
#!wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
#!wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
#!wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        gz.read(4)
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows, n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8).tolist()
    return res

train_images = torch.tensor(load_data('./content/train-images-idx3-ubyte.gz', label=False))
test_images = torch.tensor(load_data('./content/t10k-images-idx3-ubyte.gz', label=False))

train_labels = torch.tensor(load_data('./content/train-labels-idx1-ubyte.gz', label=True))
test_labels = torch.tensor(load_data('./content/t10k-labels-idx1-ubyte.gz', label=True))

print("Train images shape: ", train_images.shape)
print("Test images shape: ", test_images.shape)
print("Train labels shape: ", train_labels.shape)
print("Train labels shape: ", test_labels.shape)


train_labels = F.one_hot(train_labels, num_classes=10)
test_labels = F.one_hot(test_labels, num_classes=10)

print("Train labels shape: ", train_labels.shape)
print("Train labels shape: ", test_labels.shape)


####################################################################
# Dataset Class
####################################################################

class MNIST_dataset(Dataset):

    def __init__(self, images, labels, partition = "train"):

        print("\nLoading MNIST ", partition, " Dataset...")
        self.image_name_list = images
        self.label_list = labels
        self.partition = partition
        print("\tTotal Len.: ", len(self.label_list), "\n", 50*"-")

    def __len__(self):
        return len(self.label_list)
    
    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):

        # Image
        image_tensor = self.image_name_list[idx].float() / 255
        # care! net expect a 784 size vector and our dataset 
        # provide 1x28x28 (channels, height, width) -> Reshape!
        image_tensor = image_tensor.view(-1)

        # Label
        label = self.label_list[idx].float()

        return {"img": image_tensor, "label": label}

train_dataset = MNIST_dataset(train_images, train_labels, partition="train")
test_dataset = MNIST_dataset(test_images, test_labels, partition="test")


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
