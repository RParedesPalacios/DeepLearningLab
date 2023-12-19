import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
import torch.nn.functional as  F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print("Torch version: ", torch. __version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

####################################################################
# Defining a Compose my a D.A.
####################################################################

train_transform = transforms.Compose(
                    [
                    transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=2, translate=(0.002,0.001), scale=(0.001, 1.64)),
                    transforms.ToTensor(),
                    ])

test_transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    ])

class MNIST_dataset(Dataset):

    def __init__(self, partition = "train", transform=None):

        print("\nLoading MNIST ", partition, " Dataset...")
        self.partition = partition
        self.transform = transform
        if self.partition == "train":
            self.data = torchvision.datasets.MNIST('.data/', train=True, download=True)
        else:
            self.data = torchvision.datasets.MNIST('.data/', train=False, download=True)
        print("\tTotal Len.: ", len(self.data), "\n", 50*"-")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Image
        image = self.data[idx][0]
        image = self.transform(image)
        # care! net expect a 784 size vector and our dataset 
        # provide 1x28x28 (channels, height, width) -> Reshape!
        image = image.view(-1)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"idx": idx, "img": image, "label": label}

train_dataset = MNIST_dataset(partition="train", transform=train_transform)
test_dataset = MNIST_dataset(partition="test", transform=test_transform)

batch_size = 100
num_workers = multiprocessing.cpu_count()-1
print("Num workers", num_workers)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

class Net(nn.Module):
    def __init__(self, sizes=[[784, 1024], [1024, 1024], [1024, 1024], [1024, 10]], criterion=None):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(sizes)-1):
            dims = sizes[i]
            self.layers.append(nn.Linear(dims[0], dims[1]))
            self.layers.append(nn.BatchNorm1d(dims[1]))
            self.layers.append(nn.ReLU())

        dims = sizes[-1]
        self.classifier = nn.Linear(dims[0], dims[1])

        self.criterion = criterion

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)

        if y != None:
            loss = self.criterion(x, y)
            return loss, x
        return x


# Training Settings
criterion = nn.CrossEntropyLoss()

# Instantiating the network and printing its architecture
num_classes = 10
net = Net(sizes=[
                [784, 1024], 
                [1024, 1024], 
                [1024, 1024], 
                [1024, num_classes]
                ], 
          criterion=criterion)
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)

# Learning Rate Annealing (LRA) scheduling
# lr = 0.1     if epoch < 25
# lr = 0.01    if 25 <= epoch < 50
# lr = 0.001   if epoch >= 50
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)

net = net.to(device)

# Start training
epochs = 75

print("\n---- Start Training ----")
best_accuracy = -1
best_epoch = 0
for epoch in range(epochs):


    # TRAIN NETWORK
    train_loss, train_correct = 0, 0
    net.train()
    with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
        for batch in tepoch:

          images = batch["img"].to(device)
          labels = batch["label"].to(device)
          ids = batch["idx"].to('cpu').numpy()

          # zero the parameter gradients
          optimizer.zero_grad()

          #  Forward
          loss, outputs = net(images, labels)

          loss.backward()

          optimizer.step()

          # one hot -> labels
          labels = torch.argmax(labels, dim=1)
          pred = torch.argmax(outputs, dim=1)

          train_correct += pred.eq(labels).sum().item()

          # print statistics
          train_loss += loss.item()
    
        scheduler.step()
        print("\tLR: ", optimizer.param_groups[0]['lr'])

    train_loss /= len(train_dataloader.dataset)

    # TEST NETWORK
    test_loss, test_correct = 0, 0
    net.eval()
    with torch.no_grad():
      with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
          for batch in tepoch:

            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            ids = batch["idx"].to('cpu').numpy()

            #  Forward
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


# Load best weights
net.load_state_dict(torch.load("best_model.pt"))

test_loss, test_correct = 0, 0
net.eval()
with torch.no_grad():
    with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
        for batch in tepoch:

            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            ids = batch["idx"].to('cpu').numpy()

            #  Forward
            outputs = net(images)
            test_loss += criterion(outputs, labels)

            # one hot -> labels
            labels = torch.argmax(labels, dim=1)
            pred = torch.argmax(outputs, dim=1)

            test_correct += pred.eq(labels).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * test_correct / len(test_dataloader.dataset)
print("Final best acc: ", test_accuracy)







