import torch
import torchvision
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
# Dataset Class
####################################################################

class CIFAR10_dataset(Dataset):

    def __init__(self, partition = "train"):

        print("\nLoading CIFAR10 ", partition, " Dataset...")
        self.partition = partition
        if self.partition == "train":
            self.data = torchvision.datasets.CIFAR10('.data/', 
                                                     train=True,
                                                     download=True)
        else:
            self.data = torchvision.datasets.CIFAR10('.data/', 
                                                     train=False,
                                                     download=True)
        print("\tTotal Len.: ", len(self.data), "\n", 50*"-")

    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Image
        image = self.data[idx][0]
        # PIL Image to torch tensor
        image_tensor = self.from_pil_to_tensor(image)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}

train_dataset = CIFAR10_dataset(partition="train")
test_dataset = CIFAR10_dataset(partition="test")

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

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiating the network and printing its architecture
num_classes = 10
net = SimpleCNN(num_classes)
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

    train_loss /= (len(train_dataloader.dataset) / batch_size)

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

    test_loss /= (len(test_dataloader.dataset) / batch_size)
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
