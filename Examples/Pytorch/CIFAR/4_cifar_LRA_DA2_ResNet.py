# Library import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# Constants definition
batch_size = 100
num_classes = 10
epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating dataloaders
# ToTensor() - Converts a Image (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
# APPLY SOME DATA AUGMENTATIONS -> HorizontalFLips + Translations + Rotations + Scalation
# https://pytorch.org/docs/stable/torchvision/transforms.html
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(1.0, 1.2)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# Statistics printing
x_batch, y_batch = iter(train_loader).next()
print("Training set: {} samples - Max value: {} - Min value: {}".format(len(train_loader.dataset),
                                                                        x_batch.max(), x_batch.min()))

x_batch, y_batch = iter(test_loader).next()
print("Test set: {} samples - Max value: {} - Min value: {}".format(len(test_loader.dataset),
                                                                    x_batch.max(), x_batch.min()))
print("Example batch shape: {}".format(x_batch.shape))


# There are no GaussianNoise Layer in Pytorch
# https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device).float()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

# Creating our Neural Network - ResNet18
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # resnet connection at forward

        # Initial convolution before resnet blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        ########## 32x32@64
        # RESNET BLOCK 1
        self.b1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b1_bn1 = nn.BatchNorm2d(64)
        self.b1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b1_bn2 = nn.BatchNorm2d(64)
        # RESNET BLOCK 2
        self.b2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2_bn1 = nn.BatchNorm2d(64)
        self.b2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2_bn2 = nn.BatchNorm2d(64)

        ########## 16x16@128
        # RESNET BLOCK 3
        # we need to readapt the input map using 1x1 convolution kernel (like a MLP combining channel dimensions)
        self.b3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.b3_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b3_bn1 = nn.BatchNorm2d(128)
        self.b3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b3_bn2 = nn.BatchNorm2d(128)
        # RESNET BLOCK 4
        self.b4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b4_bn1 = nn.BatchNorm2d(128)
        self.b4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b4_bn2 = nn.BatchNorm2d(128)

        ########## 8x8@256
        # RESNET BLOCK 5
        # we need to readapt the input map using 1x1 convolution kernel (like a MLP combining channel dimensions)
        self.b5_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.b5_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.b5_bn1 = nn.BatchNorm2d(256)
        self.b5_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.b5_bn2 = nn.BatchNorm2d(256)
        # RESNET BLOCK 6
        self.b6_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.b6_bn1 = nn.BatchNorm2d(256)
        self.b6_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.b6_bn2 = nn.BatchNorm2d(256)

        ########## 4x4@512
        # RESNET BLOCK 7
        # we need to readapt the input map using 1x1 convolution kernel (like a MLP combining channel dimensions)
        self.b7_shortcut = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.b7_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.b7_bn1 = nn.BatchNorm2d(512)
        self.b7_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.b7_bn2 = nn.BatchNorm2d(512)
        # RESNET BLOCK 8
        self.b8_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.b8_bn1 = nn.BatchNorm2d(512)
        self.b8_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.b8_bn2 = nn.BatchNorm2d(512)

        ########## 1x1@512
        # Final pooling
        self.average_pooling = nn.AvgPool2d(4)

        ########## 512@num_classes
        # To connect to the number of classes
        self.Linear = nn.Linear(512, num_classes)

    def forward(self, x):
        #### 32x32@3 -> 32x32@64
        # 0. Initial convolution ==> 
        x = F.relu(self.bn1(self.conv1(x)))

        #### 32x32@64 -> 32x32@64
        # 1. First ResNet block
        b1_1 = F.relu(self.b1_bn1(self.b1_conv1(x)))
        b1_2 = self.b1_bn2(self.b1_conv2(b1_1))
        out1 = F.relu(x + b1_2)  # resnet connection plus activation
        # 2. Second ResNet block
        b2_1 = F.relu(self.b2_bn1(self.b2_conv1(out1)))
        b2_2 = self.b2_bn2(self.b2_conv2(b2_1))
        out2 = F.relu(out1 + b2_2)  # resnet connection plus activation

        #### 32x32@64 -> 16x16@128
        # 3. Third ResNet block
        # we need to readapt the number of maps of the input so it matches the output
        shortcut = self.b3_shortcut(out2)
        b3_1 = F.relu(self.b3_bn1(self.b3_conv1(out2)))
        b3_2 = self.b3_bn2(self.b3_conv2(b3_1))
        out3 = F.relu(shortcut + b3_2)  # resnet connection plus activation
        # 4. Fourth ResNet block
        b4_1 = F.relu(self.b4_bn1(self.b4_conv1(out3)))
        b4_2 = self.b4_bn2(self.b4_conv2(b4_1))
        out4 = F.relu(out3 + b4_2)  # resnet connection plus activation

        #### 16x16@128 -> 8x8@256
        # 5. Fifth ResNet block
        # we need to readapt the number of maps of the input so it matches the output
        shortcut = self.b5_shortcut(out4)
        b5_1 = F.relu(self.b5_bn1(self.b5_conv1(out4)))
        b5_2 = self.b5_bn2(self.b5_conv2(b5_1))
        out5 = F.relu(shortcut + b5_2)  # resnet connection plus activation
        # 6. Sixth ResNet block
        b6_1 = F.relu(self.b6_bn1(self.b6_conv1(out5)))
        b6_2 = self.b6_bn2(self.b6_conv2(b6_1))
        out6 = F.relu(out5 + b6_2)  # resnet connection plus activation

        #### 8x8@256 -> 4x4@512
        # 7. Seventh ResNet block
        # we need to readapt the number of maps of the input so it matches the output
        shortcut = self.b7_shortcut(out6)
        b7_1 = F.relu(self.b7_bn1(self.b7_conv1(out6)))
        b7_2 = self.b7_bn2(self.b7_conv2(b7_1))
        out7 = F.relu(shortcut + b7_2)  # resnet connection plus activation
        # 8. Eigth ResNet block
        b8_1 = F.relu(self.b8_bn1(self.b8_conv1(out7)))
        b8_2 = self.b8_bn2(self.b8_conv2(b8_1))
        out8 = F.relu(out7 + b8_2)  # resnet connection plus activation

        #### 4x4@512 -> 1x1@512
        pool_out = self.average_pooling(out8)

        #### 512 -> num_classes
        fc_out = self.Linear(pool_out.view(pool_out.size(0), -1))

        return fc_out


# Instantiating the network and printing its architecture
net = ResNet18().to(device)
print(net)

# Training hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-6)

# Learning Rate Annealing (LRA) scheduling
# lr = 0.1     if epoch < 50
# lr = 0.01    if 30 <= epoch < 100
# lr = 0.001   if epoch >= 100
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

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
            outputs = net(inputs)
            test_loss += criterion(outputs, targets)
            _, pred = outputs.max(1)  # get the index of the max log-probability
            correct += pred.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    # Get current learning rate via the optimizer
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']

    print("[Epoch {}] LR: {:.3f} - Train Loss: {:.5f} - Test Loss: {:.5f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%".format(
            epoch + 1, current_lr, train_loss, test_loss, 100. * train_correct / len(train_loader.dataset), test_accuracy
    ))

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

    scheduler.step()

print('Finished Training')
print("Best Test accuracy: {:.2f}".format(best_accuracy))
