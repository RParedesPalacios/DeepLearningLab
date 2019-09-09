# Library import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Constants definition
batch_size = 100
num_classes = 10
epochs = 75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating dataloaders
# ToTensor() - Converts a Image (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
# APPLY SOME DATA AUGMENTATIONS -> HorizontalFLips + Translations + Rotations
# https://pytorch.org/docs/stable/torchvision/transforms.html
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    torchvision.transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

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


# Creating our Neural Network - Convolutional Blocks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = self._CBGN(3, 32)
        self.block2 = self._CBGN(32, 64)
        self.block3 = self._CBGN(64, 128)
        self.block4 = self._CBGN(128, 256)
        self.block5 = self._CBGN(256, 512)
        # Flatten at forward!
        self.features = nn.Linear(512, 512)
        self.reluFeatures = nn.ReLU()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.block1(x)  # 32x32 -> 16x16 @ 32
        out = self.block2(out)  # 16x16 -> 8x8 @ 64
        out = self.block3(out)  # 8x8 -> 4x4 @ 128
        out = self.block4(out)  # 4x4 -> 2x2 @ 256
        out = self.block5(out)  # 2x2 -> 1x1 @ 512
        out = out.view(out.size(0), -1)  # Flatten
        out = self.reluFeatures(self.features(out))  # 512 -> 512
        out = self.classifier(out)  # 512 -> num_classes (10)
        return out

    # DEF A BLOCK Conv + BN + GN + MaxPool
    def _CBGN(self, in_channels, filters):
        layers = []
        layers += [nn.Conv2d(in_channels, filters, kernel_size=(3, 3), padding=1)]
        layers += [nn.BatchNorm2d(filters)]
        layers += [GaussianNoise(0.3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        return nn.Sequential(*layers)


# Instantiating the network and printing its architecture
net = Net().to(device)
print(net)

# Training hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-6)

# Learning Rate Annealing (LRA) scheduling
# lr = 0.1     at epoch < 25
# lr = 0.01    30 <= epoch < 50
# lr = 0.001   epoch >= 50
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)

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
