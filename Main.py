import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
rootDir = "./data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

trainSet = datasets.CIFAR10(
    root=rootDir,
    train=True,
    download=True,
    transform=transform,
)

testSet = datasets.CIFAR10(
    root=rootDir,
    train=False,
    download=True,
    transform=transform,
)

# %%
firstTrainImage, firstTrainLabel = trainSet[0]
print("train data size: ", len(trainSet))
print("train image type: ", type(firstTrainImage))
print("train image shape: ", firstTrainImage.shape)

# %%
print("train min: ", firstTrainImage.data.min())
print("train max: ", firstTrainImage.data.max())

# %%
firstTestImage, firstTestLabel = testSet[0]
print("test data size: ", len(testSet))
print("test image type: ", type(firstTestImage))
print("test image shape: ", firstTestImage.shape)

# %%
print("test min: ", firstTestImage.data.min())
print("test max: ", firstTestImage.data.max())

# %%
batchSize = 500

trainLoader = DataLoader(
    trainSet,
    batch_size=batchSize,
    shuffle=True,
)

testLoader = DataLoader(
    testSet,
    batch_size=batchSize,
    shuffle=True,
)

# %%
print("train batch size: ", len(trainLoader))
print("test batch size: ", len(testLoader))

# %%
nInput = 32 * 32 * 3
nOutput = 10
nHidden = 128


# %%
class CNN(nn.Module):
    def __init__(self, nOutput, nHidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)

        self.maxPool = nn.MaxPool2d((2, 2))
        # self.flatten = nn.Flatten()

        self.l1 = nn.Linear(1152, nHidden)
        self.l2 = nn.Linear(nHidden, nOutput)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.maxPool(self.relu(self.conv1(x)))
        x2 = self.maxPool(self.relu(self.conv2(x1)))
        x3 = torch.flatten(x2, 1)
        x4 = self.l2(self.relu(self.l1(x3)))

        return x4


# %%
cnn = CNN(nOutput, nHidden).to(device)
criterion = nn.CrossEntropyLoss()

# %%
lr = 0.01
optimizer = optim.SGD(cnn.parameters(), lr=lr)

# %%
history = np.zeros((0, 5))

# %%
nEpoch = 100

for epoch in range(nEpoch):
    trainAcc, trainLoss = 0, 0
    valAcc, valLoss = 0, 0
    nTrain, nTest = 0, 0

    for inputs, labels in tqdm(trainLoader):
        nTrain += len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = cnn(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predicted = torch.max(outputs, 1)[1]

        trainLoss += loss.item()
        trainAcc += (predicted == labels).sum().item()

    for inputsTest, labelsTest in testLoader:
        nTest += len(labelsTest)

        inputsTest = inputsTest.to(device)
        labelsTest = labelsTest.to(device)

        outputsTest = cnn(inputsTest)
        lossTest = criterion(outputsTest, labelsTest)

        predictedTest = torch.max(outputsTest, 1)[1]

        valLoss += lossTest.item()
        valAcc += (predictedTest == labelsTest).sum().item()

    trainAcc /= nTrain
    valAcc /= nTest

    trainLoss *= (batchSize / nTrain)
    valLoss *= (batchSize / nTest)

    print(f"Epoch [{epoch + 1}/{nEpoch}], loss: {trainLoss:.5f} acc: {trainAcc:.5f} valLoss: {valLoss:.5f} valAcc: {valAcc:.5f}")

    items = np.array([epoch + 1, trainLoss, trainAcc, valLoss, valAcc])
    history = np.vstack((history, items))

# %%
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(history[:, 0], history[:, 1], "b", label="train")
plt.plot(history[:, 0], history[:, 3], "k", label="test")
plt.xlabel("iter")
plt.ylabel("loss")
plt.title("loss carve")
plt.legend()
plt.show()

# %%
plt.rcParams["figure.figsize"] = (8, 6)
plt.plot(history[:, 0], history[:, 2], "b", label="train")
plt.plot(history[:, 0], history[:, 4], "k", label="test")
plt.xlabel("iter")
plt.ylabel("acc")
plt.title("accuracy")
plt.legend()
plt.show()

# %%
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

for testImages, testLabels in testLoader:
    break

nSize = min(len(testImages), 50)

testInputs = testImages.to(device)
testLabels = testLabels.to(device)

testOutputs = cnn(testInputs)
testPredicted = torch.max(testOutputs, 1)[1]

plt.figure(figsize=(20, 15))

for index in range(nSize):
    ax = plt.subplot(5, 10, index + 1)
    labelName = classes[testLabels[index]]

    if cnn is not None:
        predictedName = classes[testPredicted[index]]
        color = "k" if labelName == predictedName else "b"
        ax.set_title(labelName + ":" + predictedName, c=color, fontsize=20)
    else:
        ax.set_title(labelName, fontsize=20)

    imageNp = testImages[index].numpy().copy()
    image = np.transpose(imageNp, (1, 2, 0))
    image = (image + 1) / 2

    plt.imshow(image)
    ax.set_axis_off()

plt.show()