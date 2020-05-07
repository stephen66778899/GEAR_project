from torch.utils.data.dataset import Dataset
from PIL import Image
import csv
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MyCustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.to_tensor = transforms.ToTensor() 
        self.file_label_list = []
        with open(csv_file, newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                self.file_label_list.append(row)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.file_label_list[index][1]
        single_image_name = self.file_label_list[index][0]
        img_as_img = Image.open(single_image_name)
        img_as_tensor = self.to_tensor(img_as_img)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.file_label_list)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(238144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 238144)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

trainset = MyCustomDataset(csv_file='train.csv', root_dir='OOWL_in_the_wild')
#img, label = myDataset.__getitem__(10)
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
'''
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
'''
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data[0].to(device), data[1].to(device)
        inputs = data[0].to(device)
        labels = data[1]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        print(inputs.shape)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
