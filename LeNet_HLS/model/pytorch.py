import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import tqdm as tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #28,28,1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
        # 12,12,16
            nn.MaxPool2d(kernel_size=2),
        # 6,6,16
        )
        self.conv2 = nn.Sequential(
        # 6,6,16
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.ReLU(),
        #4,4,16
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Sequential(
            nn.Linear(2 * 2 * 16, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # return x for visualization


model = CNN()
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor()])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=False)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=False)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

if not os.path.exists('model/model.pth'):
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            #inputs = torch.tensor((inputs.cpu().numpy()*255).astype(int))
            # forward + backward + optimize
            #inputs = ((inputs).type(torch.float)).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
else:
    model = torch.load('model/model.pth')
    model.eval()
print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

torch.save(model, 'model/model.pth')
#torch.onnx.export(model, next(iter(test_loader))[0][0], 'model/model.onnx')

file = open('./weights_pytorch.txt', 'w')  # 参数提取
for v in model.parameters():
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    # file.write(str(v.numpy()) + '\n')
    # 将参数扩大1024倍，存为一维数组用逗号隔开
    file.write(",".join(str(i) for i in ((v*1024).cpu().detach().numpy().flatten()).astype(int).tolist()) + '\n')
file.close()
