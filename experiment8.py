import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable


myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#准备数据集
train_dataset = datasets.CIFAR10('./data', train=True, transform=myTransforms, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform=myTransforms, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsampling = False, expansion = 4):
        super(BottleNeck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
#ResNet-50
class ResNet50(nn.Module):
    def __init__(self, blocks, num_classes = 1000, expansion = 4):
        super(ResNet50, self).__init__()
        self.expansion = expansion  
        self.conv1 = nn.Sequential(  #in_planes = 3, places= 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        )
        self.layer1 = self.make_layer(in_channels=64, out_channels=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_channels=256, out_channels=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_channels=512, out_channels=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_channels=1024, out_channels=512, block=blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def make_layer(self, in_channels, out_channels, block, stride):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride, downsampling=True))
        for i in range(1, block):
            layers.append(BottleNeck(out_channels * self.expansion, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#创建ResNet50模型
model = ResNet50(blocks=[3, 4, 6, 3],num_classes=10)
if torch.cuda.is_available():
    model = model.cuda()

#模型训练
max_epoches = 10
epoch = 0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

while epoch < max_epoches:

    model.train()
    correct = 0
    for batch_idx, (data, label) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
            data = Variable(data)
            label = Variable(label)
            out = model(data)
            loss = criterion(out, label)
            _, pred = torch.max(out, dim=1)
            correct += (pred == label).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch += 1
    print(f'Epoch : {epoch}, Train, Accuracy : {correct / len(train_dataset)}')

    #每轮进行一次模型评估
    model.eval()
    eval_correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_dataloader):
            data = data.cuda()
            label = label.cuda()
            data = Variable(data)
            label = Variable(label)
            out = model(data)
            _, pred = torch.max(out, dim=1)
            eval_correct += (pred == label).sum().item()
    print(f'Epoch : {epoch}, Eval, Accuracy : {eval_correct / len(test_dataset)}')



