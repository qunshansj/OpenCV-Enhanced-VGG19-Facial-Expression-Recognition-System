
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torchvision import models as MD
import torch
import torchvision.datasets as datasets
 
 
def  Make_data(path):
    img=datasets.ImageFolder(path,
                         tensform=transforms.Compose([transforms.Scale([48, 48]), transforms.ToTensor()]))
    imgs_load=data.DataLoader(img,batch_size=100,shuffle=True)
    return imgs_load
 
 
def train(dada_loader):
    model = MD.vgg19(pretrained=False)
    model.load_state_dict(torch.load("../models/???.pth"))
    num_input = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_input, 7))
    model.classifier = nn.Sequential(*feature_model)
    model = model.cuda()
    critersion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
 
    for ench in range(200):
        sum = 0
        for i, data in enumerate(dada_loader):
            img, targe = data[1]
            targe = targe.cuda()
            img = img.cuda()
            output = model(img)
            loss = critersion(output, targe)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum += loss
        print(sum)
        if ench % 20 == 0:
            torch.save(model.state_dict(), "../models/" + str(ench) + ".pkl")
 
 
def test(dada_loader):
    model = MD.vgg19(pretrained=False)
    num_input = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_input, 7))
    model.classifier = nn.Sequential(*feature_model)
    # 加载训练过的模型进行测试
    model.load_state_dict(torch.load(""))
    model = model.cuda()
    for i, data in enumerate(dada_loader):
        img, targe = data[1]
        targe = targe.cuda()
        img = img.cuda()
        output = model(img)
        _, pred = torch.max(output.data, 1)
        print(torch.sum(pred == targe))
 
 
if __name__ == '__main__':
    trainpath="../train"
    trainimg=Make_data(trainpath)
    train(trainimg)
 
 
    testpath = "../test"
    testimg = Make_data(trainpath)
    test(testimg)
