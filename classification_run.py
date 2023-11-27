import torch
from scipy.io.matlab import loadmat
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device: "+str(device))


labels=loadmat("./imagelabels.mat")["labels"]
model = torch.hub.load("pytorch/vision", "resnet50", weights="DEFAULT")

model.avgpool = nn.AdaptiveAvgPool2d(1)
model.fc = nn.Linear(in_features=2048, out_features=103)

PATH = './model_dict.pth'
"""if os.path.exists(PATH):
    print("loading pretrained model")
    model.load_state_dict(torch.load(PATH))"""
transform=transforms.Compose([transforms.Resize([500,500]),transforms.ToTensor()])
dataset_list=[transform(Image.open(os.path.join("./jpg",path))) for path in os.listdir("./jpg/")]

test_loader = torch.utils.data.DataLoader([(dataset_list[i-1],labels[0][i-1]) for i in loadmat("./setid.mat")["trnid"][0]], batch_size=4)
val_loader = torch.utils.data.DataLoader([(dataset_list[i-1],labels[0][i-1]) for i in loadmat("./setid.mat")["valid"][0]], batch_size=4)
train_loader = torch.utils.data.DataLoader([(dataset_list[i-1],labels[0][i-1]) for i in loadmat("./setid.mat")["tstid"][0]], batch_size=96, shuffle=True)

print("train set:\t"+str(len(train_loader))+" total elements with batch size of "+str(train_loader.batch_size))
print("validation set:\t"+str(len(val_loader))+" total elements with batch size of "+str(val_loader.batch_size))
print("test set:\t"+str(len(test_loader))+" total elements with batch size of "+str(test_loader.batch_size))


plist = [
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 5e-3}
        ]
optimizer_ft = optim.Adam(plist, lr=0.00001)
lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(plist, lr=0.00001, momentum=0.9)

model.train()
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    
    tqdm_data=tqdm(enumerate(train_loader, 0),desc="epoch "+str(epoch+1),total=len(train_loader),leave=False)
    for i, data in tqdm_data:
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0], data[1]

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward + backward + optimize
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()

        tqdm_data.set_postfix(loss= loss.item(), refresh=False)
        # print statistics
        running_loss += loss.item()

    lr_sch.step()
    torch.save(model.state_dict(), PATH)
print('Finished Training')

#from matplotlib.pyplot import imshow
#import torchvision

model.eval()
it=iter(val_loader)
pred=list()
true_labels=list()
class_accuracy={str(i+1):{"true":0, "total":0} for i in range(102)}
accuracy=0
for i, data in tqdm(enumerate(val_loader, 0),total=len(val_loader)):
    val_images, val_labels = data[0], data[1]

    #imshow(torchvision.utils.make_grid(val_images).permute(1,2,0))
    #print('GroundTruth: ', ' '.join(f'{str(val_labels[j].item()):5s}' for j in range(4)))
    outputs = model(val_images)
    _, predicted = torch.max(outputs, 1)
    #print('Predicted: ', ' '.join(f'{str(predicted[j].item()):5s}' for j in range(4)))
    for b in range(4):
        if val_labels[b].item()==predicted[b].item():
            accuracy+=1
            class_accuracy[str(val_labels[b].item())]["true"]+=1
        class_accuracy[str(val_labels[b].item())]["total"]+=1
        #true_labels.append(val_labels[b].item())
        #pred.append(predicted[b].item())
accuracy=accuracy/(4*len(val_loader))
print("Accuracy:\t"+str(accuracy))
print("\nClasses accuracy:\t"+str(class_accuracy))
