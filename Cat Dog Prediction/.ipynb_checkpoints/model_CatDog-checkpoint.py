import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets,transforms,models


import os

from torch.utils.data import random_split,DataLoader,Dataset

  # Applying Transformation
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Lambda(lambda img: img.convert("L")),   #L mean to GrayScale   RCB mean to Colorful image
    transforms.ToTensor(),   #dont have to use permute
    transforms.Normalize([0.5]*1, [0.5]*1)   #*1 cause grayscale have 1 chanel only 3 if rgb
])
 

datas = datasets.ImageFolder(root="cat vs Dog",transform=transform)


train_size = int(0.8 * len(datas))
test_size = len(datas) - train_size

train_data, test_data = random_split(datas, [train_size, test_size])

train_data = DataLoader(train_data,batch_size=16,drop_last=False,shuffle=True,pin_memory=True)
test_data = DataLoader(test_data,batch_size=16,drop_last=False,shuffle=False,pin_memory=True)

print(f"Train Batch : {len(train_data)}")

print("Test Batch : ",len(test_data))

for img,label in train_data:
    print(img.shape)
    break
# Format ===> (batch,chunk = Grayscale ,height,width

# Image in each block
i=0
for img,label in train_data:
    print(f"Image Batch{i+1}", img)
    i+=1


# label in each block
i=0
for img,label in train_data:
    print(f"Image Labels Batch{i+1}", label)
    i+=1

i = 0
for img, label in train_data:
    print(f"Image {i+1} shape:", img.shape)
    i += 1
    break # Format need to change for  (batch,chunk,h,w)  ====>(batch,h,w,c)

images, labels = next(iter(train_data))

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('off')

plt.show()   

class CNN_Architecture(nn.Module):
    def __init__(self):
        super() .__init__()


        # cnn architecture


        # Key formulas  == ⌊N+2P−K/S⌋+1   new shape 
        # Max pool   (n-k/s)+1

        self.model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
                                    # Input: (1, 100, 100)
                                    # Conv: 100 - 3 + 1 = 98 → (16, 98, 98)
                                    
                                    nn.ReLU(),
                                    
                                    nn.MaxPool2d(2),
                                    # Pool: 98 / 2 = 49 → (16, 49, 49)
                                    
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                                    # Conv: 49 - 3 + 1 = 47 → (32, 47, 47)
                                    
                                    nn.ReLU(),
                                    
                                    nn.MaxPool2d(2),
                                    # Pool: 47 / 2 = 23.5 → floor → 23 → (32, 23, 23)
                                    
                                    # Final tensor = (32, 23, 23)
                                    
                                    # Flatten:
                                    # 32 × 23 × 23 = 16928
                                    
                                    nn.Flatten(),
                                    nn.Linear(32*23*23,128),
                                    nn.ReLU(),
                                    nn.Linear(128,2)
                                  )


    def forward(self,x):
        return self.model(x)
        

# using model 

# 1st defining model

model = CNN_Architecture()


# PREPARING GPU for Fast Accleration

device = ('cuda' if torch.cuda.get_device_name() else 'cpu')


# connectiong model and GPU 

model.to(device)


# Declaring Loss function

criterion = nn.CrossEntropyLoss()


# optimizer
optimizer = optim.Adam(model.parameters(),lr=1e-3)

# Declaring Epoch for training
epochs  =15     

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):


    model.train()
    total_loss = 0

    for X, y in train_data:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    model.eval()
    total_loss_val = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_val, y_val in test_data:
            X_val, y_val = X_val.to(device), y_val.to(device)

            y_pred_val = model(X_val)
            loss_val = criterion(y_pred_val, y_val)

            total_loss_val += loss_val.item()

            # Accuracy
            _, preds = torch.max(y_pred_val, 1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)

    avg_loss_val = total_loss_val / len(test_data)
    accuracy = 100 * correct / total

    val_losses.append(avg_loss_val)
    val_accuracies.append(accuracy)

    print(f"Validation Loss: {avg_loss_val:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("="*60)


plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()


# Few Parameter Tune


class CNN_Architecture(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

         
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

       
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN_Architecture()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 15

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X, y in train_data:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = criteration(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    total_loss_val = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_val, y_val in test_data:
            X_val, y_val = X_val.to(device), y_val.to(device)

            y_pred_val = model(X_val)
            loss_val = criteration(y_pred_val, y_val)

            total_loss_val += loss_val.item()

            _, preds = torch.max(y_pred_val, 1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)

    avg_loss_val = total_loss_val / len(test_data)
    accuracy = 100 * correct / total

    val_losses.append(avg_loss_val)
    val_accuracies.append(accuracy)

    print(f"Validation Loss: {avg_loss_val:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("="*60)

plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()

device = ('cuda' if torch.cuda.get_device_name() else 'cpu')

device


# Now using RGB on same PArameter
# Applying Transformation
transform = transforms.Compose([
    transforms.Resize((130, 130)),
    transforms.Lambda(lambda img: img.convert("RGB")),   
    transforms.ToTensor(),   #dont have to use permute
    transforms.Normalize([0.5]*3, [0.5]*3)   
])


datas = datasets.ImageFolder(root="cat vs Dog",transform=transform)



train_size = int(0.8 * len(datas))
test_size = len(datas) - train_size

train_data, test_data = random_split(datas, [train_size, test_size])



train_data = DataLoader(train_data,batch_size=16,drop_last=False,shuffle=True,pin_memory=True)
test_data = DataLoader(test_data,batch_size=16,drop_last=False,shuffle=False,pin_memory=True)



i = 0
for img, label in train_data:
    print(f"Image {i+1} shape:", img.shape)
    i += 1
    break


    # Format need to change for  (batch,chunk,h,w)  ====>(batch,h,w,c)

images, labels = next(iter(train_data))

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i][0])
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('off')

plt.show()

class CNN_Architecture(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Flatten(),

            nn.Linear(128, 256),       
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN_Architecture()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 15

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X, y in train_data:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = criteration(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    total_loss_val = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_val, y_val in test_data:
            X_val, y_val = X_val.to(device), y_val.to(device)

            y_pred_val = model(X_val)
            loss_val = criteration(y_pred_val, y_val)

            total_loss_val += loss_val.item()

            _, preds = torch.max(y_pred_val, 1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)

    avg_loss_val = total_loss_val / len(test_data)
    accuracy = 100 * correct / total

    val_losses.append(avg_loss_val)
    val_accuracies.append(accuracy)

    print(f"Validation Loss: {avg_loss_val:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("="*60)

plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()

# USing VGG 16 pretrained Model

# Mean ,Std (Normalization)  and shape shoud be constant cause model cnn feature exytacter is trained on same dataset

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.Lambda(lambda  img :img.convert('RGB')),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.402],std=[0.229,0.224,0.225]),
                                
                               ])
                                
datas = datasets.ImageFolder(root="cat vs Dog",transform=transform)                                                

# Train Validation Split
# 8:2 ratio

train_size = int(0.8 * len(datas))
test_size = len(datas) - train_size

train_data, test_data = random_split(datas, [train_size, test_size])

# Batching Data
train_data = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
test_data = DataLoader(test_data,batch_size=32,shuffle=False,num_workers = 4 ,pin_memory=True)

print(f"Total Train Batch {len(train_data)}       Total Test Data {len(test_data)}")

i = 1
for img,target in train_data:
    print(f"Batch : {i}   Targets {target}")
    i+=1
    
i = 1
for img,target in train_data:
    if i < 3:
        print(f"Batch : {i}   Targets {img}")
        i+=1

    else:
        break
# USing Break cause data is to large

model = models.vgg16(weights="IMAGENET1K_V1")

print(model)

from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))      #batch is give 1  as a dummy batch to visulize the dataStructure (MOdel structure)


# Wt and bias of model

i=1
for paras in model.parameters():
    if i <3:
        print(paras)
        i+=1
    else:
        break

print(model.classifier)

for name, param in model.named_parameters():
    print(name, param.shape)

print(model.features)

# Disabiling Model Features mean CNN layers

for param in model.features.parameters():
    param.requires_grad =False


for param in model.classifier.parameters():
    param.requires_grad = False
    

for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Building own classifier



# WE HAVE modified  no of neuron to prevent over fit and final layer cause we have only 2 output class

model.avgpool = nn.AdaptiveAvgPool2d((1,1))

model.classifier = nn.Sequential(
    nn.Linear(512, 128), 
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)
summary(model, input_size=(1, 3, 224, 224)) 

model.to(device)


criteration = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr = 1e-4)


epochs = 15

train_losses = []
val_losses = []

for epoch in range (epochs):
    model.train()
    total_loss = 0
    for x,y in train_data:
        x = x.to(device)
        y = y.to(device)
# Flow --> y_hat ----->loss ----->optimize(Backward pass)
        
        # y_hat
        y_pred = model(x)

        # loss
        loss = criteration(y_pred,y)

        # Reset all wt and bias before updating wt and bias in next back prop
        optimizer.zero_grad()
        
        # backpass
        loss.backward()

        # updating wt and bias

        optimizer.step()


        # Calculating total loss on that batch
        total_loss +=loss.item()


    avg_loss = total_loss/len(train_data)
    train_losses.append(avg_loss)
    
    print("=="*20)
    print(f"Train  Loss: {avg_loss:.4f}")

    


    # Now runnign val loop sideby side to check model Status
    model.eval()
    total_loss_val = 0
    
    with torch.no_grad():
        for X,Y in test_data:
            X = X.to(device)
            Y = Y.to(device)
    
            # Same flow but in evaluation we dont do backpass we use obtain wt from backpass in forward pass
    
            # y_hat -->loss
    
            y_hat = model(X)
    
    
            # Loss
            loss = criteration(y_hat,Y)
    
    
            total_loss_val += loss.item()


        avg_loss_val = total_loss_val/len(test_data)
        val_losses.append(avg_loss_val)

        
        print(f"Validation Loss: {avg_loss_val:.4f}")
        print("=="*20)
        



    model.train()

        
plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()
        
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for X, y in test_data:
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        prob = torch.softmax(outputs, dim=1)[:,1]
        probs.extend(prob.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(all_labels, all_preds))

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()
        
# Visulizing where model make mistake in which image

import matplotlib.pyplot as plt

model.eval()

images, labels = next(iter(test_data))

with torch.no_grad():
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)

for i in range(5):
    plt.imshow(images[i].permute(1,2,0))
    plt.title(f"Pred: {preds[i].item()} | True: {labels[i].item()}")
    plt.show()

# Visulizing Only Mistakes



images, labels = next(iter(test_data))

with torch.no_grad():
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)

for i in range(len(images)):
    if preds[i] != labels[i]:   
        plt.imshow(images[i].permute(1,2,0))
        plt.title(f"Pred: {preds[i].item()} | True: {labels[i].item()}")
        plt.show()


# Checking class Imbalance

from collections import Counter
labels = Counter(all_labels)
print(labels)

from sklearn.metrics import roc_curve, auc

probs = []
model.eval()

with torch.no_grad():
    for X, _ in test_data:
        X = X.to(device)
        outputs = model(X)
        prob = torch.softmax(outputs, dim=1)[:,1]
        probs.extend(prob.cpu().numpy())

fpr, tpr, _ = roc_curve(all_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}",marker = "*")
plt.plot([0,1], [0,1], linestyle='--')
plt.legend()
plt.title("ROC Curve")
plt.show()


# Let X be investment and Y be profit and chose a profit where investment is less


def predict(model, image, device):
    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return pred.item()


from PIL import Image
# Prediction Flow
# load img --->apply transformation ------>give image Gpu

image = Image.open("cat vs Dog/cat_test.jpg").convert("RGB")

image = transform(image)

image = image.to(device)


y_pred = predict(model,image,device)

if y_pred == 0:
    labels = 'Cat'

else:
    labels = 'Dog'


print(f"Given image is identified as : {labels}")

plt.imshow(image.cpu().permute(1, 2, 0))
plt.title("Test Image")
plt.show()


# Sving model wt and bisa so we dont have to Train again and again
torch.save(model.state_dict(), "cat vs Dog/model.pth")

import os
os.makedirs("saved_models", exist_ok=True)

