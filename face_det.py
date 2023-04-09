import torch
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import pickle
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dir = 'CONV_DATA'

process = False


#RANDOM_SEED = 47

# Grab the data

image_paths = []
labels = []

counter = 0


label = 0

def get_data():
    for folder in os.listdir(dir):
        process = True
        for file in os.listdir(os.path.join(dir, folder)):
            path = os.path.join(dir, folder, file)
            image = Image.open(path)
            resized_image = image.resize((400, 400))
            resized_image = resized_image.convert("RGB")
            resized_image.save(os.path.join(dir, folder, f'{counter}.jpg'))
            image_paths.append(os.path.join(dir, folder, f'{counter}.jpg'))
            labels.append(label)
            counter += 1
        label += 1

if process == False:
    for folder in os.listdir(dir):
        for file in os.listdir(os.path.join(dir, folder)):
            path = os.path.join(dir, folder, file)
            image_paths.append(path)
            labels.append(label)
            counter += 1
        label += 1

# Shuffle the data

#random.seed(RANDOM_SEED)

#random.shuffle(image_paths)
#random.shuffle(labels)

resized_paths = []

# Now let's turn the images into tensors

transform = transforms.ToTensor()

imgs = []

for i in image_paths:
    tensor = transform(Image.open(i))
    imgs.append(tensor)

print(labels)

set_y = []
for i in labels:
    if i == 0:
        set_y.append(torch.tensor([1, 0]))
    else:
        set_y.append(torch.tensor([0, 1]))

set_x = []
for i in imgs:
    set_x.append(i.view(480000))

set_x.reverse()
set_y.reverse()

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)

        return x

def train_model():
    model = Model(480000, 48, 2)    

    EPOCHS = 100
    LEARNING_RATE = 0.0001

    EPOCHS = 60

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for i in range(len(set_x)):
            x = set_x[i]
            y = set_y[i]
            model.zero_grad()
            output = model(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            
        print(loss)


    torch.save(model.state_dict(), 'model.pth')


#train_model()

net = Model(480000, 48, 2)    
net.load_state_dict(torch.load('model.pth'))

INDEX = 15

path = image_paths[INDEX]
image = Image.open(path)
image.show()
print(set_y[INDEX])
net.eval()
with torch.no_grad():
    output = (net(set_x[INDEX])).tolist()
    print(output)