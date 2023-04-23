import torch
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import audio

data_transforms = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

trainset = datasets.ImageFolder('CONV_DATA', transform=data_transforms)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=7, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (5,5)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5,5)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5,5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20736, 2)
        )

    def forward(self, x):
        return self.model(x)
    
net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
EPOCHS= 15

def train_model():
    for epoch in range(EPOCHS):
        for batch in train_loader:
            X,y = batch
            yhat = net(X)
            loss = loss_fn(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")
        torch.save(net.state_dict(), 'model.pth')

def check(path):
    img = Image.open(path)
    img_transformed = data_transforms(img)
    img_batch = img_transformed.unsqueeze(0)  # Add a batch dimension

    output = net(img_batch)
    predicted_class = torch.argmax(output)

    if predicted_class.item() == 0:
        return 'Neel'

    else:
        return 'Vedant'

def get_person():

    net.load_state_dict(torch.load('model.pth'))

    cap = cv2.VideoCapture(0)

    start_time = time.time()

    ct = 0

    while time.time() - start_time < 20:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if ct == 5:
            cv2.imwrite('frame.jpg', frame)

            person = check('frame.jpg')

            break

        ct += 1

    cap.release()
    cv2.destroyAllWindows()
    print(person)
    audio.say(person)