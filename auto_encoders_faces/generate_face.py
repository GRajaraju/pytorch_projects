import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


# read the image
img = cv2.imread('test.jpg')
cv2.imshow('original img', img)
cv2.waitKey(0)
img = np.mean(img, axis=2)
img = img/255.0
print(img.shape)
#img = np.swapaxes(img, 0, 2)
img = torch.tensor(img, dtype=torch.float)
img = torch.unsqueeze(img, dim=0)
img = torch.unsqueeze(img, dim=0)

print(img.shape)
class ImgGen(nn.Module):
    def __init__(self):
        super(ImgGen, self).__init__()

        # encode layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decode layers
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)

        self.conv_out = nn.Conv2d(32, 1, 3 )


    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        ## decode ##
        
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        
        #x = F.sigmoid(self.conv_out(x))
        x = self.conv_out(x)
                
        return x

model = ImgGen()
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

epochs = 500
out = model(img)
print(out.shape)
for epoch in range(epochs):
    out = model(img)
    optimizer.zero_grad()
    loss = criterion(out, img)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:

        print('epoch: {} --- training loss: {}'.format(epoch,loss.item()))

print(out.shape)
out = torch.squeeze(out, dim=0)
print(out.shape)
out = out.detach()
out = np.swapaxes(out, 0, 2)
print(out.shape)
out = np.array(out)
out = out * 255.0
#print(out)
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()


