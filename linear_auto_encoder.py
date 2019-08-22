import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor(torch.arange(10), dtype=torch.float)

# define the network
class AutoEncode(nn.Module):
    
    def __init__(self):
        super(AutoEncode, self).__init__()

        self.encode = nn.Linear(10, 4)
        self.decode = nn.Linear(4, 10)
        pass

    def forward(self, x):
        encode = F.relu(self.encode(x))
        decode = self.decode(encode)
        return decode


net = AutoEncode()

epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(epochs):
    out = net(x)
    
    optimizer.zero_grad()
    loss = criterion(out, x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch: {} --- loss: {}'.format(epoch, loss.item()))

print('output: ', out)

new_data = torch.tensor(torch.arange(10), dtype=torch.float)

test_data = reversed(new_data)
test_output = net(test_data)
print(test_output)

as_x = torch.tensor(torch.arange(10), dtype=torch.float)
out_x = net(as_x)
print(out_x)


   

