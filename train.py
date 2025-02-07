import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np 


from torchvision.datasets import MNIST
from models import *




# # torch.manual_seed(42)       
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# device = 'cpu'    




"""
Parameters of te model
"""
num_ep = 100
batch_size = 64
loginterval = 10
lr = 0.001


    

"""
Initialize dataset
"""
dset_train = MNIST(root='./', train=True, transform=torchvision.transforms.ToTensor(), download=True)
dset_test = MNIST(root='./', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False)


"""
Initialize model and optimizer
"""
model = MNIST_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)




"""
Training loop
"""
loss_fn = torch.nn.NLLLoss()
def train(epoch, data_loader, mode='train'):
    if mode == 'train':
        for batch_idx, (x, y) in enumerate(data_loader):
            
            model.train()
            optimizer.zero_grad()

            x = x.to(device) / 10.
            y = y.to(device)

            out = model(x)
            # print(out.shape, x.shape, y.shape)
            # print(model.get_weights().shape)
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()

            if batch_idx % loginterval == 0:
                print(f"Train epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3}")
                
    elif mode=='test':
        tot_size = 0
        tot_acc = 0.
        for batch_idx, (x, y) in enumerate(data_loader):     

            model.eval()
            x = x.to(device)  / 10.
            y = y.to(device)

            out = torch.argmax(model(x), dim=-1)
            tot_acc += (out == y).sum().item()
            tot_size += x.shape[0]

        acc = tot_acc / tot_size

        print(f"Test epoch: {epoch}, Accuracy: {acc:.3}")

        weights = model.get_weights().detach().cpu().numpy()
        # print((weights < 1.e-5).sum())
        plt.figure()
        plt.hist(np.abs(weights), bins=500, color='tab:cyan')
        plt.show()





if __name__ == "__main__":
    for i in range(1, num_ep + 1):
        print(f'Epoch {i}')
        train(i, train_loader, mode='train')
        train(i, test_loader, mode='test')
