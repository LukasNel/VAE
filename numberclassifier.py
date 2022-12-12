from pl_bolts.datamodules import CIFAR10DataModule, TinyCIFAR10DataModule
from torchvision import transforms
import torchvision.datasets as datasets
import wandb
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision.utils import make_grid
import numpy as np
from matplotlib.pyplot import imshow, figure, clf
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
# from tqdm.notebook import trange, tqdm # used for fancy status bars
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
from tqdm import tqdm
class Number_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 20)
        # These are PyTorch's predefined layers. Each is a class. In the "init" function, we just initialize and instantiate the classes, creating objects (that behave like functions)
        self.layer2 = nn.Linear(20, 10)
        self.nonlin = nn.ReLU()
        # self.softmax = nn.Softmax() # Converts numbers into probabilities

    def forward(self, x):
        x = self.layer1(x)  # Composing the functions we created below
        x = self.nonlin(x)
        x = self.layer2(x)
        return x

loss_fn = nn.CrossEntropyLoss()
def get_accuracy(output, targets):
    output = output.detach() # this removes the gradients associated with the tensor
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / output.size(0) * 100
    return accuracy
def plot_accuracies_and_loss(train_accs, test_accs, losses):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(train_accs, marker='', color='skyblue', linewidth=2,label="Training Accuracy")
    ax[0].plot(test_accs, marker='', color='olive', linewidth=2,label="Testing Accuracy")
    ax[0].legend()
    ax[0].set_title("Accuracies")
    ax[1].set_title("Loss")
    ax[1].plot(losses, marker='', color='green', linewidth=2, label="Loss")
    plt.show()

tensor_transform = transforms.ToTensor()
mnist_trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=tensor_transform
)
mnist_testset = datasets.MNIST(
    root="./data", train=False, download=True, transform=tensor_transform
)
maxtrain = torch.max(mnist_trainset.data)
maxtest = torch.max(mnist_testset.data)
# load into torch datasets
# To speed up training, we subset to 10,000 (instead of 60,000) images. You can change this if you want better performance.
train_dataset = torch.utils.data.TensorDataset(mnist_trainset.data.to(
    dtype=torch.float32)[:10000]/maxtrain, mnist_trainset.targets.to(dtype=torch.long)[:10000])
test_dataset = torch.utils.data.TensorDataset(mnist_testset.data.to(
    dtype=torch.float32)/maxtest, mnist_testset.targets.to(dtype=torch.long))

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True
)

print(device)
if __name__ == '__main__':
    t = tqdm(total=80,desc="Test Accuracy")
    
    # While training, just invoke the command, and this new optimizer will run SGD with the most recent gradient information.
    for modelnum in tqdm(range(28,500),desc="Model"):
        classifier = Number_Classifier() # create an instance of our neural net object
        classifier.to(device) # If a GPU is available
        opt = torch.optim.SGD(classifier.parameters(), lr=0.01) # This is how PyTorch does SGD -- by handing all of the model's parameters to an optimizer class. 
        train_accs, test_accs, losses = [], [], [] # lists to keep track of model stats
        t.reset()
        current_test = 0
        for e in range(1000): # this is the number of epochs to train -- each epoch iterates through the entire dataset.
            # print(e)
            classifier.train()
            for images, labels in trainloader:
                images = images.reshape(-1,784) # MNIST images are 28x28 matrices. This compresses them to (really long) vectors, so they can be successfully fed to the network.
                images = images.to(device) # move them to the GPU before handing them to the model (which is on the GPU)
                labels = labels.to(device)
                y = classifier(images) # these are the classification probabilities
                l = loss_fn(y,labels)
                # print(l)
                l.backward()
                opt.step() # Run SGD
                opt.zero_grad() # We've used the gradients, so reset them to zero (otherwise they'll accumulate ad naseum)
            # after running through the entire dataset, we can evaluate the accuracy of the model
            classifier.eval() # disable gradient computation while evaluating, since we won't be backpropogating
            train_ep_pred = classifier(mnist_trainset.data.to(dtype=torch.float32).reshape(-1,28*28).to(device))
            test_ep_pred = classifier(mnist_testset.data.to(dtype=torch.float32).reshape(-1,28*28).to(device))

            train_accuracy = get_accuracy(train_ep_pred.cpu(), mnist_trainset.targets.to(dtype=torch.long))
            test_accuracy = get_accuracy(test_ep_pred.cpu(), mnist_testset.targets.to(dtype=torch.long))
            # Add model stats to running list
            train_accs.append(train_accuracy)
            test_accs.append(test_accuracy)
            losses.append(l.detach().cpu())
            
            # if e % 10 == 0:
                # print(f"Training Accuracy: {train_accuracy} - Test Accuracy: {test_accuracy}")

            if test_accuracy > 80:
                break
            # print(test_accuracy)
            # t.display("Test Accuracy")
            t.update(round(test_accuracy-current_test,2))
            current_test = test_accuracy
        torch.save(classifier.state_dict(), "new_data/model_{}.pth".format(modelnum + 1))
        # plot_accuracies_and_loss(train_accs, test_accs, losses) 

