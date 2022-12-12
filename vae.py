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
from numberclassifier import Number_Classifier


pl.seed_everything(1234)
logger = WandbLogger(name="VAE", project="VAETesting")

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


def get_accuracy(output, targets):
    output = output.detach()  # this removes the gradients associated with the tensor
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / output.size(0) * 100
    return accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
figure(figsize=(8, 3), dpi=300)


class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        clf()
        # Z COMES FROM NORMAL(0, 1)
        rand_v = torch.randn(
            (self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        p = torch.distributions.Normal(
            torch.zeros_like(rand_v), torch.ones_like(rand_v))
        z = p.rsample()

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = pl_module.decoder(z.to(pl_module.device)).cpu()
        print("pred", pred.shape)
        pred = pred.reshape((16, 2*43, 5*37))
        # UNDO DATA NORMALIZATION
        normalize = cifar10_normalization()
        mean, std = np.array(normalize.mean), np.array(normalize.std)
        # img = make_grid(pred).numpy() #* std + mean
        samples = [wandb.Image(img) for img in pred]
        # PLOT IMAGES
        wandb.log({"images": samples})


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




class NumberSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        clf()
        # Z COMES FROM NORMAL(0, 1)
        rand_v = torch.randn(
            (self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        p = torch.distributions.Normal(
            torch.zeros_like(rand_v), torch.ones_like(rand_v))
        z = p.rsample()

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = pl_module.decoder(z.to(pl_module.device)).cpu()
        test_accuracy_out = []
        for img in pred:
            weight_1_matrix, weight_2_matrix, bias_1_matrix, bias_2_matrix = torch.split(
                img, (20 * 784, 10 * 20, 20, 10))
            classifier = Number_Classifier()

            classifier_load = {}
            classifier_load["layer1.weight"] = weight_1_matrix.reshape((20,784))
            classifier_load["layer2.weight"] = weight_2_matrix.reshape((10,20))
            classifier_load["layer1.bias"] = bias_1_matrix
            classifier_load["layer2.bias"] = bias_2_matrix

            classifier.load_state_dict(classifier_load, strict=True)
            train_ep_pred = classifier(mnist_trainset.data.to(dtype=torch.float32).reshape(-1,28*28).to(device))
            test_ep_pred = classifier(mnist_testset.data.to(dtype=torch.float32).reshape(-1,28*28).to(device))

            train_accuracy = get_accuracy(train_ep_pred.cpu(), mnist_trainset.targets.to(dtype=torch.long))
            test_accuracy = get_accuracy(test_ep_pred.cpu(), mnist_testset.targets.to(dtype=torch.long))
            print("test_accuracy", test_accuracy, "train_accuracy", train_accuracy)
            test_accuracy_out.append(test_accuracy)

        
        sumnp = pred.numpy().sum()
        print("sumnp", sumnp)
        avgtest = np.average(np.array(test_accuracy_out))
        print("avg test_accuracy", avgtest)
        wandb.log({"test_accuracy": avgtest})
        wandb.log({"sumofweights": sumnp})


# For the afficianados: the (nn.Module) subclasses PyTorch's neural network superclass, which, when initialized below...
class Encode(nn.Module):
    def __init__(self, input_height):
        # <--- does a bunch of janitorial work to make our network easier to use. For example, once our network is initialized, calling IttyBittyNetwork(data) passes the data into forward.
        super().__init__()
        self.input_height = input_height
        self.encoding_model = nn.Sequential(
            nn.Linear(input_height, 400),
            nn.ReLU(),
            # Initializing the classes adds their free variables to our network's list of parameters to update via gradient descent.
            nn.Linear(400, 200),
            nn.ReLU(),
            # Initializing the classes adds their free variables to our network's list of parameters to update via gradient descent.
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),  # Note this ends with a 10 dimensional output.
            nn.ReLU(),
            nn.Linear(50, 40),
        )
        # self.softmax = nn.Softmax() # Converts numbers into probabilities

    def encode(self, x):
        # print("shape", x.shape)
        # x = torch.reshape(x, (-1, self.input_height * self.input_height * 3))
        x = self.encoding_model(x)
        return x

    def forward(self, x):
        y = self.encode(x)
        return y


# For the afficianados: the (nn.Module) subclasses PyTorch's neural network superclass, which, when initialized below...
class Decode(nn.Module):
    def __init__(self, input_height):
        # <--- does a bunch of janitorial work to make our network easier to use. For example, once our network is initialized, calling IttyBittyNetwork(data) passes the data into forward.
        super().__init__()
        self.nonlin = nn.ReLU()
        # self.softmax = nn.Softmax() # Converts numbers into probabilities
        self.decoding_model = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, input_height),
            nn.ReLU(),
        )
        self.sig = nn.ReLU()
        self.input_height = input_height

    def decode(self, x):
        x = self.decoding_model(x)
        # x = torch.reshape(x, (-1, 3, self.input_height, self.input_height))
        return x

    def forward(self, x):
        x = self.decode(x)
        return x


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=40, latent_dim=20, input_height=15910):
        super().__init__()

        self.save_hyperparameters()
        device = torch.device("cpu")
        self.encoder = Encode(input_height)
        self.encoder.to(device)
        self.decoder = Decode(input_height)
        self.decoder.to(device)
        print("cuda", device)
        # encoder, decoder
        # self.encoder = resnet18_encoder(False, False)
        # self.decoder = resnet18_decoder(
        #     latent_dim=latent_dim,
        #     input_height=input_height,
        #     first_conv=False,
        #     maxpool1=False
        # )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_mu.to(device)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var.to(device)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.log_scale.to(device)
        device

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        mean.to(device)
        scale.to(device)

        dist = torch.distributions.Normal(mean, scale)
        x.to(device)
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        # print("log_pxz", log_pxz.shape)
        return log_pxz.sum(dim=(1))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x.to(device)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)
        x_hat.to(device)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        wandb.log({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo


model_params = []

for i in range(500):
    classifier_load = torch.load(
        "new_data/model_{}.pth".format(i+1), map_location='cuda')
    # print(classifier_load)
    weight_1_matrix = classifier_load["layer1.weight"]
    weight_2_matrix = classifier_load["layer2.weight"]
    bias_1_matrix = classifier_load["layer1.bias"]
    bias_2_matrix = classifier_load["layer2.bias"]
    print(weight_1_matrix.shape)
    print(weight_2_matrix.shape)
    print(bias_1_matrix.shape)
    print(bias_2_matrix.shape)
    
    weight_1_matrix = weight_1_matrix.reshape(20*784)
    weight_2_matrix = weight_2_matrix.reshape(200)

    classifier = Number_Classifier()

    classifier_load_test = {}
    classifier_load_test["layer1.weight"] = weight_1_matrix.reshape((20,784))
    classifier_load_test["layer2.weight"] = weight_2_matrix.reshape((10,20))
    classifier_load_test["layer1.bias"] = bias_1_matrix
    classifier_load_test["layer2.bias"] = bias_2_matrix

    classifier.load_state_dict(classifier_load, strict=True)
    train_ep_pred = classifier(mnist_trainset.data.to(dtype=torch.float32).reshape(-1,28*28).to(device))
    test_ep_pred = classifier(mnist_testset.data.to(dtype=torch.float32).reshape(-1,28*28).to(device))

    train_accuracy = get_accuracy(train_ep_pred.cpu(), mnist_trainset.targets.to(dtype=torch.long))
    test_accuracy = get_accuracy(test_ep_pred.cpu(), mnist_testset.targets.to(dtype=torch.long))
    print("test_accuracy", test_accuracy, "train_accuracy", train_accuracy)

    #bias_1_matrix = bias_1_matrix.reshpae(20)
    #bias_2_matrix = bias_2_matrix.reshpae(19)
    combo = torch.cat((weight_1_matrix, weight_2_matrix,
                      bias_1_matrix, bias_2_matrix))
    print("combo", combo.shape)

    model_params.append(combo)

    del weight_1_matrix
    del weight_2_matrix
    del bias_1_matrix
    del bias_2_matrix
    del classifier_load
    # del combo
print("modelparams", len(model_params))


class WeightsDataset(torch.utils.data.Dataset):
    def __init__(self, weights, transform=None, target_transform=None):
        self.weight_data = weights
        self.img_labels = []  # pd.read_csv(annotations_file)
        indices = []
        for i in range(len(weights)):
            self.img_labels.append(0)
            indices.append(i)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.weight_data[idx].to(device)
        label = (idx//10) % 10
        return image, torch.tensor(label).to(device)


weights_data_set = WeightsDataset(model_params)

weights_data_loader = torch.utils.data.DataLoader(
    weights_data_set, batch_size=20, shuffle=True
)


def train():
    sampler = ImageSampler()
    numsampler = NumberSampler()

    vae = VAE()
    trainer = pl.Trainer(gpus=1, logger=logger,
                         max_epochs=300, callbacks=[sampler, numsampler],auto_lr_find=True)
    trainer.fit(vae, weights_data_loader)


if __name__ == '__main__':
    train()
