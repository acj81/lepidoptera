# import PyTorch and ML libraries:
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import ToTensor, Lambda

# import project classes:
import models
import datasets




# training section

if __name__ == "__main__":
    # create models:
    model = models.CNN().to("cpu")
    # datasets:
    train_data = datasets.ButterflyDataset(
        "butterfly_species_data/data/", 
        "butterfly_species_data/data/butterflies and moths.csv", 
        "train"
    )
    test_data = datasets.ButterflyDataset(
        "butterfly_species_data/data/", 
        "butterfly_species_data/data/butterflies and moths.csv", 
        "train"
    )
    # dataloaders:
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True) 
    # hyperparameters:
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # test at start:
    print(model.test_on(test_loader, loss_fn)
    # train on dataset:
    model.train_on(train_loader, loss_fn, optimizer, epochs)
    # test after training:
    print(model.test_on(test_loader, loss_fn))
