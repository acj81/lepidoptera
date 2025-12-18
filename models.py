import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# define our model:

class CNN(nn.Module):
    def __init__(self):
        # parent init
        super().__init__()
        # transform we use - other ways of doing it but here bc readable:
        self.flatten = nn.Flatten()
        # define our layers and architecture - sequentially-layered in this case:
        self.layers = nn.Sequential(
            #  1 convolutional "block"
            #nn.Conv2d(224*224*3, 224*224, 3, stride=1),
            #nn.ReLU(),
            #nn.MaxPool2d(100),
            # 2 fully connected layers @ end:
            nn.Linear(224*224*3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
		 
    def forward(self, x):
        ''' given input x, propagate forward and get output - NOT MEANT TO CALL DIRECTLY'''
        # transform input by flattening:
        x = self.flatten(x)
        # make sure its converted to correct dtype:
        x = x.to(torch.float32)
        # propagate forward from input:
        out = self.layers(x)
        return out

    def train_on(self, dataloader, loss_fn, optimizer, epochs=1):
        ''' given input dataset in dataloader, train on that dataset and update parameters, returning nothing'''
        # get size of dataset:
        size = len(dataloader)
        # set in training mode - good practice:
        self.train()
        # handle epochs if specified:
        for i in range(epochs):
            # enumerate for each example - technically don't need but cool:
            for batch_num, (inp, exp_out) in enumerate(dataloader):
                # propagate forward, then calculate loss:
                out = self(inp)
                example_loss = loss_fn(out, exp_out)

                # BACKPROPAGATION - actual training here:
                example_loss.backward() 
                optimizer.step()
                optimizer.zero_grad()
			
                # showing items:
                print(f"training on epoch / batch: {i + 1} / {batch_num}")

    def test_on(self, dataloader, loss_fn):
        ''' given dataset, iterate through and test it: '''
	
        # prevent any gradients from being calculated - don't need to, just testing;
        with torch.no_grad():
            # need num. examples not batches:
            num_examples = len(dataloader.dataset)
            # store num. examples correctly classified, total loss:
            accuracy = 0
            avg_loss = 0
            # iterate through test examples:
            for i, (inp, exp_out) in enumerate(dataloader):
                # testing section:
                # forward propagate, get loss for this example:
                out = self(inp)
                example_loss = loss_fn(out, exp_out).item()
                # get whether or not model was right (accuracy), how far off it was (loss):
                # whether model right for this example - convert to 1 or 0 with extra methods:
                accuracy += (out.argmax(1) == exp_out).type(torch.float).sum().item()
                # how far off it was;
                avg_loss += example_loss

                # print current batch num:
                print(f"testing batch: {i}")
				
            # get averages using size:
            accuracy /= num_examples
            avg_loss /= num_examples
            #
            return (accuracy, avg_loss)
