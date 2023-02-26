import torch  # torch provides basic functions, to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network. This module torch.nn also has various
                       # layers that you can use to build your neural network.
                     # For example, nn.Linear
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.

import lightning as L  # this makes neural networks easier to train
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data case working
                                                        # with large datasets.import

# input_doses = torch.linspace(start=0, end=1, steps=11)  # Create a tensor with 11 values


class BasicLightningTrain(L.LightningModule):
    def __init__(
            self):  # __init__() is the class constructor function, and we use it to initialize the weights and biases. n",
        # NOTE: The code for __init__ () is the same as before except we now have a learning rate parameter (for
        #      gradient descent) and we modified final_bias in two ways:
        #           1) we set the value of the tensor to 0, and
        #          2) we set  "requires_grad=True
        super().__init__()  # initialize an instance of the parent class, LightningModule
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        # We want to modify final_bias to demonstrate how to optimize it with backpropagation.
        # NOTE: The optimal value for final_bias is -16...
        # self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
        # ...so we set it to 0 and tell Pytorch that it now needs to calculate the gradient for this parameter
        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.learning_rate = 0.01  # this is for gradient descent.
        # NOTE: we will improve this value later, so, technically
        # this is just a placeholder until then. In other words, we could put any value here
        # because later we will replace it with the improved value

    def forward(self, input):
        # forward() is the exact same as before
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11
        input_to_final_relu = (scaled_top_relu_output
                               + scaled_bottom_relu_output
                               + self.final_bias)
        output = F.relu(input_to_final_relu)
        return output  # output is the predicted effectiveness for a drug dose

    def configure_optimizers(self):  # this configures the optimizer we want to use for backpropagation
        return SGD(self.parameters(), lr=self.learning_rate)  # NOTE: We set the learning rate (lr) to our new variable
        # self.learning_rate

    def training_step(self, batch, batch_idx):  # take a step during gradient descent.
        # NOTE: When training_step() is called it calculates the loss with the code below...
        input_i, label_i = batch  # collect input
        output_i = self.forward(input_i)  # run input through the neural network
        loss = (output_i - label_i) ** 2  # loss = squared residual
        # ...before calling (internally and behind the scenes)...
        # optimizer.zero_grad() # to clear gradients
        # loss.backward() # to do the backpropagation
        # optimizer.step() # to update the parameters
        return loss


# model = BasicLightningTrain()
# # now run the different doses through the neural network
# output_values = model(input_doses)
# # Now draw a graph that shows the effectiveness for each dose.