import torch  # torch provides basic functions, to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network. This module torch.nn also has various
                       # layers that you can use to build your neural network.
                     # For example, nn.Linear
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.

import lightning as L  # this makes neural networks easier to train

# Create a neural network class by creating a class that inherits from LightningModule


class BasicLightning(L.LightningModule):
    def __init__(self):  # __init__() is the class constructor function,
        # and used it to initialize the weights and biases.
        super().__init__()  # initialize an instance of the parent class, L.LightningModule.

        # Now create the weights and biases that we need for our neural network.
        # Each weight or bias is an nn.Parameter, which gives us the option to optimize the parameter by setting
        # requires_grad, which is short for  "requires gradient ", to True. Since we don't need to optimize any of these
        # parameters now, we set requires_grad=False.

        # NOTE: Because our neural network is already fit to the data, we will input specific values
        # for each weight and bias. In contrast, if we had not already fit the neural network to the data
        # we might start with a random initialization of the weights and biases.
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input):  # forward() takes an input value and runs it though the neural network
        # illustrated at the top of this notebook.
        # the next three lines implement the top of the neural network (using the top node in the hidden layer)
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        # the next three lines implement the bottom of the neural network (using the bottom node in the hidden layer).
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11
        # here, we combine both the top and bottom nodes from the hidden layer with the final bias.
        input_to_final_relu = (scaled_top_relu_output
                               + scaled_bottom_relu_output
                               + self.final_bias)
        output = F.relu(input_to_final_relu)
        return output  # output is the predicted effectiveness for a drug dose.



