import torch  # torch provides basic functions, to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
# This module torch.nn also has various layers that you can use to build your neural network. For example, nn.Linear
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.


class BasicNN_train(nn.Module):

    def __init__(
            self):  # __init__ is the class constructor function, used to initialize the weights and biases.

        super().__init__()  # initialize an instance of the parent class, nn.Module.

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        # Modify final_bias to demonstrate how to optimize it with backpropagation.
        # The optimal value for final_bias is -16... from previous example
        #  self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
        # ...variable set to 0 and tell Pytorch that it now needs to calculate the gradient for this parameter.
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
