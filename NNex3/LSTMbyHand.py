

import torch  # torch provides basic functions, to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network. This module torch.nn
# also has various layers that you can use to build your neural network. For example, nn.Linear
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam  # optim contains many optimizers. Here Adam is used.

import lightning as L  # this makes neural networks easier to train
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data case
# working with large datasets.import

from pytorch_lightning.utilities.seed import seed_everything


class LSTMbyHand(L.LightningModule):

    def __init__(self):
        super().__init__()
        # The first thing we do is set the seed for the random number generator
        # This ensures that when someone creates a model from this class, that model
        # will start off with the exact same random numbers as I started out with when
        # I created this demo.
        seed_everything(seed=42)
        # NOTE: nn.LSTM() uses random values from a uniform distribution to initialize the tensors
        # Here we can do it 2 different ways 1) Normal Distribution and 2) Uniform Distribution
        # We'll start with the Normal Distribution
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        # NOTE: In this case, I'm only using the normal distribution for the Weights.
        # All Biases are initialized to 0.
        # These are the Weights and Biases in the first stage, which determines what percentage
        # of the long-term memory the LSTM unit will remember.
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        # We can also initialize all Weights and Biases using a uniform distribution. This is
        # how nn.LSTM() does it.
        # self.wlr1 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.wlr2 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.blr1 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.wpr1 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.wpr2 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.bpr1 = nn.Parameter(torch.rand(1), requires_grad=True)
        #
        # self.wp1 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.wp2 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.bp1 = nn.Parameter(torch.rand(1), requires_grad=True)
        #
        # self.wo1 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.wo2 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.bo1 = nn.Parameter(torch.rand(1), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):
        # lstm_unit does the math for a single LSTM unit.
        # long term memory is also called \"cell state\
        # short term memory is also called \"hidden state\
        # 1) The first stage determines what percent of the current long-term memory
        #    should be remembered
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                                  (input_value * self.wlr2) +
                                                  self.blr1)
        # 2) The second stage creates a new, potential long-term memory and determines what
        # percentage of that to add to the current long-term memory
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                       (input_value * self.wpr2) +
                                                       self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        # Once we have gone through the first two stages, we can update the long-term memory
        updated_long_memory = ((long_memory * long_remember_percent) +
                           (potential_remember_percent * potential_memory))
        # 3) The third stage creates a new, potential short-term memory and determines what
        # percentage of that should be remembered and used as output.
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                           (input_value * self.wo2) +
                                           self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        # Finally, we return the updated long and short-term memories
        return([updated_long_memory, updated_short_memory])

    def forward(self, input):
        # forward() unrolls the LSTM for the training data by calling lstm_unit() for each day of training data
        # that we have. forward() also keeps track of the long and short-term memories after each day and returns
        # the final short-term memory, which is the 'output' of the LSTM.
        long_memory = 0  # long term memory is also called \"cell state\" and indexed with c0, c1, ..., cN
        short_memory = 0  # long term memory is also called \"cell state\" and indexed with c0, c1, ..., cNn
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]
        # Day 1
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        # Day 2
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        # Day 3
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        # Day 4
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)
        # Now return short_memory, which is the 'output' of the LSTM.
        return short_memory

    def configure_optimizers(self):  # this configures the optimizer we want to use for backpropagation.
        # return Adam(self.parameters(), lr=0.1) # NOTE: Setting the learning rate to 0.1 trains way faster than
        # using the default learning rate, lr=0.001, which requires a lot more
        # training. However, if we use the default value, we get
        # the exact same Weights and Biases that I used in
        # the LSTM Clearly Explained StatQuest video. So we'll use the
        # default value.
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):  # Calculate the loss and log the training progress
        # take a step during gradient descent.
        input_i, label_i = batch  # collect input
        output_i = self.forward(input_i[0])  # run input through the neural network, makes prediction with training data
        loss = (output_i - label_i) ** 2  # loss = squared residual
        ###################
        # Logging the loss and the predicted values so we can evaluate the training
        ###################
        self.log("train_loss", loss)  # log method is part of the LightningModule that is inheriting from
        # the Lightning creates a file in lightning_logs, for this exercise the loss
        # NOTE: Our dataset consists of two sequences of values representing Company A and Company B
        # For Company A, the goal is to predict that the value on Day 5 = 0, and for Company B
        # the goal is to predict that the value on Day 5 = 1. We use label_i, the value we want to
        # predict, to keep track of which company we just made a prediction for and
        # log that output value in a company specific file
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
        return loss

