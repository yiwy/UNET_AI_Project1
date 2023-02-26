import torch  # torch provides basic functions, to creating tensors.
from torch.optim import SGD # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  # matplotlib allows us to draw graphs.
import seaborn as sns  # seaborn makes it easier to draw nice-looking graphs.
import NNEx1.BasicNN_Train
# Create the training data for the neural network.


def opt():
    inputs = torch.tensor([0., 0.5, 1.])
    labels = torch.tensor([0., 1., 0.])
    input_doses = torch.linspace(start=0, end=1, steps=11)  # Create a tensor with 11 values
    # ...Using the training data to train (or optimize) final_bias
    # Create the neural network to train.
    model1 = NNEx1.BasicNN_Train.BasicNN_train()

    optimizer = SGD(model1.parameters(), lr=0.1)  # here we're creating an optimizer to train the neural network.
                                             # NOTE: There are a bunch of different ways to optimize a neural network.
                                             # In this example, we'll use Stochastic Gradient Descent (SGD). However,
                                             # another popular algortihm is Adam (which will be covered in a StatQuest).

    print("Final bias, before optimization: " + str(model1.final_bias.data) + "\n")

    # this is the optimization loop. Each time the optimizer sees all of the training data is called an "epoch".
    for epoch in range(100):

        # we create and initialize total_loss for each epoch so that we can evaluate how well model fits the
        # training data. At first, when the model doesn't fit the training data very well, total_loss
        # will be large. However, as gradient descent improves the fit, total_loss will get smaller and smaller.
        # If total_loss gets really small, we can decide that the model fits the data well enough and stop
        # optimizing the fit. Otherwise, we can just keep optimizing until we reach the maximum number of epochs.
        total_loss = 0

        # this internal loop is where the optimizer sees all of the training data and where we
        # calculate the total_loss for all of the training data.
        for iteration in range(len(inputs)):
            input_i = inputs[iteration]  # extract a single input value (a single dose)...
            label_i = labels[iteration]  # ...and its corresponding label (the effectiveness for the dose).

            output_i = model1(input_i)  # calculate the neural network output for the input (the single dose).

            loss = (output_i - label_i) ** 2  # calculate the loss for the single value.
            # NOTE: Because output_i = model(input_i), "loss" has a connection to "model"
            # and the derivative (calculated in the next step) is kept and accumulated
            # in "model".

            loss.backward()  # backward() calculates the derivative for that single value and adds it to the previous one.

            total_loss += float(loss)  # accumulate the total loss for this epoch.

        if (total_loss < 0.0001):
            print("Num steps: " + str(epoch))
            break

        optimizer.step()  # take a step toward the optimal value.
        optimizer.zero_grad()  # This zeroes out the gradient stored in "model".
    # Remember, by default, gradients are added to the previous step (the gradients are accumulated),
    # and we took advantage of this process to calculate the derivative one data point at a time.
    # NOTE: "optimizer" has access to "model" because of how it was created with the call
    # (made earlier): optimizer = SGD(model.parameters(), lr=0.1).
    # ALSO NOTE: Alternatively, we can zero out the gradient with model.zero_grad().

        print("Step: " + str(epoch) + " Final Bias: " + str(model1.final_bias.data) + "\n")
    # now go back to the start of the loop and go through another epoch.

    print("Total loss: " + str(total_loss))
    print("Final bias, after optimization: " + str(model1.final_bias.data))

    # So, if everything worked correctly, the optimizer should have converged on
    # `final_bias = 16.0019` after **34** steps, or epochs.
    #
    # Lastly, let's graph the output from the optimized neural network and see if it's the same as what we started with.
    # If so, then the optimization worked.

    # run the different doses through the neural network
    output_values1 = model1(input_doses)

    # set the style for seaborn so that the graph looks cool.
    sns.set(style="whitegrid")

    # create the graph (you might not see it at this point, but you will after we save it as a PDF).
    sns.lineplot(x=input_doses,
                 y=output_values1.detach(), # NOTE: we call detach() because final_bias has a gradient
                 color='green',
                 linewidth=2.5)

    # now label the y- and x-axes.
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.suptitle('Optimized Neural Network')
    plt.savefig('BascNN_optimized.pdf')
    plt.show()

    # lastly, save the graph as a PDF.


    # path = "/home/yiwy/Documents/Python/Unet-AI/pythonProject/BascNN_optimized.pdf"
    # webbrowser.open_new(path)

    print("Final bias, after optimization: " + str(model1.final_bias.data))
