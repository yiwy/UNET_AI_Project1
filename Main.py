# Program done by Yiwy Zambrano as part of UNET AI studies for Electronc engineering Master.
# Based program taken from StatQuest as per instructor suggestion.
# Thew main program calls functions from different packages.
# These Packages are: NNEX1, NNEx2, NNex3.
# NNEX1 contains example of Neural Network with Pytorch.
# NNEX2 contains example of Neural Network with Pytorch and Lightning.
# NNEX3 contains example of Neural Network with Pytorch and Lightning using LSTM (long shore term memory).
# Packages NNEx1 and NNEx2 packages has three examples:
# 1) ideal Neural Network, where weights and bias are pre-determined.
# 2) Train Neural Network for one bias.
# 3) Use data to train Nural Network.
# For package NNEx 3, the weights and bias are determined statistically, then neural network is trained,
# finally the parameters are optimised.

import torch
import matplotlib.pyplot as plt  # matplotlib allows us to draw graphs.
import seaborn as sns  # seaborn makes it easier to draw nice-looking graphs.
import NNEx1.BasicNN
import NNEx1.BasicNN_Train
import NNEx1.OptimizedPmt
import NNEx2.BasicLightning
import NNEx2.BasicLightningTrain
import NNEx2.OptimizedPL
import NNex3.LSTMbyHand
import NNex3.TrainLSTMLightTensor
import NNex3.LSTMnn
import NNex3.LTSMfnct


input_doses = torch.linspace(start=0, end=1, steps=11)  # Create a tensor with 11 values
# Create the training data for the neural network.
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])


def menu():
    print("[1] Option 1: Run a Basic Neural Network using Pytorch. "
          "Predetermine Weights and Bias will be printed along with chart")
    print("[2] Option 2: Run, a Pytorch, Trained Neural Network with Bias bfinal = 0")
    print("[3] Option 3: Optimised the Neural Network using SGD")
    print("[4] Option 4: Run Basic Neural Network Using Pytorch and Lightning."
          "Predetermine Weights and Bias will be printed along with chart ")
    print("[5] Option 5: Run, a Pytorch and Lightning, Trained Neural Network with Bias bfinal = 0")
    print("[6] Option 6: Optimised the Neural Network Using Pytorch and Lightning")
    print("[7] Option 7: Create an LSTM model, prints out the Weights and Bias, "
          "and compare the observed and predicted values for Company A and B")
    print("[8] Option 8: Train the LTSM unit using Lightning and TesorBaord with epoch=2000/3000/5000")
    print("[9] Option 9: Using and optimizing the PyTorch LSTM, nnLSTM()")
    print("[0] Exit the program")
    print("\n\n")


def graph_trend(input_doses, output_values, fig_tittle, flag):
    # Now draw a graph that shows the effectiveness for each dose.
    # First, set the style for seaborn for appearance.
    sns.set(style="whitegrid")
    #
    # # create the graph
    if flag == 1:
        sns.lineplot(x=input_doses,
                 y=output_values,
                 color='green',
                 linewidth=2.5)
    else:
        sns.lineplot(x=input_doses,
                     y=output_values.detach(),  # NOTE: because final_bias has a gradient,   detach() is called
                     # to return a new tensor that only has the value and not the gradient.
                     color='green',
                     linewidth=2.5)
    # label the y- and x-axes.
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.suptitle(fig_tittle)
    plt.savefig(fig_tittle)
    plt.show()



def option1():
    model = NNEx1.BasicNN.BasicNN()
    for name, param in model.named_parameters():
        print(name, param.data)
    print("\n\n")
    output_values = model(input_doses)
    fig_tittle = 'BasicNN.pdf'
    flag = 1
    graph_trend(input_doses, output_values, fig_tittle, flag)


def option2():
    model = NNEx1.BasicNN_Train.BasicNN_train()
    for name, param in model.named_parameters():
        print(name, param.data)
    print("\n\n")
    output_values = model(input_doses)
    fig_tittle = 'Trained Neural Network'
    flag = 2
    graph_trend(input_doses, output_values, fig_tittle, flag)


def option3():
    NNEx1.OptimizedPmt.opt()


def option4():
    model = NNEx2.BasicLightning.BasicLightning()
    for name, param in model.named_parameters():
        print(name, param.data)
    print("\n\n")
    output_values = model(input_doses)
    fig_tittle = 'Basic Neural Network, (Pytorch & Lightning)'
    flag = 1
    graph_trend(input_doses, output_values, fig_tittle, flag)


def option5():
    model = NNEx2.BasicLightningTrain.BasicLightningTrain()
    for name, param in model.named_parameters():
        print(name, param.data)
    print("\n\n")
    output_values = model(input_doses)
    fig_tittle = 'Trained Neural Network with Pytorch and Lightning'
    flag = 2
    graph_trend(input_doses, output_values, fig_tittle, flag)


def option6():
    NNEx2.OptimizedPL.PLopt()


def option7():
    model = NNex3.LSTMbyHand.LSTMbyHand()
    print("Before optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)
    print("\nComparing the observed and predicted values...")
    # NOTE: To make predictions, we pass in the first 4 days worth of stock values \n",
    # in an array for each company. In this case, the only difference between the\n",
    # input values for Company A and B occurs on the first day. Company A has 0 and\n",
    # Company B has 1.\n",
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())


def option8():
    NNex3.TrainLSTMLightTensor.trainlstm()


def option9():
    NNex3.LTSMfnct.lstmfnt()


menu()
option = int(input("Enter your option :  "))
print("\n\n")

while option != 0:
    if option == 1:
        option1()
    elif option == 2:
        option2()
    elif option == 3:
        option3()
    elif option == 4:
        option4()
    elif option == 5:
        option5()
    elif option == 6:
        option6()
    elif option == 7:
        option7()
    elif option == 8:
        option8()
    elif option == 9:
        option9()
    else:
        print("Invalid Option.")
    print()
    menu()
    option = int(input("Enter your option :  \n\n"))

