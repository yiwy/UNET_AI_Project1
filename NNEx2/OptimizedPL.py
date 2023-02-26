import torch
import NNEx2.BasicLightningTrain
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data case working
import matplotlib.pyplot as plt  # matplotlib allows us to draw graphs.
import seaborn as sns
import lightning as L  # this makes neural networks easier to train

def PLopt():
    input_doses = torch.linspace(start=0, end=1, steps=11)  # Create a tensor with 11 values
    inputs = torch.tensor([0., 0.5, 1.] * 100)
    labels = torch.tensor([0., 1., 0.] * 100)

    # If we want to use Lightning for training, then we have to pass the Trainer the data wrapped in
    # something called a DataLoader. DataLoaders provide a handful of nice features including...
    #   1) They can access the data in mini-batches instead of all at once. In other words,
    #         The DataLoader doesn't need us to load all of the data into memory first. Instead
    #      it just loads what it needs in an efficient way. This is crucial for large datasets
    #   2) They can reshuffle the data every epoch to reduce model overfitting
    #   3) We can easily just use a fraction of the data if we want do a quick train
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    model = NNEx2.BasicLightningTrain.BasicLightningTrain()  # First, make model from the class
    # Now create a Trainer - we can use the trainer to...
    #  1) Find the optimal learning rate
    #  2) Train (optimize) the weights and biases in the model
    # By default, the trainer will run on your system's CPU
    # trainer = L.Trainer(max_epochs=34)
    # However, if we wanted to automatically take advantage of any available GPUs,
    # we would set accelerator= "auto " to automatically use available GPUs
    # and we would set devices=\"auto\" to automatically select as many GPUs as we have.
    trainer = L.Trainer(max_epochs=34, accelerator="auto", devices="auto")

    # Now let's find the optimal learning rate
    lr_find_results = trainer.tuner.lr_find(model,
                                            train_dataloaders=dataloader,  # the training data
                                            min_lr=0.001,  # minimum learning rate
                                            max_lr=1.0,    # maximum learning rate
                                            early_stop_threshold=None)  # setting this to "None" tests all 100 candidate rates
    new_lr = lr_find_results.suggestion()  # suggestion() returns the best guess for the optimal learning rate
    # now print out the learning rate
    print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")
    # now set the model's learning rate to the new value
    model.learning_rate = new_lr
    # NOTE: we can also plot the loss for each learning rate tested.
    # When you have a lot of data, this graph can be useful
    # (see https://pytorch-lightning.readthedocs.io/en/1.4.5/advanced/lr_finder.html to learn how to interpret)
    # but when you only have 3 data points, like our example, this plot is pretty hard to interpret so I did
    # not cover it in the video.
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    # Now that we have an improved training rate, let's train the model to optimize `final_bias`
    # Now that we have an improved learning rate, we can train the model (optimize final_bias)
    trainer.fit(model, train_dataloaders=dataloader)
    print(model.final_bias.data)
    # So, if everything worked correctly, the optimizer should have converged on `final_bias = 16.0070`. **BAM!**
    # Lastly, let's graph the output from the optimized neural network and see if it's the same as what we started with.
    # If so, then the optimization worked.
    # run the different doses through the neural network
    output_values = model(input_doses)
    # set the style for seaborn so that the graph looks cool.
    sns.set(style="whitegrid")
    # create the graph (you might not see it at this point, but you will after we save it as a PDF).
    sns.lineplot(x=input_doses,
                 y=output_values.detach(), # NOTE: we call detach() because final_bias has a gradient
                 color='green',
                 linewidth=2.5)
    # now label the y- and x-axes.
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.suptitle('Optimised Neural Network')
    plt.savefig('BasicLightningTrain_optimized.pdf')
    plt.show()
    # lastly, save the graph as a PDF.
    # plt.savefig('BasicLightningTrain_optimized.pdf')
