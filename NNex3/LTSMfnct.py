import torch  # torch provides basic functions, to creating tensors.
import NNex3.LSTMnn
import lightning as L  # this makes neural networks easier to train
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data
# case working with large datasets.import

def lstmfnt():
    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    # Creating a new model, and printing the values.
    model = NNex3.LSTMnn.LightningLSTM()
    # model = LightningLSTM()  # First, make model from the class

    # print out the name and value for each parameter
    print("Before optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)

    print("\nComparing the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    # on this first attempt, the predicted model are bad. Model needs training
    # However, because we've increased the learning rate to 0.1, we only need to train for 300 epochs.

    # NOTE: Because we have set Adam's learning rate to 0.1, we will train much, much faster.
    # Before, with the hand made LSTM and the default learning rate, 0.001, it took about 5000 epochs to fully train
    # the model. Now, with the learning rate set to 0.1, we only need 300 epochs. Now, because we are doing so few epochs,
    # we have to tell the trainer add stuff to the log files every 2 steps (or epoch,
    # since we have to rows of training data)
    # because the default, updating the log files every 50 steps, will result in a terrible looking graphs. So
    trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)

    trainer.fit(model, train_dataloaders=dataloader)

    print("After optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)
    # competing training, the predictions are printed.

    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
    print("\n\n")
    # After just 300 epochs, the LSTM is making great predictions.
    # the prediction for Company A is close to the observed value 0 and
    # the prediction for Company B is close to the observed value 1