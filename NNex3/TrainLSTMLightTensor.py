import torch
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data
# case working with large datasets.import
import lightning as L  # this makes neural networks easier to train
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
import NNex3.LSTMbyHand


def trainlstm():
    model = NNex3.LSTMbyHand.LSTMbyHand()
    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)
    trainer = L.Trainer(max_epochs=2000)  # with default learning rate, 0.001
    # (this tiny learning rate makes learning slow)
    trainer.fit(model, train_dataloaders=dataloader)
    path_to_checkpoint = trainer.checkpoint_callback.best_model_path
    print("\nComparing the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    print("The new trainer will start where the last left off, and the check point data is here: " +
          path_to_checkpoint+ "\n")

    # Then create a new Lightning Trainer
    trainer = L.Trainer(max_epochs=3000)  # Before, max_epochs=2000, so, by setting it to 3000, we're adding 1000 more.
    # And then call fit() using the path to the most recent checkpoint files
    # so that we can pick up where we left off.
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)

    # with 1000 epochs more the predictions are
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
    # Data fits pretty well, next step to add more epoch to see if it improves.
    path_to_checkpoint = trainer.checkpoint_callback.best_model_path  ## By default, "best" = "most recent"
    print("The new trainer will start where the last left off, and the check point data is here: " +
          path_to_checkpoint + "\n")
    # Then create a new Lightning Trainer
    trainer = L.Trainer(max_epochs=5000)  # Before, max_epochs=3000, so, by setting it to 5000, we're adding 2000 more.
    # And then call fit() using the path to the most recent checkpoint files
    # so that we can pick up where we left off.
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)
    # in total 2000 more epochs have been added, for a total of 5000

    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    print("After optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)


