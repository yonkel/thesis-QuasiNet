import time

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, MultiStepLR

from utils.Nets import *
from utils.Nets import Spirals3MLPNet
from utils.data import get_spirals_dataset
from utils.help_functions import human_readable_time
from utils.trainer_performance import train, binary_accuracy_function

if __name__ == "__main__":
    # Setup
    name = 'spiralsMLP3'

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for _ in range(10):
        model = Spirals3MLPNet()

        epochs = 2500
        batch_size = 4

        criterion = nn.MSELoss()

        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.4)
        # scheduler = MultiStepLR(optimizer, [1500], 0.5)

        train_set = get_spirals_dataset(500, num_turns=3, negative_label=0)

        seed = 42
        generator = torch.Generator().manual_seed(seed)
        train_set, test_set = torch.utils.data.random_split(train_set, [400, 100], generator)

        # Training
        start = time.time()
        model, train_loss, train_metrics, test_metrics = train(epochs, train_set, test_set, model, optimizer, criterion,
                                                               batch_size, binary_accuracy_function,  # scheduler,
                                                               verbose=False)
        end = time.time()
        print("Training time: ", human_readable_time(end - start))

        train_losses.append(train_loss)
        train_accuracies.append(train_metrics["ACC"])
        test_accuracies.append(test_metrics["ACC"])

        print(f"Train ACC: {train_metrics['ACC'][-1]}, F1: {train_metrics['F1'][-1]}, loss: {train_loss[-1]}")
        print(f"Test ACC: {test_metrics['ACC'][-1]}, F1: {test_metrics['F1'][-1]}")
        print()

    print(f"Train Loss: {[loss[-1] for loss in train_losses]}, avg = {np.mean([loss[-1] for loss in train_losses])}")
    print(
        f"Train ACC: {[loss[-1] for loss in train_accuracies]}, avg = {np.mean([loss[-1] for loss in train_accuracies])}")
    print(
        f"Test ACC: {[loss[-1] for loss in test_accuracies]}, avg = {np.mean([loss[-1] for loss in test_accuracies])}")

    # Plot Train loss
    loss_runs = np.array(train_losses)
    mean_loss = loss_runs.mean(axis=0)
    std_loss = loss_runs.std(axis=0)
    epochs = np.arange(len(mean_loss))

    plt.plot(mean_loss)
    plt.fill_between(epochs,
                     mean_loss - std_loss,
                     mean_loss + std_loss,
                     alpha=0.2,  # “less visible”
                     linewidth=0,
                     color="#4169e1")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/spirals/{name}_train_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plot ACC
    # Train
    acc_train = np.array(train_accuracies)
    mean_acc_train = acc_train.mean(axis=0)
    std_acc_train = acc_train.std(axis=0)
    epochs = np.arange(len(mean_acc_train))

    plt.plot(mean_acc_train, color="#4169e1", label="Train")
    plt.fill_between(epochs,
                     mean_acc_train - std_acc_train,
                     mean_acc_train + std_acc_train,
                     alpha=0.2,  # “less visible”
                     linewidth=0,
                     color="#4169e1")

    plt.xlabel("Epoch")
    plt.ylabel("ACC")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/spirals/{name}_train_acc.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Test
    acc_test = np.array(test_accuracies)
    mean_acc_test = acc_test.mean(axis=0)
    std_acc_test = acc_test.std(axis=0)
    epochs = np.arange(len(mean_acc_test))

    plt.plot(mean_acc_test, color="firebrick", label="Test")
    plt.fill_between(epochs,
                     mean_acc_test - std_acc_test,
                     mean_acc_test + std_acc_test,
                     alpha=0.2,  # “less visible”
                     linewidth=0,
                     color="firebrick")

    plt.xlabel("Epoch")
    plt.ylabel("ACC")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/spirals/{name}_test_acc.pdf", format="pdf", bbox_inches="tight")
    plt.show()
