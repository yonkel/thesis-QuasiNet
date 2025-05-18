import time

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LinearLR, ExponentialLR

from utils.Nets import *
from utils.data import get_spirals_dataset, get_parity_dataset
from utils.help_functions import human_readable_time
from utils.trainer_performance import train, binary_accuracy_function

if __name__ == "__main__":
    # Setup

    name = 'parity4QT'

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    torch.set_printoptions(precision=14)

    for _ in range(10):
        model = Parity4Net(7)

        epochs = 2062
        batch_size = 1

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.8)

        train_set = get_parity_dataset(degree=4, remap=True)

        # Training
        start = time.time()
        model, train_loss, train_metrics, test_metrics = train(epochs, train_set, train_set, model, optimizer,
                                                               criterion,
                                                               batch_size, binary_accuracy_function,
                                                               verbose=True)
        end = time.time()
        print("Training time: ", human_readable_time(end - start))

        train_losses.append(train_loss)
        train_accuracies.append(train_metrics["ACC"])
        test_accuracies.append(test_metrics["ACC"])

        print(f"Train ACC: {train_metrics['ACC'][-1]}, loss: {train_loss[-1]}")
        print(f"Test ACC: {test_metrics['ACC'][-1]}")
        print()

    print(f"Train Loss: {[loss[-1] for loss in train_losses]}\n"
          f"avg = {np.mean([loss[-1] for loss in train_losses])}\n"
          f"min = {np.min([loss[-1] for loss in train_losses])}")

    print(
        f"Train ACC: {[loss[-1] for loss in train_accuracies]}, avg = {np.mean([loss[-1] for loss in train_accuracies])}")
    print(
        f"Test ACC: {[loss[-1] for loss in test_accuracies]}, avg = {np.mean([loss[-1] for loss in test_accuracies])}")

    loss_runs = np.array(train_losses)

    minimal_idx = 0
    for i in range(1, len(train_losses)):
        if train_losses[i][-1] < train_losses[minimal_idx][-1]:
            minimal_idx = i

    smallest_loss = loss_runs[minimal_idx]
    smallest_acc = train_accuracies[minimal_idx]
    epochs = np.arange(len(smallest_loss))

    # Plot Minimal Train loss
    plt.plot(smallest_loss)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/parity/{name}_train_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plot ACC
    # Train
    acc_train = np.array(train_accuracies)
    mean_acc_train = acc_train.mean(axis=0)
    std_acc_train = acc_train.std(axis=0)

    plt.plot(smallest_acc, color="firebrick", label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("ACC")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/parity/{name}_train_acc.pdf", format="pdf", bbox_inches="tight")
    plt.show()

"""
tensor([[ 1., -1., -1.,  1.]]) tensor([[-0.99098616838455]])
tensor([[ 1., -1., -1., -1.]]) tensor([[0.99299627542496]])
tensor([[-1.,  1., -1.,  1.]]) tensor([[-0.99095606803894]])
tensor([[ 1.,  1., -1.,  1.]]) tensor([[0.98873573541641]])
tensor([[1., 1., 1., 1.]]) tensor([[-0.99079626798630]])
tensor([[-1.,  1.,  1.,  1.]]) tensor([[0.99308937788010]])
tensor([[-1., -1.,  1.,  1.]]) tensor([[-0.99505656957626]])
tensor([[-1.,  1., -1., -1.]]) tensor([[0.99286180734634]])
tensor([[-1., -1., -1., -1.]]) tensor([[-0.99487072229385]])
tensor([[-1., -1., -1.,  1.]]) tensor([[0.99308735132217]])
tensor([[-1.,  1.,  1., -1.]]) tensor([[-0.99538540840149]])
tensor([[-1., -1.,  1., -1.]]) tensor([[0.99721860885620]])
tensor([[ 1., -1.,  1., -1.]]) tensor([[-0.99524861574173]])
tensor([[ 1., -1.,  1.,  1.]]) tensor([[0.99289715290070]])
tensor([[ 1.,  1., -1., -1.]]) tensor([[-0.99086052179337]])
tensor([[ 1.,  1.,  1., -1.]]) tensor([[0.99327194690704]])


tq_outputs = [-0.99098616838455, 0.99299627542496, -0.99095606803894, 0.98873573541641, -0.99079626798630,
              0.99308937788010, -0.99505656957626, 0.99286180734634, -0.99487072229385, 0.99308735132217,
              -0.99538540840149, 0.99721860885620, -0.99524861574173, 0.99289715290070, -0.99086052179337,
              0.99327194690704]
labels_p4 = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]

plt.scatter(x = [range(16)], y=labels_p4, c="white", edgecolors="red", marker="o", label="Target output")
plt.scatter(x = [range(16)], y=tq_outputs, c="blue", marker="*", label="TQ network", alpha=0.8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.box(False)
plt.tight_layout()
plt.legend()
plt.savefig(f"results/parity/parity4_benchmark.pdf", format="pdf", bbox_inches="tight")
"""