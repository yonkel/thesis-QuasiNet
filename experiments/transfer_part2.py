import time

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.Nets import *
from utils.data import get_bin_cifar_dataset
from utils.help_functions import human_readable_time
from utils.trainer_performance import train, binary_accuracy_function

if __name__ == "__main__":

    # Fine-tuning
    # 2nd training - Train full net on BinaryCifar with QuasiNet in FC parts

    name = 'cifarQ'

    # Hyperparameters
    batch_size = 32
    epochs = 15
    learning_rate = 0.005

    transform = transforms.Compose([
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set, test_set = get_bin_cifar_dataset(transform=transform, negative_label=0)

    seed = 42
    generator = torch.Generator().manual_seed(seed)
    train_set, _ = torch.utils.data.random_split(train_set, [10000, 40000], generator)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for _ in range(5):

        model = ConvNet()

        state_dict = torch.load("model_weights/covn_cifar10.pth")
        model.load_state_dict(state_dict)
        # print("loaded model \n", model)

        # Replace last layer with our QuasiNet
        model.fc4 = nn.Linear(128, 8)
        # model.fc4.requires_grad = False
        model.fc5 = Quasi(8, 1)
        print("added Quasi layer \n", model)


        # Freeze the weights if needed
        # model.requires_grad = False
        # model.fc1.requires_grad = True
        # model.fc2.requires_grad = True
        # model.fc3....

        model.train()

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training
        start = time.time()
        model, train_loss, train_metrics, test_metrics = train(epochs, train_set, test_set, model, optimizer, criterion,
                                                               batch_size, binary_accuracy_function,  # scheduler,
                                                               verbose=True)
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

    plt.savefig(f"../results/cifar/{name}_train_loss.pdf", format="pdf", bbox_inches="tight")
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

    plt.savefig(f"../results/cifar/{name}_train_acc.pdf", format="pdf", bbox_inches="tight")
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

    plt.savefig(f"../results/cifar/{name}_test_acc.pdf", format="pdf", bbox_inches="tight")
    plt.show()
