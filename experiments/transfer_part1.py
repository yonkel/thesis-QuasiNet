import time

import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from matplotlib import pyplot as plt
import numpy as np

from utils.Nets import *
from utils.trainer_performance import train, accuracy_function

transform = transforms.Compose([
    transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == "__main__":
    name = "cifar_cifar10"
    # Feature extraction
    # 1st training - Full net on Cifar10 without QuasiNet

    batch_size = 16
    epochs = 30
    learning_rate = 0.005
    # momentum = 0.8

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    model = CifarNet()

    print(sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(model)
    start = time.time()
    model, train_loss, train_metrics, test_metrics = train(epochs, train_dataset, test_dataset, model, optimizer, criterion, batch_size, accuracy_function)
    end = time.time()

    print("Training time: ", end - start)


    torch.save(model.state_dict(), 'model_weights/covn_cifar10.pth')

    print(f"Train ACC: {train_metrics['ACC'][-1]}, F1: {train_metrics['F1'][-1]}, loss: {train_loss[-1]}")
    print(f"Test ACC: {test_metrics['ACC'][-1]}, F1: {test_metrics['F1'][-1]}")

    # Plot Minimal Train loss
    plt.plot(train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/spirals/{name}_train_loss.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plot ACC
    # Train
    plt.plot(train_metrics["ACC"], color="#4169e1", label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("ACC")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/parity/{name}_train_acc.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Test
    plt.plot(test_metrics["ACC"], color="firebrick", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("ACC")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(f"../results/parity/{name}_test_acc.pdf", format="pdf", bbox_inches="tight")
    plt.show()