import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score
from torcheval.metrics import BinaryAccuracy


def train(epochs, train_set, test_set, model, optimizer, criterion, batch_size, test_function, scheduler=None, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps" for macOS

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    training_loss_list = []
    training_accuracy_list = []
    training_f1_score_list = []

    test_accuracy_list = []
    test_f1_score_list = []

    print(f"Starting training on {device}, batch_size={batch_size}, lr={optimizer.param_groups[0]['lr']}, scheduler={scheduler}")
    # print(model)

    # Training
    for epoch in range(epochs):
        running_loss = 0.0

        # print(f"Epoch {epoch + 1}")
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        if scheduler:
            scheduler.step()
            # print(scheduler.get_last_lr())

        training_loss_list.append(running_loss / len(train_loader))

        if True: #(epoch % 10 == 0) and verbose:
            print(f'Epoch {epoch + 1}, loss = {running_loss / len(train_loader)}')

        # Testing
        if epoch == epochs - 1 or epoch % 500 == 0 or True: # Set by preference

            with torch.no_grad():
                verbose_test = ( (epoch % 500 == 0) or (epoch == epochs - 1) ) and verbose

                model.eval()
                test_accuracy, test_f1 = test_function(model, test_loader, device, verbose=verbose_test, save=False)
                test_accuracy_list.append(test_accuracy)
                test_f1_score_list.append(test_f1)

                train_accuracy, train_f1 = test_function(model, train_loader, device, verbose=verbose_test)
                training_accuracy_list.append(train_accuracy)
                training_f1_score_list.append(train_f1)
                model.train()

                if verbose_test:
                    print(f'Train Accuracy: at epoch {epoch}: {train_accuracy}')
                    print(f'Test Accuracy: at epoch {epoch}: {test_accuracy}')

        if test_accuracy == 1  and train_accuracy == 1:
            print(f"100% accuracy at: {epoch} epoch")

    train_eval_metrics = {
        "ACC": training_accuracy_list,
        "F1": training_f1_score_list
    }

    test_eval_metrics = {
        "ACC": test_accuracy_list,
        "F1": test_f1_score_list
    }

    return model, training_loss_list, train_eval_metrics, test_eval_metrics



def binary_accuracy_function(test_model, data_loader, device, verbose=False, save=False):
    # TODO make this automatic so you do not forget to change it again ...
    threshold = 0.5

    metric = BinaryAccuracy(threshold=threshold)

    labels_all = []
    outputs_all = []
    # outputs_raw_all = []

    for data in data_loader:
        inputs, labels = data

        if threshold == 0:
            labels[labels == -1] = 0

        outputs = test_model(inputs.to(device))
        # if verbose:
        #     print(inputs, outputs)


        if outputs.ndim != 1:
            outputs = outputs.squeeze(1)

        if labels.ndim != 1:
            labels = labels.squeeze(1)

        metric.update(outputs, labels)
        # outputs_raw_all.append(outputs.cpu())


        labels = labels.cpu()
        outputs = torch.where(outputs.cpu() < threshold, 0, 1)

        labels_all += labels.tolist()
        outputs_all += outputs.tolist()

    if verbose:
        # print(f"Precision: {precision_score(labels_all, outputs_all)}")
        # print(f"Recall: {recall_score(labels_all, outputs_all)}")
        # print("Outputs raw:", outputs_raw_all)
        # print("Outputs:", outputs_all)
        # print("Labels:", labels_all)
        print(confusion_matrix(labels_all, outputs_all))

    return metric.compute(), f1_score(labels_all, outputs_all)

def accuracy_function(test_model, data_loader, device, **other):
    total = 0
    correct = 0
    for data in data_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = test_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total, 0


if __name__ == '__main__':
    ...

