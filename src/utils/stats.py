import torch


@torch.no_grad()
def comp_stats_classification(
    model, criterion, data_loader, max_iter: int = 10
) -> tuple[float, float]:
    """Compute loss and accuracy for classification task."""

    device = next(model.parameters()).device

    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0
    running_counter = 0

    iterations = 0

    for x_data, y_data in data_loader:
        inputs, labels = x_data.to(device), y_data.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        pred = (torch.argmax(outputs, dim=1) == labels).float().sum().item()
        running_loss += loss
        running_accuracy += pred
        running_counter += labels.size(0)

        iterations += 1
        if iterations == max_iter:
            break

    loss = running_loss / running_counter
    accuracy = running_accuracy / running_counter

    model.train()

    return loss, accuracy
