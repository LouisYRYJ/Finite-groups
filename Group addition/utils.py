from einops import rearrange
import random
import torch as t


def loss_fn(logits, labels):
    """
    Compute cross entropy loss.

    Args:
        logits (Tensor): (batch, group.order) tensor of logits
        labels (Tensor): (batch) tensor of labels

    Returns:
        float: cross entropy loss
    """
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def get_accuracy(logits, labels):
    """
    Compute accuracy of model.

    Args:
        logits (torch.tensor): (batch, group.order) tensor of logits
        labels (torch.tensor): (batch) tensor of labels

    Returns:
        float: accuracy
    """
    return ((logits.argmax(-1) == labels).sum() / len(labels)).item()


def test_loss(model, params, Group_Dataset):
    """Create all possible pairs (x,y) and return loss and accuracy for G_1 and G_2"""
    test_labels_x = t.tensor([num for num in range(params.N) for _ in range(params.N)])
    test_labels_y = t.tensor([num % params.N for num in range(params.N * params.N)])

    test_labels = t.stack((test_labels_x, test_labels_y), dim=1)

    logits = model(test_labels)
    labels_group_1 = rearrange(Group_Dataset.group1, "a b-> (a b)")
    labels_group_2 = rearrange(Group_Dataset.group2, "a b-> (a b)")

    loss_group_1 = loss_fn(logits, labels_group_1)
    loss_group_2 = loss_fn(logits, labels_group_2)

    accuracy_group_1 = get_accuracy(logits, labels_group_1)
    accuracy_group_2 = get_accuracy(logits, labels_group_2)

    return (loss_group_1, loss_group_2), (accuracy_group_1, accuracy_group_2)


def random_indices(full_dataset, params):
    num_indices = int(len(full_dataset) * params.train_frac)
    picked_indices = random.sample(list(range(len(full_dataset))), num_indices)
    return picked_indices
