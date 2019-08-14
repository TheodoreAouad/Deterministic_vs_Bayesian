import torch
from tqdm import tqdm

from src.utils import set_and_print_random_seed
from src.uncertainty_measures import aggregate_data, get_predictions_from_multiple_tests, get_all_uncertainty_measures


def evaluate(model, evalloader, device, val=False):
    """
    Evaluate the model on the data evalloader with only one forward per sample. It is a classic evaluation.
    Args:
        model (torch.nn.Module child): the model we evaluate
        evalloader (torch.utils.data.dataloader.DataLoader): dataloader of the test set
        device (torch.device || str): which device to compute on (either on GPU or CPU). Either torch.device type or
                                      specific string 'cpu' or 'gpu'.
        val (Bool): if true, we are in evaliation mode and do not print the progress bar

    Returns:
        float: accuracy
        torch.Tensor: size = (number of test samples, number of classes): output of softmax of all the inputs

    """
    accuracy, all_outputs = eval_bayesian(model, evalloader, number_of_tests=1, device=device, val=val)
    return accuracy, all_outputs


def eval_bayesian(model, evalloader, number_of_tests, device='cpu', val=False):
    """
    Evaluate the model on the data evalloader with only multiple forwards per sample. Only useful if the forward can
    change for the same input. Else, use evaluate.
    Args:
        model (torch.nn.Module child): the model we evaluate
        evalloader (torch.utils.data.dataloader.DataLoader): dataloader of the test set
        number_of_tests (int): the number of times we do a forward for each input
        device (torch.device || str): which device to compute on (either on GPU or CPU). Either torch.device type or
                                      specific string 'cpu' or 'gpu'.
        val (Bool): if if true, we are in evaliation mode and do not print the progress bar

    Returns:
        float: accuracy
        torch.Tensor: size = (number of test samples, number of classes): output of softmax of all the inputs
    """
    model.eval()
    with torch.no_grad():
        number_of_samples = len(evalloader.dataset)
        batch_size = evalloader.batch_size
        number_of_classes = model.number_of_classes
        all_correct_labels = torch.zeros(1, requires_grad=False)
        all_outputs = torch.Tensor().to(device)

        if val:
            iterator = enumerate(evalloader)
        else:
            iterator = tqdm(enumerate(evalloader))
        for batch_idx, data in iterator:
            inputs, labels = data[0].to(device), data[1].to(device)
            batch_outputs = torch.zeros(
                (number_of_tests, inputs.size(0), number_of_classes)
            ).to(device)
            for test_idx in range(number_of_tests):
                output = model(inputs)
                batch_outputs[test_idx] = output
            predicted_labels = get_predictions_from_multiple_tests(batch_outputs)

            all_correct_labels += torch.sum(predicted_labels.int() - labels.int() == 0)
            all_outputs = torch.cat((all_outputs, batch_outputs), 1)

        accuracy = (all_correct_labels / number_of_samples).item()

        return accuracy, all_outputs


def eval_random(model, batch_size, img_channels, img_dim, number_of_tests, random_seed=None, show_progress=False,
                device='cpu'):
    """

    Args:
        model (torch.nn.Module child): the model we evaluate
        batch_size (int): The size of the random sample
        img_channels (int): dimension of the random sample
        img_dim (int): dimension of the random sample
        number_of_tests (int): the number of times we do a forward for each input
        random_seed (int): the seed of the random generation, for reproducibility
        show_progress (Bool): whether we want a progress bar or not
        device (torch.device || str): which device to compute on (either on GPU or CPU). Either torch.device type or
                                      specific string 'cpu' or 'gpu'.

    Returns:
        torch.Tensor: size (batch_size): the variation-ratio uncertainty
        torch.Tensor: size (batch_size): the predictive entropy uncertainty
        torch.Tensor: size (batch_size): the mutual information uncertainty
        int: the seed of the random generation, for reproducibility

    """
    number_of_classes = model.number_of_classes
    seed = set_and_print_random_seed(random_seed)
    random_noise = torch.randn(batch_size, img_channels, img_dim, img_dim).to(device)
    output_random = torch.Tensor(number_of_tests, batch_size, number_of_classes)
    if show_progress:
        iterator = tqdm(range(number_of_tests))
    else:
        iterator = range(number_of_tests)
    for test_idx in iterator:
        output_random[test_idx] = model(random_noise).detach()
    return output_random, seed
