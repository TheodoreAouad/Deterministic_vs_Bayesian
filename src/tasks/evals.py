import torch
from tqdm import tqdm

from src.utils import set_and_print_random_seed
from src.uncertainty_measures import aggregate_data, get_predictions_from_multiple_tests


def evaluate(model, testloader, device, val=False):
    accuracy, _, all_dkls = eval_bayesian(model, testloader, number_of_tests=1, device=device, val=val)
    return accuracy, all_dkls


def eval_bayesian(model, evalloader, number_of_tests, device, val=False):

    model.eval()
    number_of_samples = len(evalloader.dataset)
    all_correct_labels = torch.zeros(1, requires_grad=False)
    all_outputs = torch.Tensor().to(device).detach()

    if val:
        iterator = enumerate(evalloader)
    else:
        iterator = tqdm(enumerate(evalloader))
    for batch_idx, data in iterator:
        inputs, labels = [x.to(device).detach() for x in data]
        batch_outputs = torch.Tensor().to(device).detach()
        for test_idx in range(number_of_tests):
            output = model(inputs)
            batch_outputs[test_idx] = output.detach()
        predicted_labels = get_predictions_from_multiple_tests(batch_outputs)

        all_correct_labels += torch.sum(predicted_labels.int() - labels.int() == 0)
        all_outputs = torch.cat((all_outputs, batch_outputs))

    accuracy = (all_correct_labels / number_of_samples).item()

    return accuracy, all_outputs


def eval_random(model, batch_size, img_channels, img_dim, number_of_tests, number_of_classes, random_seed=None, device='cpu'):
    seed = set_and_print_random_seed(random_seed)
    random_noise = torch.randn(batch_size, img_channels, img_dim, img_dim).to(device)
    output_random = torch.Tensor(number_of_tests, batch_size, number_of_classes)
    for test_idx in range(number_of_tests):
        output_random[test_idx] = model(random_noise).detach()
    _, random_uncertainty, random_dkl = aggregate_data(output_random)
    return random_uncertainty, random_dkl, seed
