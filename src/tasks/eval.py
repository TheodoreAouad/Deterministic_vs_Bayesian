import torch
from src.utils import aggregate_data, set_and_print_random_seed


def eval(model, testloader, device):
    return eval_bayesian(model, testloader, number_of_tests=1, device=device)


def eval_bayesian(model, testloader, number_of_tests, device):

    model.eval()
    number_of_samples = len(testloader.dataset)
    all_correct_labels = torch.zeros(1, requires_grad=False)
    all_uncertainties = torch.Tensor().to(device).detach()
    all_dkls = torch.Tensor().to(device).detach()

    for batch_idx, data in enumerate(testloader):
        inputs, labels = [x.to(device).detach() for x in data]
        batch_outputs = torch.Tensor(number_of_tests, inputs.size(0), model.number_of_classes).to(device).detach()
        for test_idx in range(number_of_tests):
            output = model(inputs)
            batch_outputs[test_idx] = output.detach()
        predicted_labels, uncertainty, dkls = aggregate_data(batch_outputs)

        all_uncertainties = torch.cat((all_uncertainties, uncertainty))
        all_dkls = torch.cat((all_dkls, dkls))
        all_correct_labels += torch.sum(predicted_labels.int() - labels.int() == 0)

    accuracy = (all_correct_labels / number_of_samples).item()

    return accuracy, all_uncertainties, all_dkls


def eval_random(model, batch_size, img_channels, img_dim, number_of_tests, number_of_classes, random_seed=None, device='cpu'):
    seed = set_and_print_random_seed(random_seed)
    random_noise = torch.randn(batch_size, img_channels, img_dim, img_dim).to(device)
    output_random = torch.Tensor(number_of_tests, batch_size, number_of_classes)
    for test_idx in range(number_of_tests):
        output_random[test_idx] = model(random_noise).detach()
    _, random_uncertainty, random_dkl = aggregate_data(output_random)
    return random_uncertainty, random_dkl, seed
