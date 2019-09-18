from time import time

from src.tasks.evals import eval_bayesian

def uniform(_, number_of_batchs):
    return 1 / number_of_batchs


def train_bayesian_modular(
        model,
        optimizer,
        loss,
        observables,
        number_of_epochs,
        trainloader,
        valloader=None,
        number_of_tests=10,
        output_dir_tensorboard=None,
        output_dir_results=None,
        device='cpu',
        verbose=False,
):
    """
    Train Bayesian with modular arguments
    Args:
        model (torch.nn.Module child): model we want to train
        optimizer (torch.optim optimizer): how do we update the weights
        loss (src.loggers.losses.base_loss.BaseLoss child): loss object
        observables (src.loggers.observables.Observables): observable object
        number_of_epochs (int): how long do we train our model
        trainloader (torch.utils.data.dataloader.DataLoader): dataloader of train set
        valloader (torch.utils.data.dataloader.DataLoader): dataloader of validation set
        number_of_tests (int): number of tests to perform during validation evaluation
        output_dir_results (str): output directory in which to save the results (NOT IMPLEMENTED)
        output_dir_tensorboard (str): output directory in which to save the tensorboard
        device (torch.device || str): cpu or gpu
        verbose (Bool): print training steps or not
    Returns
        NOT IMPLEMENTED YET
    """
    start_time = time()
    number_of_batch = len(trainloader)
    interval = max(number_of_batch // 10, 1)

    for logger in [loss, observables]:
        logger.set_number_of_epoch(number_of_epochs)
        logger.set_number_of_batch(number_of_batch)
        logger.init_tensorboard_writer(output_dir_tensorboard)
        logger.init_results_writer(output_dir_results)

    model.train()
    for epoch in range(number_of_epochs):
        loss.set_current_epoch(epoch)
        observables.set_current_epoch(epoch)

        for batch_idx, data in enumerate(trainloader):
            loss.set_current_batch_idx(batch_idx)
            observables.set_current_batch_idx(batch_idx)

            inputs, labels = [x.to(device) for x in data]
            optimizer.zero_grad()
            outputs = model(inputs)

            observables.compute_train_on_batch(outputs, labels)
            loss.compute(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % interval == interval - 1:
                if valloader is not None:
                    val_acc, val_outputs = eval_bayesian(model, valloader, number_of_tests=number_of_tests,
                                                         device=device, verbose=False)
                    observables.compute_val(val_acc, val_outputs)

                if verbose:
                    print('======================================')
                    print(f'Epoch [{epoch + 1}/{number_of_epochs}]. Batch [{batch_idx}/{number_of_batch}].')
                    loss.show()
                    observables.show()
                    print(f'Time Elapsed: {round(time() - start_time)} s')

                loss.write_tensorboard()
                observables.write_tensorboard()
                if output_dir_results is not None:
                    loss.write_results()
                    observables.write_results()

        observables.compute_train_on_epoch(model, trainloader, device)

    loss.close_writer()
    observables.close_writer()
    if verbose:
        print('Finished Training')

    return loss.results(), observables.results()
