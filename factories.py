# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
                        AdamW,
                        get_constant_schedule_with_warmup,
                        get_cosine_schedule_with_warmup,
                        get_linear_schedule_with_warmup,
                        get_polynomial_decay_schedule_with_warmup,
                        get_cosine_with_hard_restarts_schedule_with_warmup
)

from .yaml_config import CfgNode


def loss_fn_factory(config: CfgNode):
    """ A convenience function that initializes some of the common loss
        functions supported by PyTorch.

        Supports:
            - l1
            - mean squared error
            - binary cross entropy
            - binary cross entropy with logits
            - cross entropy
            - negative log likelihood
            - kullback-leibler divergence

        For more information on these loss functions, see the PyTorch
        documentation.

        Args:
            config: dict
                Contains the parameters needed to initialize the loss function.

        Returns:
            loss_fn: nn.Loss
                The loss function.

        Raises:
            ValueError
    """
    # Get all of the possible arguments we might need
    ignore_index = config.loss_fn.get("ignore_index", -100)
    pos_weight = config.loss_fn.get("pos_weight", None)
    reduce = config.loss_fn.get("reduce", None)
    reduction = config.loss_fn.get("reduction", "mean")
    size_average = config.loss_fn.get("size_average", None)
    weight = config.loss_fn.get("weight", None)
    if weight:
        weight = torch.FloatTensor(weight)
    if 'cpu' not in config.model.device and weight:
        weight = weight.cuda(config.model.device)

    if config.loss_fn.type == "l1":
        return nn.L1Loss(
            size_average=size_average, reduce=reduce, reduction=reduction)
    elif config.loss_fn.type == "mean squared error":
        return nn.MSELoss(
            size_average=size_average, reduce=reduce, reduction=reduction)
    elif config.loss_fn.type == "cross entropy":
        return nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction)
    elif config.loss_fn.type == "negative log likelihood":
        return nn.NLLLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce)
    elif config.loss_fn.type == "kullback-leibler divergence":
        return nn.KLDivLoss(
            size_average=size_average, reduce=reduce, reduction=reduction)
    elif config.loss_fn.type == "binary cross entropy":
        return nn.BCELoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction)
    elif config.loss_fn.type == "binary cross entropy with logits":
        return nn.BCEWithLogitsLoss(
            weight=None,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight)
    else:
        raise ValueError("Unrecognized loss_fn type.")


def optimizer_factory(config: CfgNode, params):
    """ A convenience function that initializes some of the common optimizers
        supported by PyTorch.
        Supports:
            - adadelta
            - adagrad
            - adam
            - adamw
            - adamax
            - rmsprop
            - sgd

        For more information on these optimizers, see the PyTorch documentation.

        Args:
            config: dict
                Contains the parameters needed to initialize the optimizer,
                such as the learning rate, weight decay, etc.
            params: iterable
                An iterable of parameters to optimize or dicts defining
                parameter groups.

        Returns:
            optim: optim.Optimizer
                An optimizer object
    """
    if config.optimizer.type == "adadelta":
        return optim.Adadelta(
            params,
            lr=config.optimizer.get("lr", 1.0),
            rho=config.optimizer.get("rho", 0.9),
            eps=config.optimizer.get("eps", 1e-6),
            weight_decay=config.optimizer.get("weight_decay", 0))
    elif config.optimizer.type == "adagrad":
        return optim.Adagrad(
            params,
            lr=config.optimizer.get("lr", 0.01),
            lr_decay=config.optimizer.get("lr_decay", 0),
            weight_decay=config.optimizer.get("weight_decay", 0),
            initial_accumulator_value=config.optimizer.get("initial_accumulator_value", 0))
    elif config.optimizer.type == "adam":
        return optim.Adam(
            params,
            lr=config.optimizer.get("lr", 0.001),
            betas=config.optimizer.get("betas", (0.9, 0.999)),
            eps=config.optimizer.get("eps", 1e-8),
            weight_decay=config.optimizer.get("weight_decay", 0),
            amsgrad=config.optimizer.get("amsgrad", False))
    elif config.optimizer.type == "adamw":
        return AdamW(
            params,
            lr=config.optimizer.get("lr", 0.001),
            betas=config.optimizer.get("betas", (0.9, 0.999)),
            eps=config.optimizer.get("eps", 1e-8),
            weight_decay=config.optimizer.get("weight_decay", 0),
            correct_bias=config.optimizer.get("correct_bias", True)
        )
    elif config.optimizer.type == "adamax":
        return optim.Adamax(
            params,
            lr=config.optimizer.get("lr", 0.002),
            betas=config.optimizer.get("betas", (0.9, 0.999)),
            eps=config.optimizer.get("eps", 1e-8),
            weight_decay=config.optimizer.get("weight_decay", 0))
    elif config.optimizer.type == "rmsprop":
        return optim.RMSProp(
            params,
            lr=config.optimizer.get("lr", 0.01),
            alpha=config.optimizer.get("alpha", 0.99),
            eps=config.optimizer.get("eps", 1e-8),
            weight_decay=config.optimizer.get("weight_decay", 0),
            momentum=config.optimizer.get("momentum", 0),
            centered=config.optimizer.get("centered", False))
    elif config.optimizer.type == "sgd":
        return optim.SGD(
            params,
            lr=config.optimizer.get("lr", 0.001),
            momentum=config.optimizer.get("momentum", 0),
            dampening=config.optimizer.get("dampening", 0),
            weight_decay=config.optimizer.get("weight_decay", 0),
            nesterov=config.optimizer.get("nesterov", False))
    else:
        raise ValueError("Unrecognized optimizer type.")


def scheduler_factory(config: CfgNode, optimizer: optim.Optimizer):
    """ A convenience function that initializes some of the common learning
        rate schedulers supported by PyTorch.
        Supports:
            - step learning rate
            - exponential learning rate
            - reduce learning rate on plateau

        For more information about these learning rate schedulers, see
        the PyTorch documentation.

        Args:
            config: dict
                Contains the parameters needed to initialize the scheduler.
            optimizer: optim.Optimizer
                The optimizer for which we want to adjust the learning rate.
        Returns:
            scheduler: optim.lr_scheduler
                The learning rate scheduler.
    """
    # Get all of the possible arguments we might need

    if config.scheduler.type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.get("step_size", 5),
            gamma=config.scheduler.get("gamma", 0.1),
            last_epoch=config.scheduler.get("last_epoch", -1))
    elif config.scheduler.type == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler.get("gamma", 0.5),
            last_epoch=config.scheduler.get("last_epoch", -1))
    elif config.scheduler.type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.get("mode", "min"),
            factor=config.scheduler.get("factor", 0.1),
            patience=config.scheduler.get("patience", 10),
            verbose=config.scheduler.get("verbose", False),
            threshold=config.scheduler.get("threshold", 1e-4),
            threshold_mode=config.scheduler.get("threshold_mode", "rel"),
            cooldown=config.scheduler.get("cooldown", 0),
            min_lr=config.scheduler.get("min_lr", 0),
            eps=config.scheduler.get("eps", 1e-8))
    else:
        raise ValueError("Unrecognized scheduler type")


def scheduler_with_warmup_factory(
                                config,
                                optimizer: optim.Optimizer,
                                num_warmup_steps: int,
                                num_training_steps: int) -> optim.lr_scheduler.LambdaLR:
    if config.scheduler.type == 'constant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif config.scheduler.type == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif config.scheduler.type == 'linear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif config.scheduler.type == 'polynomial':
        return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif config.scheduler.type == 'cosine_with_hard_restarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        raise ValueError("Unrecognized scheduler warmup type")


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def print_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(param.grad)
