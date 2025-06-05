import copy
import math
import traceback
import logging
import torch
from context import initialize_training_context
from phases.selection import Selector

from phases.training import train_model
from phases.reset import perform_restart_step
from utils import replace_config

def fit_with_config(context, config, model, verbose = False, writer = None):
    restart_config = config["restart"]

    total_steps = config["steps"]

    training_selector = Selector(**config["selector"])
    post_training_selector = Selector(**config["selector"])

    context.training_set = model.generator.generate_training_set()
    context.selector = training_selector

    post_train = config["post_training"] is not None
    if post_train:
        post_training_config = copy.deepcopy(config)
        post_training_config["training"] = post_training_config["post_training"]
        replace_config(post_training_config["training"], config["training"])
        post_training_context = initialize_training_context(post_training_config, model)
        post_training_context.training_set = model.generator.generate_training_set()
        post_training_context.selector = post_training_selector

    for step in range(1, total_steps+1):
        if verbose:
            print(f"##########################")
            print(f"Step #{step}/{total_steps}")

        if training_selector.best_value is not None:
            with torch.no_grad():
                model.load_state_dict(training_selector.best_state_dict)
                context.optimizer.load_state_dict(training_selector.best_optimizer_state_dict)

                if restart_config: # and not last_step:
                    perform_restart_step(model, restart_config, context.progress(), verbose)

        context.increase_progress = True
        _ = train_model(context, model, config["training"],
                        verbose = verbose,
                        writer = writer,
                        overwrite_state = True) 

        if post_train:
            if verbose:
                print(f"Post-training...")

            post_training_value = train_model(post_training_context, model, post_training_config["training"],
                            verbose = False,
                            writer = writer,
                            overwrite_state = False) 

            if verbose:
                print(f"Post-training best value: {post_training_value} (Best: {post_training_selector.best_value})")

    if post_train:
        final_selector = post_training_selector
    else:
        final_selector = training_selector

    model.load_state_dict(final_selector.best_state_dict)

    return float(final_selector.best_value)

