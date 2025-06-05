import debug
import statistics
import math
import time

import torch
from phases.qat import recalibrate_quantizers, set_quantization_amount

from utils import clearlines, get_or, linear_reduction, printdots

def train_model(context, model, config, verbose=True, writer=None, overwrite_state=False):
    loss_fn = context.loss_fn

    optimizer = context.optimizer
    scheduler = context.scheduler

    iterations = config["iterations"]
    noise_vars = config["training_noise"]
    # optimizer.zero_grad()

    # model.train()

    log_length = 8 + 3
    stop_training = False

    if verbose:
        printdots(log_length)

    last_log_time = time.time()
    iteration = 0
    while iteration < iterations and not stop_training:
        iteration += 1
        if context.increase_progress:
            context.iteration += 1

        context.quantization_amount = linear_reduction(noise_vars["start"], noise_vars["end"], 
                                        math.pow(context.progress(), noise_vars["smoothing"]))

        set_quantization_amount(model, context.quantization_amount)

        if iteration % noise_vars["recalibration_interval"]:
            recalibrate_quantizers(model)

        loss_values = list()

        training_set = model.generator.generate_training_set()
        for batch in training_set:
            loss_norm = 1.0 / len(batch)
            optimizer.zero_grad()
            for (input_sample, original_sample) in batch:
                reconstructed_sample = model(input_sample)

                (loss_value, loss_components) = loss_fn(original_sample, reconstructed_sample)

                loss_value = loss_value * loss_norm
                loss_value.backward(retain_graph=True)

                loss_values.append(loss_value.item())

            optimizer.step()

        context.avg_loss = statistics.mean(loss_values)
        
        context.selector.check_best(context, model)

        scheduler.step()

        if iteration % config["log_interval"] == 0 or stop_training:
            clearlines(log_length)

            log_time = time.time()
            past_time = log_time - last_log_time
            last_log_time = log_time
            iterations_per_second = config["log_interval"] / past_time
            estimated_time_remaining = (iterations - iteration) / (iterations_per_second * 60)
            estimated_minutes_remaining = math.floor(estimated_time_remaining)
            estimated_seconds_remaining = math.floor((estimated_time_remaining % 1) * 60)

            with torch.no_grad():
                learning_rate = scheduler._last_lr

                if verbose:
                    print("#" * 70)
                    print(f"# Iteration {iteration}/{iterations}, Learning rate: {learning_rate[0]}")
                    print(f"# Iterations/s: {iterations_per_second:.2f}, estimated time remaining: {estimated_minutes_remaining}m{estimated_seconds_remaining}s")
                    print(f"# Training noise: {context.quantization_amount}")

                if writer is not None:
                    writer.add_scalar("learning_rate", learning_rate[0], context.iteration)
                    writer.add_scalar("loss", loss_value, context.iteration)
                    writer.flush()

            if verbose:
                print(f"# Total loss: {context.avg_loss:.10f}")
                for (idx, key, value) in loss_components:
                    print(f"## [{idx}] {key}: {value:.10f}")

                print("#" * 70)

        debug.STEP += 1

    if overwrite_state:
        with torch.no_grad():
            if context.selector.best_value is not None:
                model.load_state_dict(context.selector.best_state_dict)
                context.optimizer.load_state_dict(context.selector.best_optimizer_state_dict)

    return context.selector.best_value
