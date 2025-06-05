import json
import math
import random
import itertools
import logging
import statistics
import debug
import torch
import copy
import numpy

from context import initialize_training_context
from filewise_export_stats import ms_ssim_reshape
from phases.fitting import fit_with_config
from phases.infer import patched_forward
from phases.qat import generate_quantization_config, initialize_quantizers, set_quantization_max_symbol
from phases.qat.eval import calculate_eval_value
from phases.qat.utils import apply_max_symbols_in_config, extract_max_symbols, load_parameters_stats

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

from ax import optimize

from utils import load_device, replace_config

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def find_best_max_symbols(config, model, unquantized_state_dict):
    quantization_search_config = copy.deepcopy(config["quantization_search"])
    replace_config(quantization_search_config, config["quantization"])

    drop_tolerance = quantization_search_config["drop_tolerance"]

    default_config = generate_quantization_config(quantization_search_config, model)
    parameters_stats = load_parameters_stats(default_config, model)

    search_config = quantization_search_config["max_symbol_search"] 

    def calculate_value_and_cost(entropy):
        max_symbols = symbols_dict_from_entropy(entropy)
        staging_config = apply_max_symbols_in_config(max_symbols, copy.deepcopy(default_config))
        qat_with_config(quantization_search_config, model, unquantized_state_dict, 
                                        staging_config)

        cost = calculate_cost(entropy, parameters_stats)
        eval_value = calculate_eval_value(model, quantization_search_config["eval_mode"])

        return eval_value, cost

    symbol_search_config = quantization_search_config["max_symbol_search"]

    def symbols_dict_from_entropy(entropy):
        symbols_dict = dict()
        for key in default_config:
            symbols_dict[key] = int(2 ** entropy) - 1
        return symbols_dict

    ref_entropy = symbol_search_config["ref_entropy"]
    ref_eval_value, ref_cost = calculate_value_and_cost(ref_entropy)
    best_entropy = ref_entropy
    best_gain = 0.0

    LOGGER.debug(f"Reference eval value: {ref_eval_value} (Cost: {ref_cost})")

    def evaluate(entropy):
        eval_value, cost = calculate_value_and_cost(entropy)
        LOGGER.debug(f"Eval value: {eval_value}, Cost: {cost})")

        eval_delta = 1.0 - (eval_value / ref_eval_value)
        cost_delta = 1.0 - (cost / ref_cost)

        gain = cost_delta * drop_tolerance - eval_delta

        LOGGER.debug(f"Gain: {gain} (Eval delta: {eval_delta}, Cost delta: {cost_delta})")
        return gain

    max_steps = symbol_search_config["max_steps"]
    samples_per_step = symbol_search_config["samples_per_step"]

    interval = symbol_search_config["first_interval"]
    step = -1
    for step in range(0, max_steps):
        LOGGER.debug(f"### Step {step+1}/{max_steps}, interval: {interval}")
        step_samples = numpy.linspace(interval[0], interval[1], samples_per_step)
        if step > 0:
            step_samples = step_samples[1:-1]
            interval_updated = False
        else:
            step_best_gain = None
            step_best_entropy = None

            step_second_best_gain = None
            step_second_best_entropy = None

        for staging_entropy in step_samples:
            LOGGER.debug(f"## Evaluating with entropy {staging_entropy}")
            LOGGER.debug(f"# Best entropies: [{step_best_entropy}, {step_second_best_entropy}]")
            LOGGER.debug(f"# Best gains: [{step_best_gain}, {step_second_best_gain}]")
            gain = evaluate(staging_entropy)

            if step_best_gain is None or gain > step_best_gain:
                interval_updated = True

                step_second_best_gain = step_best_gain
                step_second_best_entropy = step_best_entropy

                step_best_gain = gain
                step_best_entropy = staging_entropy
            elif step_second_best_gain is None or gain > step_second_best_gain:
                step_second_best_gain = gain
                step_second_best_entropy = staging_entropy

        if step_best_gain > best_gain:
            best_gain = step_best_gain
            best_entropy = step_best_entropy

        if step_second_best_entropy is None:
            LOGGER.warning("Only one best entropy found in this step, collapsing interval")
            step_second_best_entropy = step_best_entropy

        if abs(step_best_entropy - step_second_best_entropy) < symbol_search_config["precision"]:
            break

        if not interval_updated:
            break

        interval = [
            min(step_second_best_entropy, step_best_entropy), 
            max(step_second_best_entropy, step_best_entropy)
        ]

    LOGGER.debug(f"Found best entropy: {best_entropy} (in {step+1} steps)")

    best_symbols_dict = symbols_dict_from_entropy(best_entropy)
    return best_symbols_dict

def calculate_cost(entropy, parameters_stats):
    costs = list()
    for key in parameters_stats:
        stats = parameters_stats[key]
        cost = entropy * stats["num_elements"]
        costs.append(cost)

    return statistics.mean(costs)

def qat_with_config(config, model, base_state_dict, quantization_config):
    with torch.no_grad():
        model.load_state_dict(copy.deepcopy(base_state_dict))
        context = initialize_training_context(config, model)

    initialize_quantizers(quantization_config, model)
    # set_quantization_max_symbol(model, max_symbol)

    return fit_with_config(context, config, model, verbose=False, writer=debug.WRITER)
