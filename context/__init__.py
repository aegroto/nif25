import torch
import copy


from context.downsample import build_downsample_case
from context.shuffle import build_shuffle_case
from losses import build_loss_fn
from phases.scheduler import CustomScheduler

class TrainingContext:
    def __init__(self):
        self.iteration = 0
        self.optimizer = None
        self.scheduler = None

        self.grid = None
        self.image = None

        self.shuffle = None

        self.loss_distance = 0.0

        self.cases = list()

    def initialize_batches(self, config, accumulation_increase=1):
        self.accumulation_increase = accumulation_increase
        del self.cases
        self.cases = list()

        config = copy.deepcopy(config)
        for case_config in config:
            mode = case_config["mode"]
            del case_config["mode"]

            repetitions = 1
            if "repeat" in case_config:
                repetitions = case_config["repeat"]
                del case_config["repeat"]

            if "accumulation" not in case_config:
                case_config["accumulation"] = 1

            case_config["accumulation"] *= accumulation_increase

            if mode == "shuffle":
                case = build_shuffle_case(self.grid, self.image, **case_config)
                self.shuffle = case_config["scale"]
            elif mode == "downsample":
                case = build_downsample_case(self.grid, self.image, **case_config)

            for _ in range(repetitions):
                self.cases.append(case)

    def get_shuffle(self):
        return self.shuffle

    def progress(self):
        return self.iteration / self.total_iterations

def initialize_training_context(config, model):
    context = TrainingContext()

    context.config = config

    context.optimizer = torch.optim.AdamW(model.parameters(), **config["training"]["optimizer"])
    context.scheduler = CustomScheduler(context.optimizer, **config["training"]["scheduler"])

    context.loss_fn = build_loss_fn(config["training"]["loss"], model, True)

    context.total_iterations = int(config["steps"] * config["training"]["iterations"])
    context.increase_progress = True

    return context 
