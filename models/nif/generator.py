import torch
from torch import nn
from context.shuffle import shuffle_grid, shuffle_image
from input_encoding import generate_coordinates_tensor 
import torch.nn.functional as F

from models.nif.processors import NIFPostProcessor, NIFTargetPreProcessor
from modules.latent_grids import LatentGrids
from modules.positional_encoder import PositionalEncoder
from modules.positional_encoder.learned import LearnedPositionalEncoder
from modules.positional_encoder.spectra import SpectraPositionalEncoder

class NIFGenerator(nn.Module):
    def __init__(self, 
                frequency_params,
                height, width, 
                shuffle=1,
                accumulation_shuffle=1,
                device=None):
        super(NIFGenerator, self).__init__()

        self.device = device

        self.height = height 
        self.width = width

        self.shuffle = shuffle
        self.accumulation_shuffle = accumulation_shuffle

        self.target_preprocessor = NIFTargetPreProcessor()
        self.target_postprocessor = NIFPostProcessor()

        self.coordinates = generate_coordinates_tensor(width, height, device)
        self.frequency_encoder = PositionalEncoder(self.coordinates.size(-1), **frequency_params)
        self.out_dim = self.frequency_encoder.out_dim

        self.target_tensor = None

    def set_shuffle(self, value):
        self.shuffle = value

    def set_accumulation_shuffle(self, value):
        self.accumulation_shuffle = value

    def set_target(self, image):
        preprocessed_tensor = self.target_preprocessor(image).to(self.device)
        target_tensor = self.target_postprocessor.calibrate(preprocessed_tensor)
        self.target_tensor = target_tensor

    def generate_input(self):
        p_features = self.frequency_encoder(self.coordinates)
        shuffled_p_features = F.pixel_unshuffle(p_features.unsqueeze(0).movedim(-1, 0), self.shuffle) \
                                .movedim(0, -1)
        unsplit_batches = shuffled_p_features.unbind(0)

        batches = list()
        for unsplit_batch in unsplit_batches:
            batch = F.pixel_unshuffle(unsplit_batch.unsqueeze(0).movedim(-1, 0), self.accumulation_shuffle) \
                                .movedim(0, -1) \
                                .unbind(0)
            batches.append(batch)

        return batches 

    def generate_target(self):
        t_features = self.target_tensor
        shuffled_t_features = F.pixel_unshuffle(t_features.unsqueeze(1), self.shuffle) 
        unsplit_batches = shuffled_t_features.unbind(1)

        batches = list()
        for unsplit_batch in unsplit_batches:
            batch = F.pixel_unshuffle(unsplit_batch.unsqueeze(1), self.accumulation_shuffle) \
                                .unbind(1)
            batches.append(batch)

        return batches 

    def get_shuffling(self):
        return self.shuffle * self.accumulation_shuffle

    def generate_training_set(self):
        input_batches = self.generate_input()
        target_batches = self.generate_target()

        set = list()
        for i in range(0, len(input_batches)):
            input_batch = input_batches[i]
            target_batch = target_batches[i]

            batch = list(zip(input_batch, target_batch))
            set.append(batch)

        return set

