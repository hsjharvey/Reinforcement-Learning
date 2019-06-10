# -*- coding:utf-8 -*-
from ..utils import *
import numpy as np
import tensorflow as tf


class CategoricalDQNActor():
    def __init__(self, config):
        self.atoms = tf.convert_to_tensor(
            np.linspace(
                config.Categorical_Vmin,
                config.Categorical_Vmax,
                config.Categorical_n_atoms,
            ))  # Z

        self.delta_z = (config.Categorical_Vmax - config.Categorical_Vmin) / float(config.Categorical_n_atoms - 1)

        self.NN_model = None


