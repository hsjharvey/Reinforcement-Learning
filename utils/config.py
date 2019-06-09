# -*- coding:utf-8 -*-
import tensorflow as tf


class Config:
    device = tf.device('/gpu:0')

    def __init__(self):
        pass
