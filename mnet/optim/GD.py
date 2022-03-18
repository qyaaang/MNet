#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 10:13
@description:  
@version: 1.0
"""


import numpy as np


class GD:

    def __init__(self, params, lr, momentum):
        self.params = params  # Parameters
        self.lr = lr  # Learning rate
        self.momentum = momentum  # Momentum
        self.v = np.zeros_like(params)  # Velocity

    def step(self, grads):
        """
        Update parameters
        :param grads: Current gradients
        :return: Updated parameters
        """
        self.v = self.momentum * self.v - self.lr * grads
        self.params += self.v
