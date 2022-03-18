#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 10:15
@description:  
@version: 1.0
"""


import numpy as np


class Adam:

    def __init__(self, params, lr, betas):
        self.params = params  # Parameters
        self.lr = lr  # Learning rate
        self.betas = betas  # Exponential decay rates for the moment estimates
        self.s = np.zeros_like(params)  # Biased first moment estimate
        self.r = np.zeros_like(params)  # Biased second raw moment estimate

    def step(self, grads, t):
        """
        Update parameters
        :param grads: Current gradients
        :param t: Current time step
        :return: Updated parameters
        """
        self.s = self.betas[0] * self.s + (1 - self.betas[0]) * grads
        self.r = self.betas[1] * self.r + (1 - self.betas[1]) * np.multiply(grads, grads)
        s_hat = self.s / (1 - np.power(self.betas[0], t + 1))  # Bias-corrected first moment estimate
        r_hat = self.r / (1 - np.power(self.betas[1], t + 1))  # Bias-corrected second raw moment estimate
        delta_param = np.multiply(s_hat, 1 / (np.sqrt(r_hat) + 1e-8))
        self.params -= np.multiply(self.lr, delta_param)
