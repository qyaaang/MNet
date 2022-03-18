#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 15:29
@description:  
@version: 1.0
"""


import numpy as np
import copy


class NumericalDiff:

    def __init__(self, criteria, params):
        self.criteria = criteria  # Criteria
        self.params = params  # Parameters

    def __call__(self, *args, **kwargs):
        return self.grad()

    def grad(self, h=0.001):
        """
        Compute gradients
        :param h: Step size
        :return: Gradients
        """
        grads = np.zeros_like(self.params)
        for idx in range(len(self.params)):
            param_tmp = copy.deepcopy(self.params)
            a = param_tmp[idx]
            param_tmp[idx] = a + h
            loss1 = self.criteria(param_tmp)
            param_tmp[idx] = a - h
            loss2 = self.criteria(param_tmp)
            grads[idx] = (loss1 - loss2) / (2 * h)
        return grads
