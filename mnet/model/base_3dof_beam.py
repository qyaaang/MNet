#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 14:38
@description:  
@version: 1.0
"""


import numpy as np


class Base3DOFBeam:

    def __init__(self, m, h, ei):
        self.m = m
        self.h = h
        self.ei = ei
        self.num_dof = 3

    def assembly_mass_matrix(self):
        return self.m * np.diag([1, 1, 1])

    def assembly_stiff_matrix(self):
        a = ((81 * self.ei) / (13 * np.power(self.h, 3)))
        return a * np.array([[  7, -16,  12],
                             [-16,  44, -46],
                             [ 12, -46,  80]])

    def exact_mode(self):
        freqs = np.sqrt(self.ei / (self.m * np.power(self.h, 3))) * np.array([1.51979, 9.95139, 26.73744])
        mode_shape = np.array([[    1.0, -0.66331,  0.21519],
                               [0.53165,      1.0, -0.69898],
                               [0.15642,  0.84172,      1.0]])
        return freqs, mode_shape
