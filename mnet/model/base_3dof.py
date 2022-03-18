#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 11:48
@description:  
@version: 1.0
"""


import numpy as np


class Base3DOF:

    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.num_dof = 3

    def assembly_mass_matrix(self):
        return self.m * np.diag([2, 1, 1])

    def assembly_stiff_matrix(self):
        return (self.k / 15) * np.array([[20, -5,  0],
                                         [-5,  8, -3],
                                         [ 0, -3,  3]])

    def exact_mode(self):
        freqs = np.sqrt(self.k / self.m) * np.array([0.29357, 0.66734, 0.93192])
        mode_shape = np.array([[0.16339, 0.75306, -0.82589],
                               [0.56908,     1.0,      1.0],
                               [1.0,    -0.81517,  0.29919]])
        return freqs, mode_shape
