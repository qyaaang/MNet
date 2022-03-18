#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 16:13
@description:  
@version: 1.0
"""


import numpy as np


class Base2DOF:

    def __init__(self, rou, l, a, e):
        self.rou = rou  # Density
        self.l = l  # Length
        self.a = a  # Area
        self.e = e   # Elastic modules
        self.num_dof = 2

    def assembly_mass_matrix(self):
        a = (self.rou * self. a * self.l) / 12
        return a * np.array([[6, 1],
                             [1, 2]])

    def assembly_stiff_matrix(self):
        a = (2 * self.a * self.e) / self.l
        return a * np.array([[3, -1],
                             [-1, 1]])

    def exact_mode(self):
        freqs = np.zeros(self.num_dof)
        freqs[0] = np.sqrt(self.e / (self.rou * np.power(self.l, 2))) * 1.983851
        freqs[1] = np.sqrt(self.e / (self.rou * np.power(self.l, 2))) * 5.158468
        mode_shape = np.array([[0.57735027, -0.57735027],
                               [        1.,           1]])
        return freqs, mode_shape
