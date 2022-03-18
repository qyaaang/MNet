#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 09:17
@description:  
@version: 1.0
"""


import numpy as np


class Base2DOF:

    def __init__(self, m1, m2, k1, k2):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.num_dof = 2

    def assembly_mass_matrix(self):
        return np.diag([self.m1, self.m2])

    def assembly_stiff_matrix(self):
        return np.array([[ self.k1,          -self.k1],
                         [-self.k1, self.k1 + self.k2]])

    def exact_mode(self):
        freqs = np.zeros(self.num_dof)
        mode_shape = np.zeros((self.num_dof, self.num_dof))
        a = self.m1 * self.m2
        b = self.k1 * (self.m1 + self.m2) + self.k2 * self.m1
        c = self.k1 * self.k2
        w1 = (b - np.sqrt(np.power(b, 2) - 4 * a * c)) / (2 * a)
        w2 = (b + np.sqrt(np.power(b, 2) - 4 * a * c)) / (2 * a)
        freqs[0] = np.sqrt(w1)
        freqs[1] = np.sqrt(w2)
        mode_shape[:, 0] = np.array([1, (self.k1 - w1 * self.m1) / self.k1])
        mode_shape[:, 1] = np.array([self.k1 / (self.k1 - w2 * self.m1), 1])
        return freqs, mode_shape


if __name__ == '__main__':
    model = Base2DOF(10, 10, 1000, 1000)
    m = model.assembly_mass_matrix()
    k = model.assembly_stiff_matrix()
    w, phi = model.exact_mode()
