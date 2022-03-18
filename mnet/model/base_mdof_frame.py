#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 15:59
@description:  
@version: 1.0
"""


import numpy as np


class MDOFFrame:

    def __init__(self, mass, stiff):
        self.mass = mass
        self.stiff = stiff
        self.num_dof = self.stiff.shape[0]

    def assembly_mass_matrix(self):
        return np.diag(self.mass)

    def assembly_stiff_matrix(self):
        k = np.zeros((self.num_dof + 1, self.num_dof + 1))
        for ele, k_val in enumerate(self.stiff):
            k_ele = np.array([[ k_val, -k_val],
                              [-k_val,  k_val]])
            i = [[ele], [ele + 1]]
            j = [ele, ele + 1]
            k[i, j] = k[i, j] + k_ele
        return k[0: self.num_dof, 0: self.num_dof]


if __name__ == '__main__':
    ms = np.array([1, 2, 3, 4])
    ks = np.array([800, 1600, 2400, 3200])
    model = MDOFFrame(ms, ks)
    M = model.assembly_mass_matrix()
    K = model.assembly_stiff_matrix()
