#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 31/05/22 11:12
@description:  
@version: 1.0
"""


import numpy as np


class ILEEBlg:

    def __init__(self, args, params):
        self.args = args
        self.params = params

    def model(self):
        """
                 k_c
        m_w  *----------* m_f
             |          |
             |k_w       |k_f
             |          |
             |          |
             *          *
        :return:
        """
        m = np.diag([self.args.m_w, self.args.m_f])
        k_w, k_f, k_c = self.params[0], self.params[1], self.params[2]
        k = np.array([[k_w + k_c, -k_c],
                      [-k_c, k_f + k_c],
                      ])
        return m, k


if __name__ == '__main__':
    from mnet.eigen.inverse_matrix_iteration import IMI
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--m_w', default=4.465, type=float)
    # parser.add_argument('--m_f', default=39.485, type=float)
    parser.add_argument('--m_w', default=4.465, type=float)
    parser.add_argument('--m_f', default=10.485, type=float)
    args = parser.parse_args()
    params = np.array([12993.07, 78.125, 8000])
    mdl = ILEEBlg(args, params)
    m_train, k_train = mdl.model()
    mode = IMI(m_train, k_train)
    w_pred, phi_pred = mode()
    print(w_pred / (2 * np.pi))
