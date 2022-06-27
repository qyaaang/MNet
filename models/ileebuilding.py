#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 19/05/22 11:18
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
                 k_c2
        m_w2 *----------* m_f2
             |          |
             |k_w2      |k_f2
             |          |
             |   k_c1   |
        m_w1 *----------* m_f1
             |          |
             |k_w1      |k_f1
             |          |
             |          |
             *          *
        :return:
        """
        m = np.diag([self.args.m_w1, self.args.m_w2, self.args.m_f1, self.args.m_f2])
        k_w1, k_w2 = self.params[0], self.params[1]
        k_f1, k_f2 = self.params[2], self.params[3]
        k_c1, k_c2 = self.params[4], self.params[5]
        k = np.array([[k_w1 + k_w2 + k_c1, -k_w2, -k_c1, 0],
                      [-k_w2, k_w2 + k_c2, 0, -k_c2],
                      [-k_c1, 0, k_f1 + k_f2 + k_c1, -k_f2],
                      [0, -k_c2, -k_f2, k_f2 + k_c2]])
        return m, k


if __name__ == '__main__':
    from mnet.eigen.inverse_matrix_iteration import IMI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_w1', default=2.88, type=float)
    parser.add_argument('--m_w2', default=1.585, type=float)
    parser.add_argument('--m_f1', default=23.22, type=float)
    parser.add_argument('--m_f2', default=16.265, type=float)
    args = parser.parse_args()
    # params = np.array([25986.13, 25986.13, 156.25, 156.25, 20000, 20000])
    params = np.array([29497.69, 41408.2, -3484.42, 21323.96, 26512.93, 30848.15])
    # params = np.array([85072.3, 18707.25, 11085.58, 12979.36, -6896.1, 73622.23])

    mdl = ILEEBlg(args, params)
    m_train, k_train = mdl.model()
    mode = IMI(m_train, k_train)
    w_pred, phi_pred = mode()
    print(w_pred / (2 * np.pi))
