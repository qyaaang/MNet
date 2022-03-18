#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 09:09
@description:  
@version: 1.0
"""


import numpy as np


class IMI:

    def __init__(self, m, k):
        self.m = m  # Mass matrix
        self.k = k  # Stiffness matrix
        self.num_dof = k.shape[0]  # Number of DOF

    def __call__(self, *args, **kwargs):
        phi_pred = self.mode(100)
        w_pred = self.freq(phi_pred)
        return w_pred, phi_pred

    def trial_phi(self):
        """
        Initialize trial first mode shape vector
        :return: Trial first mode shape vector
        """
        a = np.zeros((self.num_dof, 1))
        a[0, 0] = 1.
        return a

    def matrix_iter(self, num_iter, e):
        """
        Matrix iteration
        :param num_iter: Number of iterations
        :param e: Iteration matrix
        :return: First mode shape vector
        """
        a = self.trial_phi()
        for _ in range(num_iter):
            b = e.dot(a)
            m_bar = b[np.argmax(np.abs(b))]
            a = b / m_bar
        return a

    def mode(self, num_iter):
        """
        Compute mode shape matrix
        :param num_iter: Number of iterations
        :return: Mode shape matrix
        """
        e = np.linalg.inv(self.k).dot(self.m)  # Iteration matrix
        i = np.eye(self.num_dof)
        s = i  # Clean matrix
        mode_shape = np.zeros((self.num_dof, self.num_dof))
        for idx in range(self.num_dof):
            e = np.dot(e, s)
            phi = self.matrix_iter(num_iter, e)
            m_star = np.dot(np.dot(phi.T, self.m), phi)
            s -= np.dot(np.dot(phi, phi.T), self.m) / m_star.item()
            mode_shape[:, idx] = phi[:, 0]
        return mode_shape

    def freq(self, mode_shape):
        """
        Compute frequency vector
        :param mode_shape: Mode shape matrix
        :return: Frequency vector
        """
        freqs = np.zeros(self.num_dof)
        for idx in range(self.num_dof):
            phai = mode_shape[:, idx]
            m_star = np.dot(np.dot(phai.T, self.m), phai)
            k_star = np.dot(np.dot(phai.T, self.k), phai)
            freqs[idx] = np.sqrt(k_star / m_star)
        return freqs
