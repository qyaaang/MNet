#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 15/03/22 08:59
@description:  
@version: 1.0
"""


import copy


class StepLR:

    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer  # Wrapped optimizer
        self.step_size = step_size  # Period of learning rate decay
        self.gamma = gamma  # Multiplicative factor of learning rate decay. Default: 0.1

    def step(self, epoch):
        if epoch % self.step_size == 0 and epoch > 0:
            self.optimizer.lr *= self.gamma


class MultiStepLR:

    """
    Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    """

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer  # Wrapped optimizer
        self.milestones = milestones  # List of epoch indices. Must be increasing
        self.gamma = gamma  # Multiplicative factor of learning rate decay. Default: 0.1
        self.lr_tmp = copy.deepcopy(self.optimizer.lr)

    def locate(self, epoch):
        if epoch < self.milestones[0]:
            return 0
        for i in range(len(self.milestones) - 1):
            if self.milestones[i] <= epoch < self.milestones[i + 1]:
                return i + 1
        if epoch >= self.milestones[-1]:
            return len(self.milestones)

    def step(self, epoch):
        idx = self.locate(epoch)
        if idx > 0:
            if epoch == self.milestones[idx - 1]:
                self.optimizer.lr = self.lr_tmp * self.gamma ** int(idx)


class ExponentialLR:

    """
    Decays the learning rate of each parameter group by gamma every epoch.
    """

    def __init__(self, optimizer, gamma=0.1):
        self.optimizer = optimizer  # Wrapped optimizer
        self.gamma = gamma  # Multiplicative factor of learning rate decay. Default: 0.1

    def step(self):
        self.optimizer.lr *= self.gamma
