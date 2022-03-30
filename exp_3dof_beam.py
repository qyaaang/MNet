#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 14/03/22 14:16
@description:  
@version: 1.0
"""


from mnet.eigen.inverse_matrix_iteration import IMI
from mnet.model.base_3dof_beam import Base3DOFBeam
from mnet.grad.numerical_diff import NumericalDiff
from mnet.optim.GD import GD
from mnet.optim.Adam import Adam
from mnet.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import argparse
import json
import time


class BaseExp:

    def __init__(self, args):
        self.args = args
        self.mdl_real = None
        self.w = None
        self.phi = None
        self.lh = {}

    def init_mdl_real(self, params):
        self.mdl_real = Base3DOFBeam(self.args.m, self.args.h, params[0])
        self.w, self.phi = self.mdl_real.exact_mode()

    def forward(self, params):
        model_train = Base3DOFBeam(self.args.m, self.args.h, params[0])
        m_train = model_train.assembly_mass_matrix()
        k_train = model_train.assembly_stiff_matrix()
        mode = IMI(m_train, k_train)
        w_pred, phi_pred = mode()
        return w_pred, phi_pred

    def loss(self, params):
        w_pred, phi_pred = self.forward(params)
        loss_w = np.linalg.norm(w_pred - self.w, ord=self.args.norm, axis=0)
        loss_phi = np.linalg.norm(phi_pred - self.phi, ord=self.args.norm)
        return np.power(loss_w, self.args.norm) + np.power(loss_phi, self.args.norm) * self.args.lm

    def train(self, params):
        if self.args.optimizer == 'GD':
            optimizer = GD(params, lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = Adam(params, lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        grad = NumericalDiff(self.loss, params)
        param_names = ['EI']
        t1 = time.time()
        if self.args.optimizer == 'GD':
            scheduler = MultiStepLR(optimizer, [50000, 52000, 55000], gamma=0.1)  # GD
        else:
            scheduler = MultiStepLR(optimizer, [10000, 15000, 18000], gamma=0.1)  # Adam
        for epoch in range(self.args.num_epoch):
            self.lh[epoch + 1] = {}
            grads = grad()
            if self.args.optimizer == 'GD':
                optimizer.step(grads)
            else:
                optimizer.step(grads, epoch + 1)
            loss = self.loss(params)
            for idx, param_name in enumerate(param_names):
                self.lh[epoch + 1][param_name] = params[idx]
            self.lh[epoch + 1]['Loss'] = loss
            if self.args.lr_scheduler:
                scheduler.step(epoch)
            if epoch % 100 == 0:
                print('\033[1;32mEpoch: {:06d}\033[0m\t'
                      '\033[1;31mLoss: {:.16f}\033[0m\t'
                      '\033[1;34mEI: {:.6f}\033[0m\t'
                      '\033[1;33mLR: {:.5f}\033[0m'.format(epoch, loss, params[0], optimizer.lr))
            if loss <= self.args.tol:
                break
        t2 = time.time()
        print('\033[1;33mTime cost: {:.2f}s\033[0m'.format(t2 - t1))
        return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--tol', default=1e-10, type=float)
    parser.add_argument('--num_epoch', default=50000, type=int)
    parser.add_argument('--optimizer', default='GD', type=str)
    parser.add_argument('--lr', default=10, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--lm', default=0.1, type=float)  # Lagrange multiplier
    parser.add_argument('--m', default=100, type=float)  # Mass (kg)
    parser.add_argument('--h', default=3, type=float)  # Length (m)
    parser.add_argument('--lr_scheduler', action='store_true')
    args = parser.parse_args()
    exp = BaseExp(args)
    params_real = np.array([1.25e5])
    exp.init_mdl_real(params_real)
    # ei_0 = 1.3e5  # Bending stiffness (Nm2)
    ei_0 = 1.2e5
    params_trial = np.array([ei_0])
    params_pred = exp.train(params_trial)
    err = np.linalg.norm(params_pred - params_real, ord=1)
    exp.lh['Prediction'] = list(params_pred)
    exp.lh['Error'] = err
    print('\033[1;32mE_pred: {}\tError:{}\033[0m'.format(params_pred[0], err))
    lh = json.dumps(exp.lh, indent=2)
    with open('./results/3dof_beam/L{}_{}_{}_{}_{}.json'.
              format(args.norm, args.optimizer, args.lr, args.tol, ei_0), 'w') as f:
        f.write(lh)
