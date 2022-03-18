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
            s, r = np.zeros_like(params), np.zeros_like(params)
            v = None
        grad = NumericalDiff(self.loss, params)
        param_names = ['EI']
        lh = {}
        t1 = time.time()
        # scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
        scheduler = MultiStepLR(optimizer, [20000, 25000, 30000], gamma=0.1)
        for epoch in range(self.args.num_epoch):
            lh[epoch + 1] = {}
            grads = grad()
            if self.args.optimizer == 'GD':
                optimizer.step(grads)
            else:
                optimizer.step(grads, epoch + 1)
            loss = self.loss(params)
            for idx, param_name in enumerate(param_names):
                lh[epoch + 1][param_name] = params[idx]
            lh[epoch + 1]['Loss'] = loss
            scheduler.step(epoch)
            if epoch % 100 == 0:
                print('\033[1;32mEpoch: {:06d}\033[0m\t'
                      '\033[1;31mLoss: {:.8f}\033[0m\t'
                      '\033[1;34mEI: {:.6f}\033[0m\t'
                      '\033[1;33mLR: {:.5f}\033[0m'.format(epoch, loss, params[0], optimizer.lr))
            if loss <= self.args.tol:
                break
        t2 = time.time()
        print('\033[1;33mTime cost: {:.2f}s\033[0m'.format(t2 - t1))
        # lh = json.dumps(lh, indent=2)
        # with open('./results/3dof_beam/lh_L{}_{}_{}.json'.
        #           format(self.args.norm, self.args.optimizer, self.args.lr), 'w') as f:
        #     f.write(lh)
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
    parser.add_argument('--m', default=6.25, type=float)  # Mass
    parser.add_argument('--h', default=1, type=float)  # Length
    args = parser.parse_args()
    exp = BaseExp(args)
    params_real = np.array([1.25e5])
    exp.init_mdl_real(params_real)
    params_trial = np.array([2e5])
    params_pred = exp.train(params_trial)
    print(params_pred)