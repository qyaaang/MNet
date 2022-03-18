#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 15/03/22 15:00
@description:  
@version: 1.0
"""


from mnet.eigen.inverse_matrix_iteration import IMI
from mnet.model.base_mdof_frame import MDOFFrame
from mnet.grad.numerical_diff import NumericalDiff
from mnet.optim.GD import GD
from mnet.optim.Adam import Adam
from mnet.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import argparse
import json
import time


class FrameModel:

    def __init__(self, params):
        self.params = params
        self.mass = None

    def __call__(self, *args, **kwargs):
        m, k = self.model()
        return m, k

    def init_mass(self, mass):
        self.mass = mass

    def model(self):
        model = MDOFFrame(self.mass, self.params)
        m = model.assembly_mass_matrix()
        k = model.assembly_stiff_matrix()
        return m, k


class BaseExp:

    def __init__(self, args):
        self.args = args
        self.mdl_real = None
        self.mass = None
        self.w = None
        self.phi = None

    def init_mdl_real(self, mass, params):
        self.mdl_real = FrameModel(params)
        self.mdl_real.init_mass(mass)
        m, k = self.mdl_real()
        mode = IMI(m, k)
        w, phi = mode()
        # np.save('./data/{}dof_freq.npy'.format(self.args.num_dof), w)
        # np.save('./data/{}dof_mode_shape.npy'.format(self.args.num_dof), phi)

    def load_dataset(self):
        self.w = np.load('./data/{}dof_freq.npy'.format(self.args.num_dof))
        self.phi = np.load('./data/{}dof_mode_shape.npy'.format(self.args.num_dof))

    def forward(self, params):
        model_train = FrameModel(params)
        model_train.init_mass(self.mdl_real.mass)
        m_train, k_train = model_train()
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
        param_names = ['k1', 'k2', 'k3', 'k4']
        lh = {}
        t1 = time.time()
        # scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
        # scheduler = MultiStepLR(optimizer, [20000, 25000, 30000, 35000], gamma=0.1)
        # scheduler = MultiStepLR(optimizer, [2000, 3000, 4000, 5000], gamma=0.1)
        scheduler = MultiStepLR(optimizer, [11000, 13000, 15000], gamma=0.1)
        # scheduler = MultiStepLR(optimizer, [20000, 50000, 100000, 110000, 150000, 170000], gamma=0.5)
        for epoch in range(self.args.num_epoch):
            lh[epoch + 1] = {}
            grads = grad()  # Gradients
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
                      '\033[1;34mk: {}\033[0m\t'
                      '\033[1;33mLR: {:.5f}\033[0m'.
                      format(epoch, loss, [round(param, 2) for param in params], optimizer.lr))
            if loss <= self.args.tol:
                break
        t2 = time.time()
        print('\033[1;33mTime cost: {:.2f}s\033[0m'.format(t2 - t1))
        # lh = json.dumps(lh, indent=2)
        # with open('./results/{}dof/lh_L{}_{}_{}.json'.
        #           format(self.args.num_dof, self.args.norm, self.args.optimizer, self.args.lr), 'w') as f:
        #     f.write(lh)
        return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if 0:
        parser.add_argument('--norm', default=1, type=int)
        parser.add_argument('--tol', default=1e-10, type=float)
        parser.add_argument('--num_epoch', default=50000, type=int)
        parser.add_argument('--optimizer', default='GD', type=str)
        parser.add_argument('--lr', default=1, type=float)
    if 1:
        parser.add_argument('--norm', default=2, type=int)
        parser.add_argument('--tol', default=1e-10, type=float)
        parser.add_argument('--num_epoch', default=20000, type=int)
        parser.add_argument('--optimizer', default='Adam', type=str)
        parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--lm', default=0.1, type=float)  # Lagrange multiplier
    if 0:
        parser.add_argument('--num_dof', default=4, type=int)
        ms = np.array([1, 2, 2, 3])
        ks = np.array([800, 1600, 2400, 3200])
        params_trial = np.array([769., 1765., 2346., 3086.])
    if 1:
        parser.add_argument('--num_dof', default=10, type=int)
        ms = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        ks = np.array([1000, 2000, 2000, 3000, 3000, 3000, 3000, 4000, 5000, 5000])
        # params_trial = np.array([1100, 2100, 2100, 3100, 3200, 3200, 3150, 4020, 5100, 5100], dtype=np.float32)
        params_trial = np.array([1010, 2010, 2010, 3010, 3010, 3020, 3020, 4020, 5010, 5010], dtype=np.float32)
    args = parser.parse_args()
    exp = BaseExp(args)
    exp.init_mdl_real(ms, ks)
    exp.load_dataset()
    params_pred = exp.train(params_trial)
    print(params_pred)
