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
from scipy.optimize import minimize, dual_annealing
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
        self.lh = {}

    def init_mdl_real(self, mass, params):
        self.mdl_real = FrameModel(params)
        self.mdl_real.init_mass(mass)
        m, k = self.mdl_real()
        mode = IMI(m, k)
        w, phi = mode()
        # np.save('./data/{}dof_freq.npy'.format(self.args.num_dof), w)
        # np.save('./data/{}dof_mode_shape.npy'.format(self.args.num_dof), phi)
        return w, phi

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
        param_names = ['k{}'.format(i + 1) for i in range(self.args.num_dof)]
        t1 = time.time()
        if self.args.num_dof == 4:
            # scheduler = MultiStepLR(optimizer, [20000, 25000, 30000, 35000], gamma=0.1)  # 4dof
            scheduler = MultiStepLR(optimizer, [25000], gamma=0.1)
        elif self.args.num_dof == 10:
            if self.args.optimizer == 'GD':
                scheduler = MultiStepLR(optimizer, [35000, 40000, 45000], gamma=0.1)  # 10dof GD L1 L2
            else:
                scheduler = MultiStepLR(optimizer, [26000, 30000, 35000, 40000], gamma=0.1)  # 10dof Adam L1 L2
        else:
            scheduler = MultiStepLR(optimizer, [2000, 2200, 2500], gamma=0.1)
        for epoch in range(self.args.num_epoch):
            self.lh[epoch + 1] = {}
            grads = grad()  # Gradients
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
                      '\033[1;31mLoss: {:.8f}\033[0m\t'
                      '\033[1;34mk: {}\033[0m\t'
                      '\033[1;33mLR: {:.5f}\033[0m'.
                      format(epoch, loss, [round(param, 2) for param in params], optimizer.lr))
            if loss <= self.args.tol:
                break
        t2 = time.time()
        print('\033[1;33mTime cost: {:.2f}s\033[0m'.format(t2 - t1))
        return params

    def global_optim(self, lw, up):
        ret = dual_annealing(self.loss, bounds=list(zip(lw, up)), maxfun=50000, seed=10)
        print(ret.x, ret.fun)
        return ret.x

    def local_optim(self, params):
        ret = minimize(self.loss, params, method='L-BFGS-B', jac='2-point',
                       options={'ftol': 1e-20, 'gtol': 1e-20, 'disp': True, 'eps': 1e-20,
                                'maxfun': 50000, 'maxiter': 50000})
        print(ret.x, ret.fun)
        return ret.x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_dof', default=4, type=int)
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--tol', default=1e-10, type=float)
    parser.add_argument('--num_epoch', default=50000, type=int)
    parser.add_argument('--optimizer', default='GD', type=str)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--lm', default=0.1, type=float)  # Lagrange multiplier
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--test', default=1, type=int)
    args = parser.parse_args()
    if args.num_dof == 4:
        mass = np.array([1, 2, 2, 3])
        params_real = np.array([800, 1600, 2400, 3200])
        if args.test == 1:
            params_trial = np.array([769., 1765., 2346., 3086.])  # 1
        elif args.test == 2:
            params_trial = np.array([100., 100., 100., 100.])  # 2
        elif args.test == 3:
            params_trial = np.array([100., 10000., 100., 10000.])  # 3
        else:
            params_trial = np.array([10000., 10000., 10000., 10000.])  # 4
        # lw = [100] * 4
        # up = [5000] * 4
    elif args.num_dof == 10:
        mass = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        params_real = np.array([1000, 2000, 2000, 3000, 3000, 3000, 3000, 4000, 5000, 5000])
        # params_trial = np.array([1010., 2010., 2010., 3010., 3010., 3020., 3020., 4020., 5010., 5010.])
        if args.test == 1:
            params_trial = np.array([1100., 2100., 2100., 3100., 3100., 3200., 3200., 4200., 5100., 5100.])
        else:
            params_trial = np.array([900., 1900., 1900., 2900., 2900., 2800., 2800., 3800., 4900., 4900.])
        # lw = [900, 1900, 1900, 2900, 2900, 2900, 2900, 3900, 4900, 4900]
        # up = [1100, 2100, 2100, 3100, 3100, 3100, 3100, 4100, 5100, 5100]
    else:
        mass = np.ones(args.num_dof)
        params_real = 1000 * np.ones(args.num_dof)
        if args.test == 1:
            params_trial = 950. * np.ones(args.num_dof)
        else:
            params_trial = 1050. * np.ones(args.num_dof)
            # lw = [500] * args.num_dof
        # up = [1500] * args.num_dof
    args = parser.parse_args()
    exp = BaseExp(args)
    exp.init_mdl_real(mass, params_real)
    exp.load_dataset()
    params_pred = exp.train(params_trial)
    err = np.linalg.norm(params_pred - params_real, ord=1)
    exp.lh['Prediction'] = list(params_pred)
    exp.lh['Error'] = err
    print('\033[1;32mParam_pred: {}\tError:{}\033[0m'.format(params_pred, err))
    lh = json.dumps(exp.lh, indent=2)
    with open('./results/{}dof/L{}_{}_{}_{}_{}.json'.
              format(args.num_dof, args.norm, args.optimizer, args.lr, args.tol, args.test), 'w') as f:
        f.write(lh)
    # params_pred = exp.global_optim(lw, up)
    # params_pred = exp.local_optim(params_trial)
