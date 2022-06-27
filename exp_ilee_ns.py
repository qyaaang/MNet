#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 31/05/22 11:21
@description:  
@version: 1.0
"""


from mnet.eigen.inverse_matrix_iteration import IMI
from mnet.grad.numerical_diff import NumericalDiff
from mnet.optim.GD import GD
from mnet.optim.Adam import Adam
from mnet.optim.lr_scheduler import StepLR, MultiStepLR
from models.ilee_ns import ILEEBlg
import numpy as np
import argparse
import json
import time


class ILEEExp:

    def __init__(self, args):
        self.args = args
        self.mdl_real = None
        self.f = None
        self.phi = None
        self.lh = {}

    def init_mdl_real(self):
        self.f = np.array([3.15, 6.72])
        self.phi = np.zeros((2, 2))

    def forward(self, params):
        model_train = ILEEBlg(self.args, params)
        m_train, k_train = model_train.model()
        mode = IMI(m_train, k_train)
        w_pred, phi_pred = mode()
        return w_pred, phi_pred

    def loss(self, params):
        w_pred, phi_pred = self.forward(params)
        loss_w = np.linalg.norm(w_pred / (2 * np.pi) - self.f, ord=self.args.norm, axis=0)
        loss_phi = np.linalg.norm(phi_pred - self.phi, ord=self.args.norm)
        return np.power(loss_w, self.args.norm) + np.power(loss_phi, self.args.norm) * self.args.lm

    def train(self, params):
        if self.args.optimizer == 'GD':
            optimizer = GD(params, lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = Adam(params, lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        grad = NumericalDiff(self.loss, params)
        param_names = ['k_w', 'k_f', 'k_c']
        t1 = time.time()
        scheduler = MultiStepLR(optimizer, [2000, 15000, 20000, 25000], gamma=0.1)
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
            # scheduler.step(epoch)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--tol', default=1e-10, type=float)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--num_epoch', default=50000, type=int)
    parser.add_argument('--optimizer', default='GD', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--lm', default=0.1, type=float)  # Lagrange multiplier
    # parser.add_argument('--m_w', default=4.465, type=float)
    # parser.add_argument('--m_f', default=39.485, type=float)
    parser.add_argument('--m_w', default=4.465, type=float)
    parser.add_argument('--m_f', default=10.485, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    args = parser.parse_args()
    exp = ILEEExp(args)
    exp.init_mdl_real()
    params_trial = np.array([12993.07, 78.125, 10000])
    params_pred = exp.train(params_trial)
    exp.lh['Prediction'] = list(params_pred)
    print('\033[1;32mk_pred: {}\033[0m'.format([round(param, 2) for param in params_pred]))
    # lh = json.dumps(exp.lh, indent=2)
    # with open('./results/ILEE/L{}_{}_{}_{}.json'.
    #           format(args.norm, args.optimizer, args.lr, args.tol), 'w') as f:
    #     f.write(lh)
