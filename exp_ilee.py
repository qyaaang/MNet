#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 19/05/22 14:30
@description:  
@version: 1.0
"""


from mnet.eigen.inverse_matrix_iteration import IMI
from mnet.grad.numerical_diff import NumericalDiff
from mnet.optim.GD import GD
from mnet.optim.Adam import Adam
from mnet.optim.lr_scheduler import StepLR, MultiStepLR
from models.ileebuilding import ILEEBlg
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
        # self.f = np.array([3.15479647673952, 6.71579459118375, 28.2332708140746, 40.3618718865138])
        self.f = np.array([2.94, 9.06, 22.51, 39.13])
        # self.phi = np.array([[0.400682014722270, 0.382343349855978, 0.464772682862388, 0.547205909664659],
        #                      [1, 1, 1, 1],
        #                      [0.624474774107691, -0.0691684349614421, 0.0286306630482624, 0.0159270168830770],
        #                      [0.770853279150979, 0.219198749699524, -0.0818871956233983, -0.0889123498083239]])
        self.phi = np.zeros((4, 4))

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
        param_names = ['k_w1', 'k_w2', 'k_f1', 'k_f2', 'k_c1', 'k_c2']
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
    parser.add_argument('--m_w1', default=2.88, type=float)
    parser.add_argument('--m_w2', default=1.585, type=float)
    parser.add_argument('--m_f1', default=23.22, type=float)
    parser.add_argument('--m_f2', default=16.265, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    args = parser.parse_args()
    exp = ILEEExp(args)
    exp.init_mdl_real()
    # params_trial = np.array([25986.13, 25986.13, 156.25, 156.25, 20000, 20000])
    # params_trial = np.array([53419.6, 29190.92, -7622.6, 7522.77, 26034.94, 58105.5])
    # params_trial = np.array([85072.3, 18707.25, 11085.58, 12979.36, -6896.1, 73622.23])

    # params_trial = np.array([27060.64, 28010.55, 928.22, 1527.07, 21300.53, 21847.37])
    # params_trial = np.array([27329.09, 28513.2, 1114.05, 1852.82, 21621.62, 22304.18])
    # params_trial = np.array([27461.56, 28760.8, 1204.91, 2011.52, 21779.6, 22528.96])
    # params_trial = np.array([30248.23, 35944.0, 543.77, 6373.28, 26008.01, 28911.03])
    # params_trial = np.array([34361.46, 41761.85, -3855.62, 5958.06, 30798.5, 36390.82])
    # params_trial = np.array([35302.3, 41039.83, -4547.12, 5857.42, 31937.46, 37406.35])
    # params_trial = np.array([37052.76, 39742.56, -5511.06, 5659.22, 33991.74, 39206.11])
    # params_trial = np.array([40420.55, 37224.75, -7347.71, 5369.92, 37963.96, 42687.63])
    # params_trial = np.array([43287.57, 35060.69, -8893.16, 5222.43, 41358.34, 45667.93])
    # params_trial = np.array([44228.68, 34358.86, -9391.36, 5196.71, 42443.18, 46631.0])

    # params_trial = np.array([44243.58, 34353.52, -9391.15, 5199.94, 42414.15, 46646.93]) # W-12
    # params_trial = np.array([43390.8, 32960.58, -9798.49, 6144.38, 41668.94, 45456.42])
    # params_trial = np.array([38489.63, 34583.72, -8458.13, 14750.76, 37037.13, 40449.47])
    # params_trial = np.array([31297.76, 39976.15, -4540.11, 21028.84, 28818.62, 32875.87])
    params_trial = np.array([30840.1, 40340.95, -4272.76, 21099.66, 28231.68, 32360.02])
    params_pred = exp.train(params_trial)
    exp.lh['Prediction'] = list(params_pred)
    print('\033[1;32mk_pred: {}\033[0m'.format([round(param, 2) for param in params_pred]))
    # lh = json.dumps(exp.lh, indent=2)
    # with open('./results/ILEE/L{}_{}_{}_{}.json'.
    #           format(args.norm, args.optimizer, args.lr, args.tol), 'w') as f:
    #     f.write(lh)
