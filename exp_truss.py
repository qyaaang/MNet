#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/03/22 10:12
@description:  
@version: 1.0
"""


from mnet.fem.model import Model
from mnet.fem.material import UniaxialMaterialElastic
from mnet.fem.element import TrussElement
from mnet.eigen.inverse_matrix_iteration import IMI
from mnet.grad.numerical_diff import NumericalDiff
from mnet.optim.GD import GD
from mnet.optim.Adam import Adam
from mnet.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import argparse
import json
import time


class TrussModel:

    def __init__(self, params):
        self.model = Model()
        self.model.mdl(2, 2)
        self.params = params  # Parameters

    def __call__(self, *args, **kwargs):
        self.node()
        self.fix()
        self.mass()
        self.material()
        self.element()
        m, k = self.model.system()
        return m, k

    def node(self):
        self.model.node(1, 0, 0)
        self.model.node(2, 3000, 0)
        self.model.node(3, 6000, 0)
        self.model.node(4, 9000, 0)
        self.model.node(5, 1500, 3000)
        self.model.node(6, 4500, 3000)
        self.model.node(7, 7500, 3000)

    def fix(self):
        self.model.fix(1, 1, 1)
        self.model.fix(4, 0, 1)

    def mass(self):
        l1 = 3000
        l2 = np.sqrt(3000 ** 2 + 1500 ** 2)
        a = 4532
        rou = 2.5e-9
        m1 = 0.5 * rou * a * l1
        m2 = 0.5 * rou * a * l2
        self.model.mass(1, 0.5 * (m1 + m2), 0.5 * (m1 + m2))
        self.model.mass(4, 0.5 * (m1 + m2), 0.5 * (m1 + m2))
        self.model.mass(5, 0.5 * m1 + m2, 0.5 * m1 + m2)
        self.model.mass(7, 0.5 * m1 + m2, 0.5 * m1 + m2)
        self.model.mass(2, m1 + m2, m1 + m2)
        self.model.mass(3, m1 + m2, m1 + m2)
        self.model.mass(6, m1 + m2, m1 + m2)

    def material(self):
        material = UniaxialMaterialElastic(self.model)
        material.uniaxial_material_elastic(1, self.params[0])
        material.uniaxial_material_elastic(2, self.params[1])
        material.uniaxial_material_elastic(3, self.params[2])

    def element(self):
        element = TrussElement(self.model)
        a = 4532
        element.truss_element(1, 1, 2, a, 2)
        element.truss_element(2, 2, 3, a, 2)
        element.truss_element(3, 3, 4, a, 2)
        element.truss_element(4, 5, 6, a, 1)
        element.truss_element(5, 6, 7, a, 1)
        element.truss_element(6, 1, 5, a, 3)
        element.truss_element(7, 2, 6, a, 3)
        element.truss_element(8, 3, 7, a, 3)
        element.truss_element(9, 2, 5, a, 3)
        element.truss_element(10, 3, 6, a, 3)
        element.truss_element(11, 4, 7, a, 3)


class BaseExp:

    def __init__(self, args):
        self.args = args
        self.w = None
        self.phi = None

    def init_mdl_real(self):
        truss = TrussModel(np.array([self.args.E1, self.args.E2, self.args.E3]))
        m, k = truss()
        mode = IMI(m, k)
        w_pred, phi_pred = mode()
        num_dof = truss.model.get_num_dof()
        phi = np.zeros((num_dof, phi_pred.shape[1]))
        for i in range(phi_pred.shape[1]):
            phi[truss.model.active_dof, i] = phi_pred[:, i]
        np.save('./data/truss_freq.npy', w_pred)
        np.save('./data/truss_mode_shape.npy', phi_pred)
        np.save('./data/truss_mode_shape_all.npy', phi)

    def forward(self, params):
        model_train = TrussModel(params)
        m_train, k_train = model_train()
        mode = IMI(m_train, k_train)
        w_pred, phi_pred = mode()
        return w_pred, phi_pred

    def load_dataset(self):
        self.w = np.load('./data/truss_freq.npy')
        self.phi = np.load('./data/truss_mode_shape.npy')

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
        self.load_dataset()
        grad = NumericalDiff(self.loss, params)
        param_names = ['E1', 'E2', 'E3']
        lh = {}
        t1 = time.time()
        # scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
        scheduler = MultiStepLR(optimizer, [3000, 4000, 5000], gamma=0.1)
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
                      '\033[1;34mE1: {:.6f} E2:{:.6f} E3:{:.6f}\033[0m\t'
                      '\033[1;33mLR: {:.5f}\033[0m'.
                      format(epoch, loss, params[0], params[1], params[2], optimizer.lr))
            if loss <= self.args.tol:
                break
        t2 = time.time()
        print('\033[1;33mTime cost: {:.2f}s\033[0m'.format(t2 - t1))
        # lh = json.dumps(lh, indent=2)
        # with open('./results/truss/lh_L{}_{}_{}.json'.
        #           format(self.args.norm, self.args.optimizer, self.args.lr), 'w') as f:
        #     f.write(lh)
        return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if 1:
        parser.add_argument('--norm', default=1, type=int)
        parser.add_argument('--tol', default=1e-10, type=float)
        parser.add_argument('--num_epoch', default=50000, type=int)
        parser.add_argument('--optimizer', default='GD', type=str)
        parser.add_argument('--lr', default=10, type=float)
    if 0:
        parser.add_argument('--norm', default=2, type=int)
        parser.add_argument('--tol', default=1e-20, type=float)
        parser.add_argument('--num_epoch', default=50000, type=int)
        parser.add_argument('--optimizer', default='Adam', type=str)
        parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--lm', default=0.1, type=float)  # Lagrange multiplier
    parser.add_argument('--E1', default=2e4, type=float)  # E1
    parser.add_argument('--E2', default=2.5e4, type=float)  # E2
    parser.add_argument('--E3', default=3e4, type=float)  # E3
    args = parser.parse_args()
    exp = BaseExp(args)
    exp.load_dataset()
    params_trial = np.array([1.5e4, 3e4, 2.5e4], dtype=np.float32)
    params_pred = exp.train(params_trial)
    print(params_pred)
