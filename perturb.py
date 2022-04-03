#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 15/03/22 21:46
@description:  
@version: 1.0
"""


import numpy as np
import argparse
import json


class Perturbation:

    def __init__(self, exp, params_fixed):
        self.exp = exp
        self.params_fixed = params_fixed

    def gen_perturb(self, num_pert, sigma):
        np.random.seed(10)
        num_params = len(self.params_fixed)
        if len(sigma) != num_params:
            raise ValueError('Length of sigma must be equal to number of parameters.')
        perturb = np.zeros((num_pert, num_params))  # Perturbations
        for i in range(num_params):
            perturb[:, i] = np.random.normal(0, sigma[i], num_pert)
        return perturb

    def run_exp(self, perturb, params_trial):
        num_perturb = perturb.shape[0]
        perturb_loss = np.zeros(num_perturb)  # Perturbation loss
        perturb_info = {}
        for i in range(num_perturb):
            params_perturb = self.params_fixed + perturb[i, :]
            perturb_info[i] = list(params_perturb)
            self.exp.init_mdl_real(params_perturb)
            print('\033[1;32mPerturbation test: {}\tPerturbed params:{}\033[0m'.format(i + 1, params_perturb))
            params_pred = self.exp.train(params_trial)
            perturb_loss[i] = np.linalg.norm(params_pred - params_perturb, ord=1)
        perturb_expect = np.mean(perturb_loss)  # Perturbation expectation
        perturb_info['Loss'] = list(perturb_loss)
        perturb_info['Expectation'] = perturb_expect
        perturb_info = json.dumps(perturb_info, indent=2)
        with open('./results/{}/perturbation.json'.format(self.exp.args.model), 'w') as f:
            f.write(perturb_info)
        print('\033[1;32mPerturbation expectation: {}\033[0m'.format(perturb_expect))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='GD', type=str)
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--tol', default=1e-10, type=float)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--num_epoch', default=50000, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)  # Beta1
    parser.add_argument('--beta_2', default=0.999, type=float)  # Beta2
    parser.add_argument('--lm', default=0.1, type=float)  # Lagrange multiplier
    # 2-DOF
    if 0:
        from exp_2dof import BaseExp
        parser.add_argument('--model', default='2dof', type=str)  # Model
        parser.add_argument('--m1', default=10, type=float)  # Mass 1
        parser.add_argument('--m2', default=10, type=float)  # Mass 2
        args = parser.parse_args()
        exp = BaseExp(args)
        params_fixed = np.array([1000., 1000.])
        params_trial = np.array([569., 765.])
        test = Perturbation(exp, params_fixed)
        perturb = test.gen_perturb(100, sigma=[0.1, 0.15])
        test.run_exp(perturb, params_trial)
    # 2-DOF FEM
    if 0:
        from exp_2dof_fem import BaseExp
        parser.add_argument('--model', default='2dof_fem', type=str)  # Model
        parser.add_argument('--rou', default=7.85e-9, type=float)  # Density
        parser.add_argument('--a', default=4000, type=float)  # Area
        parser.add_argument('--l', default=3000, type=float)  # Length
        parser.add_argument('--lr_scheduler', action='store_true')
        args = parser.parse_args()
        exp = BaseExp(args)
        params_fixed = np.array([2e5])
        params_trial = np.array([1.9e5])
        test = Perturbation(exp, params_fixed)
        perturb = test.gen_perturb(100, sigma=[0.1])
        test.run_exp(perturb, params_trial)
    # 3-DOF
    if 0:
        from exp_3dof import BaseExp
        parser.add_argument('--model', default='3dof', type=str)  # Model
        parser.add_argument('--m', default=10, type=float)  # Mass
        parser.add_argument('--lr_scheduler', action='store_true')
        args = parser.parse_args()
        exp = BaseExp(args)
        params_fixed = np.array([1000.])
        params_trial = np.array([1265.])
        test = Perturbation(exp, params_fixed)
        perturb = test.gen_perturb(100, sigma=[0.1])
        test.run_exp(perturb, params_trial)
    # 3-DOF beam
    if 1:
        from exp_3dof_beam import BaseExp
        parser.add_argument('--model', default='3dof_beam', type=str)  # Model
        parser.add_argument('--m', default=100, type=float)  # Mass (kg)
        parser.add_argument('--h', default=3, type=float)  # Length (m)
        parser.add_argument('--lr_scheduler', action='store_true')
        args = parser.parse_args()
        exp = BaseExp(args)
        params_fixed = np.array([1.25e5])
        params_trial = np.array([1.2e5])
        test = Perturbation(exp, params_fixed)
        perturb = test.gen_perturb(100, sigma=[0.1])
        test.run_exp(perturb, params_trial)
