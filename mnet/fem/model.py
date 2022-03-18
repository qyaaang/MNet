#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/03/22 10:02
@description:  
@version: 1.0
"""


import numpy as np


class Model:

    def __init__(self):
        self.ndm = 0  # Spatial dimension of problem
        self.ndf = 0  # Number of degrees-of-freedom at nodes
        self.nodes = {}  # Node coordinates
        self.mat = {}  # Materials
        self.ele = {}  # Elements
        self.masses = {}  # Nodal masses
        self.m = None  # Global mass matrix
        self.k_ele = {}  # Element stiffness matrices
        self.k = None  # Global elastic stiffness matrix
        self.cstrts = {}  # Constraint conditions
        self.fixed_dof = []  # Fixed DOF
        self.active_dof = []  # Active DOF

    def mdl(self, ndm, ndf):
        """
        Define spatial dimension of problem and number of degrees-of-freedom at nodes
        :param ndm: Spatial dimension of problem
        :param ndf: Number of degrees-of-freedom at nodes
        """
        self.ndm = ndm
        self.ndf = ndf

    def node(self, tag, x, y):
        """
        Create nodes of modeling
        :param tag: Node tag
        :param x: x coordinate
        :param y: y coordinate
        """
        self.nodes[str(tag)] = [x, y]

    def get_num_node(self):
        """
        Get number of nodes
        :return: Number of nodes
        """
        return len(self.nodes.keys())

    def get_num_dof(self):
        """
        Get number of DOF
        :return: Number of DOF
        """
        num_node = self.get_num_node()  # Number of nodes
        return num_node * self.ndf

    def get_activate_dof(self):
        """
        Get activate DOF
        :return:
        """
        num_dof = self.get_num_dof()
        all_dof = [dof + 1 for dof in range(num_dof)]
        active_dof = list(set(all_dof) - set(self.fixed_dof))
        active_dof_i = [[dof - 1] for dof in active_dof]
        active_dof_j = [[dof - 1 for dof in active_dof]]
        active_dof = [dof - 1 for dof in active_dof]
        return active_dof_i, active_dof_j, active_dof

    def mass(self, node_tag, *mass_values):
        """
        Create nodal masses
        :param node_tag: Node mass
        :param mass_values:
        """
        self.masses[str(node_tag)] = mass_values
        if len(mass_values) != self.ndf:
            raise Exception('The length of nodal mass vectors must be equal to the ndf.')

    def fix(self, node_tag, *cstrts):
        """
        Fix degrees of freedom
        :param node_tag: Node needs to be fixed
        :param cstrts: Constraint conditions
        """
        self.cstrts[str(node_tag)] = cstrts
        if len(cstrts) == self.ndf:
            for idx, cstrt in enumerate(cstrts):
                if cstrt == 1:
                    self.fixed_dof.append(self.ndf * (node_tag - 1) + idx + 1)
        else:
            raise Exception('The length of constraint conditions must be equal to the ndf.')

    def locate_ele(self, node_i, node_j):
        """
        Locate element DOF in the global stiffness matrix
        :param node_i: Node i
        :param node_j: Node j
        :return: Element DOF in the global stiffness matrix
        """
        dof_i = [self.ndf * node_i - (self.ndf - i) for i in range(self.ndf)]
        dof_j = [self.ndf * node_j - (self.ndf - i) for i in range(self.ndf)]
        dofs = dof_i + dof_j
        ele_dof_i = [[dof] for dof in dofs]
        ele_dof_j = [dofs]
        return ele_dof_i, ele_dof_j

    def assemble_k(self, num_dof):
        """
        Assemble element stiffness matrix
        :param num_dof: Number of DOF
        """
        self.k = np.zeros((num_dof, num_dof))  # Global stiffness matrix
        for ele in self.k_ele.keys():
            node_i, node_j = self.ele[ele][0], self.ele[ele][1]
            ele_dof_i, ele_dof_j = self.locate_ele(node_i, node_j)
            self.k[ele_dof_i, ele_dof_j] = self.k[ele_dof_i, ele_dof_j] + self.k_ele[ele]

    def assemble_m(self, num_dof):
        """
        Assemble mass matrix
        :param num_dof: Number of DOF
        """
        self.m = np.zeros((num_dof, num_dof))  # Global mass matrix
        for node in self.nodes.keys():
            dof_i = [[dof] for dof in range(2 * int(node) - 2, 2 * int(node))]
            dof_j = [[dof for dof in range(2 * int(node) - 2, 2 * int(node))]]
            self.m[dof_i, dof_j] = np.diag(self.masses[node])

    def system(self):
        """
        Store the system of equations
        """
        num_dof = self.get_num_dof()
        i, j, self.active_dof = self.get_activate_dof()  # Active DOF
        self.assemble_k(num_dof)  # Assemble stiffness matrix
        self.assemble_m(num_dof)
        return self.m[i, j], self.k[i, j]
