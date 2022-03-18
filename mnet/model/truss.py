#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2022, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/03/22 09:00
@description:  
@version: 1.0
"""


import numpy as np


class Model:

    def __init__(self):
        self.ndm = None  # Spatial dimension of problem
        self.ndf = None  # Number of degrees-of-freedom at nodes
        self.nodes = {}  # Node coordinates
        self.mat = {}  # Materials
        self.ele = {}  # Elements
        self.masses = {}  # Nodal masses
        self.m = None  # Global mass matrix
        self.k_ele = {}  # Element stiffness matrices
        self.k = None  # Global elastic stiffness matrix
        self.cstrts = {}  # Constraint conditions
        self.fixed_dof = []  # Fixed DOF
        self.active_i, self.active_j = None, None

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

    def uniaxial_material_elastic(self, mat_tag, e):
        """
        Uniaxial elastic material
        :param mat_tag: Material tag
        :param e: Young's modules
        """
        self.mat[str(mat_tag)] = {'e': e}

    def truss_element(self, tag, node_i, node_j, a, mat_tag):
        """
        2D truss element
        :param tag: Element tag
        :param node_i: End node i
        :param node_j: End node j
        :param a: Cross-sectional area of element
        :param mat_tag: Material tag
        """
        e = self.mat[str(mat_tag)]['e']
        self.ele[str(tag)] = [node_i, node_j]
        x_i, y_i = self.nodes[str(node_i)][0], self.nodes[str(node_i)][1]
        x_j, y_j = self.nodes[str(node_j)][0], self.nodes[str(node_j)][1]
        delta_x = x_j - x_i
        delta_y = y_j - y_i
        l_ele = np.sqrt(delta_x ** 2 + delta_y ** 2)  # Element length
        c = delta_x / l_ele
        s = delta_y / l_ele
        t = np.array([[ c, s,  0, 0],
                      [-s, c,  0, 0],
                      [ 0, 0,  c, s],
                      [ 0, 0, -s, c]])  # Transfer matrix
        k_ele = ((a * e) / l_ele) * np.array([[1, 0, -1, 0],
                                              [0, 0, 0, 0],
                                              [-1, 0, 1, 0],
                                              [0, 0, 0, 0]])  # Local element stiffness matrix
        self.k_ele[str(tag)] = np.dot(np.dot(t.T, k_ele), t)  # Global element stiffness matrix

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

    def assemble_k(self, num_dof):
        """
        Assemble element stiffness matrix
        :param num_dof: Number of DOF
        """
        self.k = np.zeros((num_dof, num_dof))  # Global stiffness matrix
        for ele in self.k_ele.keys():
            node_i, node_j = self.ele[ele][0], self.ele[ele][1]
            ele_dof_i = [[2 * node_i - 2], [2 * node_i - 1],
                         [2 * node_j - 2], [2 * node_j - 1]]
            ele_dof_j = [[2 * node_i - 2, 2 * node_i - 1, 2 * node_j - 2, 2 * node_j - 1]]
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
        self.active_i, self.active_j, _ = self.get_activate_dof()  # Active DOF
        self.assemble_k(num_dof)  # Assemble stiffness matrix
        self.assemble_m(num_dof)
        return self.k[self.active_i, self.active_j]


if __name__ == '__main__':
    model = Model()
    model.mdl(2, 2)
    model.node(1, 0, 0)
    model.node(2, 3000, 0)
    model.node(3, 6000, 0)
    model.node(4, 9000, 0)
    model.node(5, 1500, 3000)
    model.node(6, 4500, 3000)
    model.node(7, 7500, 3000)
    for i in range(1, 8):
        model.mass(i, 1, 1)
    model.uniaxial_material_elastic(1, 2e5)
    a = 4532
    model.truss_element(1, 1, 2, a, 1)
    model.truss_element(2, 2, 3, a, 1)
    model.truss_element(3, 3, 4, a, 1)
    model.truss_element(4, 5, 6, a, 1)
    model.truss_element(5, 6, 7, a, 1)
    model.truss_element(6, 1, 5, a, 1)
    model.truss_element(7, 2, 6, a, 1)
    model.truss_element(8, 3, 7, a, 1)
    model.truss_element(9, 2, 5, a, 1)
    model.truss_element(10, 3, 6, a, 1)
    model.truss_element(11, 4, 7, a, 1)
    model.fix(1, 1, 1)
    model.fix(4, 0, 1)
    k = model.system()
