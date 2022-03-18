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


class TrussElement:

    def __init__(self, model):
        self.model = model

    def truss_element(self, tag, node_i, node_j, a, mat_tag):
        """
        2D truss element
        :param tag: Element tag
        :param node_i: End node i
        :param node_j: End node j
        :param a: Cross-sectional area of element
        :param mat_tag: Material tag
        """
        e = self.model.mat[str(mat_tag)]['e']
        self.model.ele[str(tag)] = [node_i, node_j]
        x_i, y_i = self.model.nodes[str(node_i)][0], self.model.nodes[str(node_i)][1]
        x_j, y_j = self.model.nodes[str(node_j)][0], self.model.nodes[str(node_j)][1]
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
        self.model.k_ele[str(tag)] = np.dot(np.dot(t.T, k_ele), t)  # Global element stiffness matrix


class BeamElement2D:

    def __init__(self, model):
        self.model = model

    def beam_element_2d(self, tag, node_i, node_j, a, e, i_z):
        """
        2D elastic Euler beam element
        :param tag: Element tag
        :param node_i: End node i
        :param node_j: End node j
        :param a: Cross-sectional area of element
        :param e: Young's Modulus
        :param i_z: Second moment of area about the local z-axis
        """
        self.model.ele[str(tag)] = [node_i, node_j]
        x_i, y_i = self.model.nodes[str(node_i)][0], self.model.nodes[str(node_i)][1]
        x_j, y_j = self.model.nodes[str(node_j)][0], self.model.nodes[str(node_j)][1]
        delta_x = x_j - x_i
        delta_y = y_j - y_i
        l_ele = np.sqrt(delta_x ** 2 + delta_y ** 2)  # Element length
        c = delta_x / l_ele
        s = delta_y / l_ele
        t = np.array([[c,  s, 0, 0,  0, 0],
                      [-s, c, 0, 0,  0, 0],
                      [0,  0, 1, 0,  0, 0],
                      [0,  0, 0, c,  s, 0],
                      [0,  0, 0, -s, c, 0],
                      [0,  0, 0, 0,  0, 1]])  # Transfer matrix
        e_a = e * a / l_ele
        e_i1 = e * i_z / l_ele
        e_i2 = 6.0 * e * i_z / (l_ele ** 2)
        e_i3 = 12.0 * e * i_z / (l_ele ** 3)
        k_ele = np.array([[e_a,     0,        0, -e_a,     0,        0],
                          [0,    e_i3,     e_i2,    0, -e_i3,     e_i2],
                          [0,    e_i2, 4 * e_i1,    0, -e_i2, 2 * e_i1],
                          [-e_a,    0,        0,  e_a,     0,        0],
                          [0,   -e_i3,    -e_i2,    0,  e_i3,    -e_i2],
                          [0,    e_i2, 2 * e_i1,    0, -e_i2, 4 * e_i1]])  # Local element stiffness matrix
        self.model.k_ele[str(tag)] = np.dot(np.dot(t.T, k_ele), t)  # Global element stiffness matrix
