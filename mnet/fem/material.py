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


class UniaxialMaterialElastic:

    def __init__(self, model):
        self.model = model

    def uniaxial_material_elastic(self, mat_tag, e):
        """
        Uniaxial elastic material
        :param mat_tag: Material tag
        :param e: Young's modules
        """
        self.model.mat[str(mat_tag)] = {'e': e}

