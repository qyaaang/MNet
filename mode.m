clc;
clear;
%%
% model_file = 'exp_mdof';
% 4-DOF frame
% mass = [1, 2, 2, 3];
% params = [800, 1600, 2400, 3200];
% [omega, phi] = solve_eigen(model_file, 4, params, mass);
% py.numpy.save('./data/4dof_freq.npy', py.numpy.array(omega));
% py.numpy.save('./data/4dof_mode_shape.npy', py.numpy.array(phi));
% 10-DOF frame
% mass = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
% params = [1000, 2000, 2000, 3000, 3000, 3000, 3000, 4000, 5000, 5000];
% [omega, phi] = solve_eigen(model_file, 10, params, mass);
% py.numpy.save('./data/10dof_freq.npy', py.numpy.array(omega));
% py.numpy.save('./data/10dof_mode_shape.npy', py.numpy.array(phi));
%%
model_file = 'exp_truss';
params = [2e4, 2.5e4, 3e4];
[omega, phi] = solve_eigen(model_file, 11, params);
% py.numpy.save('./data/truss_freq.npy', py.numpy.array(omega));
% py.numpy.save('./data/truss_mode_shape.npy', py.numpy.array(phi));