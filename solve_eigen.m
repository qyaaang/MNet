function [omega,phi] = solve_eigen(model_file,num_dof,params,mass)
%Compute frequcy and mode shape
%   Solve eigenvalue problem given M and K
    obj = py.importlib.import_module(model_file);
    py.importlib.reload(obj);
    if nargin==3
        params = py.numpy.array(params);
        mdl = py.exp_truss.TrussModel(params);
    end
    if nargin==4
        mass = py.numpy.array(mass);
        params = py.numpy.array(params);
        mdl = py.exp_mdof.FrameModel(params);
        mdl.init_mass(mass);
    end
    sys = mdl();
    m = sys(1);
    k = sys(2);
    M = double(py.array.array('d',py.numpy.nditer(m{1})));
    M = reshape(M,[num_dof num_dof]);
    K = double(py.array.array('d',py.numpy.nditer(k{1}))); 
    K = reshape(K,[num_dof num_dof]);
    [eig_vec,eig_val] = eig(M \ K);
    [omega,w_order]    = sort(sqrt(diag(eig_val)));   
    phi = eig_vec(:,w_order);  
    for i = 1:size(phi)
        [~,idx] =  max(abs(phi(:,i)));
        phi(:,i) = phi(:,i) / phi(idx, i);
    end
end

