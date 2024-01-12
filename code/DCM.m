clc
clearvars

addpath('../D-STEM/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Building model     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("traffic.mat")

% Variables
ns = size(traffic.Y{1}, 1);   % number of stations    
T = size(traffic.Y{1}, 2);    % number of time steps

% latent variables => markovian behaivour
traffic.X_z{1} = ones(ns, 1);
traffic.X_z_name{1} = {'constant'};

% Model containg all the variables
obj_stem_varset_p = stem_varset(traffic.Y, traffic.Y_name, ...
                                [],[], ...
                                traffic.X_beta, traffic.X_beta_name, ...
                                traffic.X_z, traffic.X_z_name, ...
                                traffic.X_p, traffic.X_p_name);


% Coordinates
obj_stem_gridlist_p = stem_gridlist();
traffic.coordinates{1} = [traffic.latitude, traffic.longitude];
obj_stem_grid = stem_grid(traffic.coordinates{1}, 'deg', 'sparse', 'point');
obj_stem_gridlist_p.add(obj_stem_grid);


obj_stem_datestamp = stem_datestamp(traffic.time_ini, traffic.time_fin, T);
shape = shaperead('../D-STEM/Maps/worldmap');
obj_stem_modeltype = stem_modeltype('DCM');

obj_stem_data = stem_data(obj_stem_varset_p, obj_stem_gridlist_p, ...
                          [], [], obj_stem_datestamp, [], obj_stem_modeltype, shape);

% Model creation
obj_stem_par_constraints = stem_par_constraints();
obj_stem_par = stem_par(obj_stem_data, 'exponential', obj_stem_par_constraints);
obj_stem_model = stem_model(obj_stem_data, obj_stem_par);

% Data transform
obj_stem_model.stem_data.log_transform;
obj_stem_model.stem_data.standardize;

% Starting values
obj_stem_par.beta = obj_stem_model.get_beta0();
obj_stem_par.theta_p = [100 100 100]; %km
obj_stem_par.v_p = eye(3,3);
obj_stem_par.sigma_eta = 0.2;
obj_stem_par.G = 0.8;
obj_stem_par.sigma_eps = 0.3;
obj_stem_model.set_initial_values(obj_stem_par);

% Model estimation
exit_toll = 0.001;
max_iterations = 100;
obj_stem_EM_options = stem_EM_options();
obj_stem_EM_options.max_iterations = 100;
obj_stem_model.EM_estimate(obj_stem_EM_options);
obj_stem_model.set_varcov;
obj_stem_model.set_logL;

% Kriging


%% Printing 

print(obj_stem_model)
obj_stem_model.stem_EM_result
plot(obj_stem_model.stem_EM_result.stem_kalmansmoother_result)
