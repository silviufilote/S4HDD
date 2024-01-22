clc
clearvars
close all

addpath('../D-STEM');
load("traffic.mat")


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Building model     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ns = size(traffic.Y{1}, 1);                         % number of stations    
T = size(traffic.Y{1}, 2);                          % number of time steps

% latent variables => markovian behaivour
traffic.X_z{1} = ones(ns, 1);
traffic.X_z_name{1} = {'constant'};

% latent variable => sptially correlated
traffic.X_p = traffic.X_spa;
traffic.X_p_name = traffic.X_spa_name;

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


% Validation
obj_stem_modeltype = stem_modeltype('DCM');
S_val= 1:2:ns;
obj_stem_validation = stem_validation({'traffic'}, {S_val}, 0, {'point'});

obj_stem_data = stem_data(obj_stem_varset_p, obj_stem_gridlist_p, ...
                          [], [], obj_stem_datestamp, obj_stem_validation, obj_stem_modeltype, shape);

% Model creation
obj_stem_par_constraints = stem_par_constraints();
obj_stem_par = stem_par(obj_stem_data, 'exponential', obj_stem_par_constraints);
obj_stem_model = stem_model(obj_stem_data, obj_stem_par);

% Data transform
obj_stem_model.stem_data.log_transform;
obj_stem_model.stem_data.standardize;

% Starting values: try different starting values and see if they converge
% at the same values 
obj_stem_par.beta = obj_stem_model.get_beta0();
obj_stem_par.theta_p = [100 100 100]; %km
% obj_stem_par.v_p = [1];
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

% Result of the EM estimation
EM_result = obj_stem_model.stem_EM_result;

% location of stations 
gs = geoscatter(traffic.latitude, traffic.longitude);
geobasemap("topographic") 
geolimits([40 41],[-112 -111.60]) 
gs.MarkerFaceColor = [0, 0.270, 0.2410]

% Starting values: try different starting values and see if they converge
% at the same values


% Kriging


% CV validation attention at the cofficients used in order to estimate and
% compare the model



%% Printing 
% After estimation the parameters are updated in the obj_stem_par 

print(obj_stem_model)
obj_stem_model.stem_EM_result;
plot(obj_stem_model.stem_EM_result.stem_kalmansmoother_result)

figure 
plot(EM_result.R2)

figure 
plot(EM_result.res{1}(3,:))

figure 
plot(obj_stem_model.stem_validation_result{1}.res(3,:))

