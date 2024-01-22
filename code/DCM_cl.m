clc
clearvars
close all

addpath('../D-STEM/');
addpath('../D-STEM/Src/');
load("dtraffic.mat")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Building model     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ns = size(dtraffic.Y{1}, 1);                         % number of stations    
T = size(dtraffic.Y{1}, 2);                          % number of time steps

% Process 1
X_beta = dtraffic.X_beta{1};
X_beta_name = dtraffic.X_beta_name{1};
X_z = ones(ns, 1);
X_z_name = {'constant'};
X_p = dtraffic.X_spa{1};
X_p_name = dtraffic.X_spa_name{1};
theta_p = [30 30 30];
sigma_eta = 0.2;
G = 0.8;

[obj_stem_model1, obj_stem_validation1, EM_result1] = model_estimate(dtraffic, X_beta, X_beta_name, ...
                                                                               X_z, X_z_name, ...
                                                                               X_p, X_p_name, ...
                                                                               theta_p, sigma_eta, G, 2);

% % Process 2
% X_beta = dtraffic.X_beta{1};
% X_beta_name = dtraffic.X_beta_name{1};
% X_z = dtraffic.X_spa{1};
% X_z_name = dtraffic.X_spa_name{1};
% X_p = ones(ns, 1);
% X_p_name = {'constant'};
% theta_p = 30;
% sigma_eta = diag([0.2 0.2 0.2]);
% G = diag([0.8 0.8 0.8]);
% 
% [obj_stem_model2, obj_stem_validation2, EM_result2] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 2);

% % Process 3
% X_beta = ones(ns,4,T);
% X_beta(:,1,:) = dtraffic.X_beta{1}(:,1,:); 
% X_beta(:,2,:) = dtraffic.X_beta{1}(:,3,:); 
% X_beta(:,3,:) = dtraffic.X_beta{1}(:,6,:); 
% X_beta(:,4,:) = dtraffic.X_beta{1}(:,7,:); 
% X_beta_name = {'weekend', 'mean prec', 'US', 'RS'};
% 
% X_z = ones(ns, 1);
% X_z_name = {'constant'};
% X_p = dtraffic.X_spa{1};
% X_p_name = dtraffic.X_spa_name{1};
% theta_p = [30 30 30];
% sigma_eta = 0.2;
% G = 0.8;
% 
% [obj_stem_model3, obj_stem_validation3, EM_result3] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 2);

% 
% % Process 4
% X_beta = ones(ns,4,T);
% X_beta(:,1,:) = dtraffic.X_beta{1}(:,1,:); 
% X_beta(:,2,:) = dtraffic.X_beta{1}(:,3,:); 
% X_beta(:,3,:) = dtraffic.X_beta{1}(:,6,:); 
% X_beta(:,4,:) = dtraffic.X_beta{1}(:,7,:); 
% X_beta_name = {'weekend', 'mean prec', 'US', 'RS'};
% 
% X_z = ones(ns, 1);
% X_z_name = {'constant'};
% X_p = dtraffic.X_spa{1};
% X_p_name = dtraffic.X_spa_name{1};
% theta_p = [100 100 100];
% sigma_eta = 0.8;
% G = 0.2;
% 
% 
% 
% [obj_stem_model3a, obj_stem_validation3a, EM_result3a] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 2);


% % Process 4
% X_beta = ones(ns,4,T);
% X_beta(:,1,:) = dtraffic.X_beta{1}(:,1,:); 
% X_beta(:,2,:) = dtraffic.X_beta{1}(:,3,:); 
% X_beta(:,3,:) = dtraffic.X_beta{1}(:,6,:); 
% X_beta(:,4,:) = dtraffic.X_beta{1}(:,7,:); 
% X_beta_name = {'weekend', 'mean prec', 'US', 'RS'};
% 
% X_z = ones(ns, 1);
% X_z_name = {'constant'};
% X_p = dtraffic.X_spa{1};
% X_p_name = dtraffic.X_spa_name{1};
% theta_p = [100 100 100];
% sigma_eta = 0.2;
% G = 0.2;
% 
% [obj_stem_model3b, obj_stem_validation3b, EM_result3b] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 2);

% 
% % Process 4
% X_beta = ones(ns,4,T);
% X_beta(:,1,:) = dtraffic.X_beta{1}(:,1,:); 
% X_beta(:,2,:) = dtraffic.X_beta{1}(:,3,:); 
% X_beta(:,3,:) = dtraffic.X_beta{1}(:,6,:); 
% X_beta(:,4,:) = dtraffic.X_beta{1}(:,7,:); 
% X_beta_name = {'weekend', 'mean prec', 'US', 'RS'};
% 
% X_z = ones(ns, 1);
% X_z_name = {'constant'};
% X_p = dtraffic.X_spa{1};
% X_p_name = dtraffic.X_spa_name{1};
% theta_p = [100 100 100];
% sigma_eta = 1;
% G = - 0.9;
% 
% [obj_stem_model3c, obj_stem_validation3c, EM_result3c] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Verify convergence EM     %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% X_beta = ones(ns,4,T);
% X_beta(:,1,:) = dtraffic.X_beta{1}(:,1,:); 
% X_beta(:,2,:) = dtraffic.X_beta{1}(:,3,:); 
% X_beta(:,3,:) = dtraffic.X_beta{1}(:,6,:); 
% X_beta(:,4,:) = dtraffic.X_beta{1}(:,7,:); 
% X_beta_name = {'weekend', 'mean prec', 'US', 'RS'};
% 
% X_z = ones(ns, 1);
% X_z_name = {'constant'};
% X_p = dtraffic.X_spa{1};
% X_p_name = dtraffic.X_spa_name{1};
% theta_p = [100 100 100];                    % poco identificabili
% 
% models_AIC.info = ones(20,4);
% for x = 1:20
%     sigma_eta = randi(10,1) *  rand(1,1);
%     G = -1 + 2*rand(1,1);
%     [obj_stem_model, obj_stem_validation, EM_result] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 2);
% 
%     models_AIC.info(x,:) = [obj_stem_model.stem_EM_result.logL obj_stem_model.stem_EM_result.AIC sigma_eta G];
%     models_AIC.models{x} = obj_stem_model;
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                kriging                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

krig_lat = [40.771278; 40.771018; 40.771538];
krig_lon = [-112.144589; -112.133259; -112.088627];
krig.coordinates = [krig_lat(:) krig_lon(:)];

% generate covariates data
krig.covariates_data = ones(3, 8, size(dtraffic.Y{1},2));
krig.covariates_names = {'weekend', 'holidays', 'mean temp', 'mean prec', 'interstate', 'US', 'RS', 'constant'};

% obj_stem_krig_grid = stem_grid(krig.coordinates, 'deg', 'sparse', 'point', [], 'square', 0.5, 0.5);
obj_stem_krig_grid = stem_grid(krig.coordinates, 'deg', 'sparse', 'point');
obj_stem_krig_data = stem_krig_data(obj_stem_krig_grid, krig.covariates_data, krig.covariates_names);
obj_stem_krig = stem_krig(obj_stem_model1, obj_stem_krig_data);

obj_stem_krig_options = stem_krig_options();
obj_stem_krig_result = obj_stem_krig.kriging(obj_stem_krig_options);
obj_stem_krig_result{1}.plot(1)

clear krig_lat krig_lon

% location of stations 
gs = geoscatter(dtraffic.latitude, dtraffic.longitude);
geobasemap("topographic") 
geolimits([40 41],[-112 -111.60]) 
gs.MarkerFaceColor = [0, 0.270, 0.2410]


% CV validation attention at the cofficients used in order to estimate and
% compare the model


% residual analysis



%% Printing 
% After estimation the parameters are updated in the obj_stem_par 

print(obj_stem_model1)
obj_stem_model.stem_EM_result;
plot(obj_stem_model1.stem_EM_result.stem_kalmansmoother_result)
plot(obj_stem_model2.stem_EM_result.stem_kalmansmoother_result)

print(obj_stem_model2)

figure 
plot(EM_result.R2)

figure 
plot(EM_result.res{1}(3,:))

figure 
plot(obj_stem_model.stem_validation_result{1}.res(3,:))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Residual analysis     %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tiledlayout(2,3)
nexttile
plot(EM_result2.res{1}(3,:))

nexttile
plot(EM_result.res{1}(6,:))

nexttile
plot(EM_result.res{1}(22,:))

nexttile
plot(EM_result.res{1}(21,:))

nexttile
plot(EM_result.res{1}(17,:))

nexttile
plot(EM_result.res{1}(11,:))


tiledlayout(2,3)
nexttile
autocorr(EM_result.res{1}(3,:))

nexttile
autocorr(EM_result.res{1}(6,:))

nexttile
autocorr(EM_result.res{1}(22,:))

nexttile
autocorr(EM_result.res{1}(21,:))

nexttile
autocorr(EM_result.res{1}(17,:))

nexttile
autocorr(EM_result.res{1}(11,:))


tiledlayout(2,3)
nexttile
histogram(EM_result.res{1}(3,:))

nexttile
histogram(EM_result.res{1}(6,:))

nexttile
histogram(EM_result.res{1}(22,:))

nexttile
histogram(EM_result.res{1}(21,:))

nexttile
histogram(EM_result.res{1}(17,:))

nexttile
histogram(EM_result.res{1}(11,:))


% lbqtest - H0: series of residuals exhibits no autocorrelation
tests = zeros(1,size(EM_result.res{1},1));
size = size(EM_result.res{1},1);

for x = 1:size
    tests(1,x) = lbqtest(EM_result.res{1}(x,:));
end

residualsTest.lbqtest{1} = tests;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Estimating model      %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [obj_stem_model, obj_stem_validation, EM_result] = model_estimate(dtraffic, X_beta, X_beta_name, X_z, X_z_name, X_p, X_p_name, theta_p, sigma_eta, G, nIterations)
    ns = size(dtraffic.Y{1}, 1);                         % number of stations    
    T = size(dtraffic.Y{1}, 2);                          % number of time steps
    
    % fixed effect model => x_beta
    dtraffic.X_beta{1} = X_beta;
    dtraffic.X_beta_name{1} = X_beta_name;

    % latent variables => markovian behaivour
    dtraffic.X_z{1} = X_z;
    dtraffic.X_z_name{1} = X_z_name;

    % latent variable => sptially correlated
    dtraffic.X_p{1} = X_p;
    dtraffic.X_p_name{1} = X_p_name;
    
    % Model containg all the variables
    obj_stem_varset_p = stem_varset(dtraffic.Y, dtraffic.Y_name, ...
                                    [],[], ...
                                    dtraffic.X_beta, dtraffic.X_beta_name, ...
                                    dtraffic.X_z, dtraffic.X_z_name, ...
                                    dtraffic.X_p, dtraffic.X_p_name);
    
    
    % Coordinates
    obj_stem_gridlist_p = stem_gridlist();
    dtraffic.coordinates{1} = [dtraffic.latitude, dtraffic.longitude];
    obj_stem_grid = stem_grid(dtraffic.coordinates{1}, 'deg', 'sparse', 'point');
    obj_stem_gridlist_p.add(obj_stem_grid);
    
    
    obj_stem_datestamp = stem_datestamp(dtraffic.time_ini, dtraffic.time_fin, T);
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
    obj_stem_par.theta_p = theta_p; % poco identificabile
    % obj_stem_par.v_p = 2 * ones(1,1,3);
    obj_stem_par.sigma_eta = sigma_eta;
    obj_stem_par.G = G;
    obj_stem_par.sigma_eps = 0.3;
    obj_stem_model.set_initial_values(obj_stem_par);
    
    % Model estimation
    obj_stem_EM_options = stem_EM_options();
    obj_stem_EM_options.max_iterations = nIterations;
    obj_stem_model.EM_estimate(obj_stem_EM_options);
    obj_stem_model.set_varcov;
    obj_stem_model.set_logL;
    
    
    % Result of the EM estimation
    EM_result = obj_stem_model.stem_EM_result;
end