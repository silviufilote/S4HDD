clc
clearvars
close all

addpath('../D-STEM/');
addpath('../D-STEM/Src/');
load("traffic.mat");
load("krig.mat");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Optimization log likelihood                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dtraffic, krig] = setup_traffic(traffic, krig, 24, 0);
clear traffic

% for x = 1:24
% 
%     ns = size(dtraffic.Y{1}, 1);                         % number of stations    
% 
%     % Process 1
%     X_beta = dtraffic.X_beta{1};
%     X_beta_name = dtraffic.X_beta_name{1};
%     X_z = ones(ns, 1);
%     X_z_name = {'constant'};
%     X_p = dtraffic.X_spa{1};
%     X_p_name = dtraffic.X_spa_name{1};
%     theta_p = [10 10 10];
%     sigma_eta = 0.2;
%     G = 0.8;
%     v = ones(1,1,3);
% 
%     [dtraffic, obj_stem_model, obj_stem_validation, EM_result] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                    X_z, X_z_name, ...
%                                                                                    X_p, X_p_name, ...
%                                                                                    theta_p, v, sigma_eta, G, 10);
% 
%     print_models(1,x) = obj_stem_model;
% end
% 
% 
% for x = 1:24
%     print(print_models(x));
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Building model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ns = size(dtraffic.Y{1}, 1);                         % number of stations    
T = size(dtraffic.Y{1}, 2);                          % number of time steps

% Process 1
X_beta = dtraffic.X_beta{1};
X_beta_name = dtraffic.X_beta_name{1};
X_z = dtraffic.X_spa{1};
X_z_name = dtraffic.X_spa_name{1};
X_p = ones(ns, 1);
X_p_name = {'constant'};
theta_p = 1;
sigma_eta = diag([0.2 0.2 0.2]);
G = diag([0.8 0.8 0.8]);
v = 1;

[dtraffic, obj_stem_model1, obj_stem_validation1, EM_result1] = model_estimate(dtraffic, X_beta, X_beta_name, ...
                                                                               X_z, X_z_name, ...
                                                                               X_p, X_p_name, ...
                                                                               theta_p, v, sigma_eta, G, 1);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Verify convergence EM                         %
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
%     [dtraffic, obj_stem_model, obj_stem_validation, EM_result] = model_estimate(dtraffic, X_beta, X_beta_name, ...
%                                                                                X_z, X_z_name, ...
%                                                                                X_p, X_p_name, ...
%                                                                                theta_p, sigma_eta, G, 100);
% 
%     models_AIC.info(x,:) = [obj_stem_model.stem_EM_result.logL obj_stem_model.stem_EM_result.AIC sigma_eta G];
%     models_AIC.models{x} = obj_stem_model;
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                kriging                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
gs2 = geoscatter(krig.coordinates(:,1),krig.coordinates(:,2) );
geobasemap("topographic") 
geolimits([40 41],[-112 -111.60]) 
gs2.MarkerFaceColor = [1 0 1];
title("Cluster selection")

% generate covariates data
krig.covariates_data{1} = zeros(4, 10, size(dtraffic.Y{1},2));
krig.covariates_names = {'weekend', 'holidays', 'mean temp', 'mean prec', 'traffic on', 'hours', 'interstate', 'US', 'RS', 'constant'};

% Beta krip enrichment
krig.covariates_data = zeros(4, 10, size(dtraffic.Y{1},2));
krig.covariates_data(:,1,:) = dtraffic.X_beta{1}(1:4,1,:);
krig.covariates_data(:,2,:) = dtraffic.X_beta{1}(1:4,2,:);
krig.covariates_data(:,3,:) = krig.temp_mean;
krig.covariates_data(:,4,:) = krig.prec_mean;
krig.covariates_data(:,5,:) = dtraffic.X_beta{1}(1:4,5,:);
krig.covariates_data(:,6,:) = dtraffic.X_beta{1}(1:4,6,:);
krig.covariates_data(:,7,:) = krig.interstate;
krig.covariates_data(:,8,:) = krig.us;
krig.covariates_data(:,9,:) = krig.rs;
krig.covariates_data(:,10,:) = ones(4,size(dtraffic.Y{1},2));


% obj_stem_krig_grid = stem_grid(krig.coordinates, 'deg', 'sparse', 'point', [], 'square', 0.5, 0.5);
obj_stem_krig_grid = stem_grid(krig.coordinates, 'deg', 'regular', 'point');
obj_stem_krig_data = stem_krig_data(obj_stem_krig_grid, krig.covariates_data, krig.covariates_names);
obj_stem_krig = stem_krig(obj_stem_model1, obj_stem_krig_data);

obj_stem_krig_options = stem_krig_options();
obj_stem_krig_result = obj_stem_krig.kriging(obj_stem_krig_options);
obj_stem_krig_result{1}.plot(1)

clear krig_lat krig_lon

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Statistics                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

visualize_info(obj_stem_model1, dtraffic);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Residual analysis                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

residualsTest = visualize_res(obj_stem_model1, dtraffic);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Estimating model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [dtraffic, krig] = setup_traffic(traffic, krig, freq_seasoned, s_data_drop)
    
    % PARAMETERS TO SET
    if(freq_seasoned > 0)
        seasonality = false;                                        % enable seasonality        
    else
        seasonality = true;                                         % enable seasonality
    end

    d1 = true;                                                      % enable the Seasonal Differencing
    s_data = s_data_drop;                                           % data initial drop 
    nanTo = false;                                                  % switch NaN into traffic to 0  
    
    ns = size(traffic.Y{1}, 1);                                     % number of stations    
    T = size(traffic.Y{1}, 2) - s_data;                             % number of time steps
    s_data = s_data + 1;                                            % gli indici partono da 1 devo aggiungere + 1
    
    if seasonality == true
        if d1 == true
            D1 = LagOp({1 -1},'Lags',[0,1]);                        % Differencing
            Dx = LagOp({1 -1},'Lags',[0, freq_seasoned]);           % lag operator seasonality
            D = D1 * Dx;                                            % total lag operator
            freq_seasoned = freq_seasoned + 1; 
        else
            D = LagOp({1 -1},'Lags',[0, freq_seasoned]);            % lag operator seasonality
        end
    else
        D = LagOp({1 -1},'Lags',[0, 0]);
        freq_seasoned = 0;
    end
    
    
    if nanTo == true
        traffic.Y{1}(isnan(traffic.Y{1})) = 0;
    end
    
    
    traffic_season = ones(ns, T - freq_seasoned);
    for x = 1:ns
        traffic_season(x,:) = filter(D, traffic.Y{1}(x,s_data:end));
    end
    
    
    dtraffic.Y{1} = traffic_season(:,:);
    dtraffic.Y_name{1} = 'traffic';
    dtraffic.weekend{1} = traffic.weekend{1}(s_data + freq_seasoned:end, 1);                                                       
    dtraffic.holiday{1} = traffic.holiday{1}(s_data + freq_seasoned:end, 1);
    dtraffic.mean_prec{1} = traffic.mean_prec{1}(s_data + freq_seasoned:end, 1); 
    dtraffic.mean_temp{1} = traffic.mean_temp{1}(s_data + freq_seasoned:end, 1);
    dtraffic.latitude = traffic.latitude;
    dtraffic.longitude = traffic.longitude;
    dtraffic.route = traffic.route;
    dtraffic.station_id = traffic.station_id;
    dtraffic.time_ini = datetime('01-01-2022 00:00:00','Format','dd-MM-yyyy HH:mm:ss') + hours(s_data - 1) + hours(freq_seasoned);
    dtraffic.time_fin = datetime('31-01-2022 00:00:00','Format','dd-MM-yyyy HH:mm:ss') + hours(23);

    
    dtraffic.dates = [datetime(dtraffic.time_ini):hours(1):datetime(dtraffic.time_fin)];
    traffic_on_full = ones(1,size(dtraffic.Y{1},2));
    traffic_hours = ones(1,size(dtraffic.Y{1},2));
    for x = 1:size(dtraffic.Y{1},2)
        if(ismember(dtraffic.dates(x), traffic.traffic_on_date))
            traffic_on_full(1,x) = 1;
        else
            traffic_on_full(1,x) = 0;
        end

        traffic_hours(1,x) = hour(dtraffic.dates(x));
    end
    dtraffic.traffic_on{1} = traffic_on_full;
    dtraffic.hours{1} = traffic_hours;
    
    Y_mean_seas = zeros(1,size(dtraffic.Y{1},2));
    for x = 1:size(dtraffic.Y{1},2)
        Y_mean_seas(1,x) = mean(dtraffic.Y{1}(:,x), "omitnan");
    end
    dtraffic.Y_mean{1} = Y_mean_seas;
    
    
    % Creating X_beta
    T = size(dtraffic.Y{1}, 2);
    
    X_beta = zeros(ns, 9, T);
    X_beta(:,1,:) = repelem(dtraffic.weekend{1}', ns, 1);
    X_beta(:,2,:) = repelem(dtraffic.holiday{1}', ns, 1);
    X_beta(:,3,:) = repelem(dtraffic.mean_temp{1}', ns, 1);
    X_beta(:,4,:) = repelem(dtraffic.mean_prec{1}', ns, 1);
    X_beta(:,5,:) = repelem(dtraffic.traffic_on{1}, ns, 1);
    X_beta(:,6,:) = repelem(dtraffic.hours{1}, ns, 1);
    X_beta(:,7,:) = repelem(traffic.route_type{1}(:,1), 1, T);
    X_beta(:,8,:) = repelem(traffic.route_type{1}(:,2), 1, T);
    X_beta(:,9,:) = repelem(traffic.route_type{1}(:,3), 1, T);
    
    dtraffic.X_beta{1} = X_beta;  
    dtraffic.X_beta_name{1} = {'weekend', 'holidays', 'mean temp', 'mean prec', 'traffic on', 'hours', 'interstate', 'US', 'RS'};
    
    dtraffic.X_spa = traffic.route_type;
    dtraffic.X_spa_name{1}  = {'interstate', 'US', 'RS'};


    % Kriging covariates
    krig.prec_mean = repelem(krig.prec_mean(1, s_data + freq_seasoned:end), size(krig.coordinates,1), 1);
    krig.temp_mean = repelem(krig.temp_mean(1, s_data + freq_seasoned:end), size(krig.coordinates,1), 1);
    krig.interstate = repelem(krig.route(:,1), 1, T);
    krig.us = repelem(krig.route(:,2), 1, T);
    krig.rs = repelem(krig.route(:,3), 1, T);
    
    
    % Validation - clustering 
    dtraffic.cluster_coordinates{1} = [3 8 9 10];
    dtraffic.cluster_coordinates{2} = [11 18 19 20];
    dtraffic.cluster_coordinates{3} = [21 12 2 14 5 22];
    dtraffic.cluster_coordinates{4} = [6 13 23 24 25 15];
    dtraffic.cluster_coordinates{5} = [4 7 16 26];
    dtraffic.cluster_coordinates{6} = [27 28 29];
    dtraffic.cluster_coordinates{7} = [31 30 33 32];
    dtraffic.cluster_coordinates{8} = [34 36 35 37];
    dtraffic.cluster_coordinates{9} = [39 38 40 42 41];
    dtraffic.cluster_coordinates{10} = [44 43 45 46 47 48 49 50];
    
    figure
    tiledlayout(4,2)
    
    nexttile
    hold on
    plot(traffic.dates, traffic.Y_mean{1});
    title("Y mean traffic");
    
    nexttile
    hold on
    autocorr(traffic.Y_mean{1})
    title("autocorr Y mean traffic");
    
    nexttile
    hold on
    plot(traffic.dates(1,2:end), diff(traffic.Y_mean{1}));
    title("diff Y mean traffic");
    
    nexttile
    hold on
    autocorr(diff(traffic.Y_mean{1}));
    title("Autocorr diff Y mean traffic");
    
    nexttile
    hold on
    plot(dtraffic.dates, dtraffic.Y_mean{1});
    title("Seasonality Y mean traffic");
    
    nexttile
    hold on
    autocorr(dtraffic.Y_mean{1});
    title("Autocorr seasonality Y mean traffic");
    
    nexttile
    hold on
    plot(dtraffic.dates(1,2:end), diff(dtraffic.Y_mean{1}));
    title("Seasonality + diff Y mean traffic");
    
    nexttile
    hold on
    autocorr(diff(dtraffic.Y_mean{1}));
    title("Autocorr seasonality + diff Y mean traffic");
    
    % clear traffic s_data ns T st freq_seasoned D12 traffic_season x Dx D1
    % clear D X_beta Y_mean d1 seasonality nanTo traffic_on traffic_on_full Y_mean_seas
    save('dtraffic.mat');
end




function [dtraffic, obj_stem_model, obj_stem_validation, EM_result] = model_estimate(dtraffic, X_beta, X_beta_name, X_z, X_z_name, X_p, X_p_name, theta_p, v, sigma_eta, G, nIterations)
    
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
    S_val = ones(1,10);
    dtraffic.cluster_choosen = ones(10,2);
    for x = 1:1:size(dtraffic.Y{1},1)*0.2
        j = randi([1 size(dtraffic.cluster_coordinates{x},2)],1,1);
        S_val(1,x) = dtraffic.cluster_coordinates{x}(j);
        dtraffic.cluster_choosen(x,1) = dtraffic.latitude(S_val(1,x),:);
        dtraffic.cluster_choosen(x,2) = dtraffic.longitude(S_val(1,x),:);        
    end
    obj_stem_validation = stem_validation({'traffic'}, {S_val}, 0, {'point'});
    
    obj_stem_data = stem_data(obj_stem_varset_p, obj_stem_gridlist_p, ...
                              [], [], obj_stem_datestamp, obj_stem_validation, obj_stem_modeltype, shape);
    
    % Model creation
    obj_stem_par_constraints = stem_par_constraints();
    obj_stem_par = stem_par(obj_stem_data, 'exponential', obj_stem_par_constraints);
    obj_stem_model = stem_model(obj_stem_data, obj_stem_par);
    
    % Data transform
    % obj_stem_model.stem_data.log_transform;
    obj_stem_model.stem_data.standardize;                   % mean = 0 and std = 1
    
    % Starting values: try different starting values and see if they converge
    % at the same values 
    obj_stem_par.beta = obj_stem_model.get_beta0();
    obj_stem_par.theta_p = theta_p; % poco identificabile
    obj_stem_par.v_p = v;
    obj_stem_par.sigma_eta = sigma_eta;
    obj_stem_par.G = G;
    obj_stem_par.sigma_eps = 0.1;
    obj_stem_model.set_initial_values(obj_stem_par);
    
    % Model estimation
    obj_stem_EM_options = stem_EM_options();
    obj_stem_EM_options.max_iterations = nIterations;
    obj_stem_model.EM_estimate(obj_stem_EM_options);
    obj_stem_model.set_varcov;
    obj_stem_model.set_logL;
    
    % Result of the EM estimation
    EM_result = obj_stem_model.stem_EM_result;

    % mean Y 
    Y_mean = zeros(1,size(dtraffic.Y{1},2));
    for x = 1:size(dtraffic.Y{1},2)
        Y_mean(1,x) = mean(obj_stem_model.stem_data.Y(:,x), "omitnan");
    end
    dtraffic.Y_mean_trans{1} = Y_mean;

    % location of stations 
    figure
    tiledlayout(1,2)
    nexttile
    gs1 = geoscatter(dtraffic.latitude, dtraffic.longitude);
    geobasemap("topographic") 
    geolimits([40 41],[-112 -111.60]) 
    gs1.MarkerFaceColor = [0 0 1];
    title("All stations")
    
    nexttile
    gs2 = geoscatter(dtraffic.cluster_choosen(:,1), dtraffic.cluster_choosen(:,2));
    geobasemap("topographic") 
    geolimits([40 41],[-112 -111.60]) 
    gs2.MarkerFaceColor = [1 0 0];
    title("Cluster selection")
end


function visualize_info(obj_stem_model, dtraffic)

    print(obj_stem_model)

    plot(obj_stem_model.stem_EM_result.stem_kalmansmoother_result)
    title("Latent")

    figure
    tiledlayout(3,2)
    nexttile
    plot(obj_stem_model.stem_EM_result.R2)
    ylim([-1.1, 1.1]);
    title("R2 training")
    
    nexttile
    plot(obj_stem_model.stem_validation_result{1}.cv_R2_s)
    ylim([-1.1, 1.1]);
    title("R2 validation - ns")
    
    nexttile
    plot(dtraffic.dates, sqrt(obj_stem_model.stem_validation_result{1}.cv_mse_t))
    title("CV RMSE - ts")
    
    nexttile
    plot(sqrt(obj_stem_model.stem_validation_result{1}.cv_mse_s))
    title("CV RMSE - ns")
    
    nexttile
    plot(dtraffic.dates, dtraffic.Y_mean{1})
    title("Mean traffic raw")
    
    nexttile
    plot(dtraffic.dates, dtraffic.Y_mean_trans{1})
    title("Mean traffic log")
end


function [residualsTest] = visualize_res(obj_stem_model, dtraffic, residualsTest)
    
    res_mean = zeros(1, size(obj_stem_model.stem_EM_result.res{1},2));
    for x = 1:size(obj_stem_model.stem_EM_result.res{1},2)
        res_mean(1,x) = mean(obj_stem_model.stem_EM_result.res{1}(:,x), "omitnan");
    end

    % lbqtest - H0: series of residuals exhibits no autocorrelation
    % archtest - H0: no conditional heteroscedasticity

    residualsTest.lbqtest{1} = lbqtest(res_mean);
    residualsTest.archtest{1} = archtest(res_mean);
    
    figure
    tiledlayout(1,3)
    nexttile
    plot(dtraffic.dates, res_mean)
    title("Residuals")
    
    nexttile
    histogram(res_mean)
    title("Residuals")
    
    nexttile
    autocorr(res_mean,100)
    title("Residuals")
end