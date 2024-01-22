clc
clearvars
close all

addpath('../D-STEM/Src/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Dimensions                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   {q}     is a cell array of length q
%   q       the number of variables
%   n_i     the number of sites for the i-th variable, i=1,...,q
%   nb_i    the number of loading vectors for the i-th variable related to the beta parameter
%   nz_i    the number of loading vectors for the i-th variable related to the latent variable z
%   nw      the number of latent variables w
%   N       N=n_1+...+n_q total number of observation sites for all the variables
%   T       number of time steps
%   TT      TT = T if the loading coefficients are time-variant and TT=1 if they are time-invariant

% properties
%     Y={};               %[double]   {q}(n_i x T)           observed data
%     X_bp={};            %[double]   {q}(n_i x TT)          loading vectors related to the latent variable w_b
%     X_beta={};          %[double]   {q}(n_i x nb_i x TT)   loading vectors related to the beta parameter
%     X_z={};             %[double]   {q}(n_i x nz_i x TT)   loading vectors related to the latent variable z
%     X_p={};             %[double]   {q}(n_i x nw x TT)     loading vectors related to the latent variable w_p
%     X_h={};             %[double]   {q}(n_i x TT)          domain of the functional observations in Y
%     Y_name={};          %[string]   {q}                    variable names
%     X_bp_name={};       %[string]   {q}                    name of the loading vectors related to the latent variable w_b
%     X_beta_name={};     %[string]   {q}{nb_i}              name of the loading vectors related to the beta parameter
%     X_z_name={};        %[string]   {q}{nz_i}              name of the loading vectors related to the latent variable z
%     X_p_name={};        %[string]   {q}{nw}                name of the loading vectors related to the latent variable w_p
%     X_h_name=[];        %[string]   (1x1)                  name of the domain of the function observations in Y
% 
%     Y_unit={};          %[string]   (1x1)                  unit of variable y
%     X_beta_unit={};     %[string]   (1x1)                  unit of covariates
%     X_h_unit=[];        %[string]   (1x1)                  unit of function domain
%     T_unit=[];          %[string]   (1x1)                  unit of time step
% 
%     simulated=0;        %[boolean]  (1x1)                  1: the Y data are simulated; 0: the Y data are observed data
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            Data  building                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("data\utah_traffic.mat")
load("data\utah_prec.mat")
load("data\utah_meta.mat")
load("data\utah_temp.mat")

traffic.Y{1} = utah_traffic(:,3:end)';              
traffic.weekend{1} = utah_traffic(:,1);                                                         
traffic.holiday{1} = utah_traffic(:,2);
traffic.prec{1} = utah_prec(:,:);
traffic.temp{1} = utah_temp(:,:);
traffic.latitude = utah_meta(:,2);
traffic.longitude = utah_meta(:,1);
traffic.route = utah_meta(:,3);
traffic.station_id = utah_meta(:,4); 
traffic.time_ini = '01-01-2022 00:00';
traffic.time_fin = '31-01-2022 23:00';

% Info to build matrices
ns = size(traffic.Y{1}, 1); % number of stations
T = size(traffic.Y{1}, 2);  % time-step

% Precipitation 
mean_prec = zeros(size(utah_prec,1),1);

for c = 1:size(utah_prec,1)
    m = mean(utah_prec(c,:));
    mean_prec(c) = m;
end

% Temperature
mean_temp = zeros(size(utah_temp,1),1);

for c = 1:size(utah_temp,1)
    m = mean(utah_temp(c,:));
    mean_temp(c) = m;
end

traffic.mean_prec{1} = mean_prec;
traffic.mean_temp{1} = mean_temp;

% Traffic routes variables
route_type = zeros(ns,3);
route_type(:,1) = utah_meta(:,5);
route_type(:,2) = utah_meta(:,6);
route_type(:,3) = utah_meta(:,7);
traffic.route_type{1} = route_type;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Renaming                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

traffic.Y_name{1} = 'traffic';

% Creating X_beta
X_beta = zeros(ns, 7, T);
X_beta(:,1,:) = repelem(traffic.weekend{1}', ns, 1);
X_beta(:,2,:) = repelem(traffic.holiday{1}', ns, 1);
X_beta(:,3,:) = repelem(traffic.mean_temp{1}', ns, 1);
X_beta(:,4,:) = repelem(traffic.mean_prec{1}', ns, 1);
X_beta(:,5,:) = repelem(utah_meta(:,5), 1, T);
X_beta(:,6,:) = repelem(utah_meta(:,6), 1, T);
X_beta(:,7,:) = repelem(utah_meta(:,7), 1, T);

traffic.X_beta{1} = X_beta;  
traffic.X_beta_name{1} = {'weekend', 'holidays', 'mean temp', 'mean prec', 'interstate', 'US', 'RS'};

traffic.X_spa = traffic.route_type;
traffic.X_spa_name{1}  = {'interstate', 'US', 'RS'};

clear c m mean_prec mean_temp route_type
clear utah_traffic utah_temp utah_prec utah_meta
clear X_beta ns T

save('traffic.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Operation on the Y                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Variables
ns = size(traffic.Y{1}, 1);                         % number of stations    
T = size(traffic.Y{1}, 2);                          % number of time steps
st = 3;                                             % station considered

% Stationary process?
% Differencing order 1 -> delete the trend 

% adftest       H0: the time series contains a unit root and is non-stationary 
% kpsstest      H0: that the time series is stationary
% vratiotest    H0: no random walks constant variance 
% lbqtest       H0: no residual autocorrelation

% [h1,pValue1] = adftest(diff(traffic.Y{1}(3,:)));                % unit root? 1 - trend stationary
% [h2,pValue2] = kpsstest(traffic.Y{1}(3,:));                     % unit root? 0 - trend stationary
% [h3,pValue3] = vratiotest(traffic.Y{1}(3,:));                   % random walks? heteroschedasticity


tiledlayout(3,4)

nexttile
hold on
plot(traffic.Y{1}(st,:));
title(['Time series ',num2str(st),'rd station']);

nexttile
hold on
autocorr(traffic.Y{1}(st,:),40);
title(['Autocorrelation ',num2str(st),'rd station']);

nexttile
hold on
plot(diff(traffic.Y{1}(st,:),2));
title(['Diff ',num2str(st),'rd station']);

nexttile
hold on
autocorr(diff(traffic.Y{1}(st,:),2), 40);
title(['Diff ',num2str(st),'rd autocorrelation']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     Data cleaning and seasoned                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% removing first 10 data -> more consistent
% Differencing: stabilize mean, seasonality and trend

s_data = 10;                                        % data initial drop 
ns = size(traffic.Y{1}, 1);                         % number of stations    
T = size(traffic.Y{1}, 2) - s_data + 1;             % number of time steps - data drop + 1 partenza al 10*
freq_seasoned = 24;                                 % frequency seasonality
D = LagOp({1 -1},'Lags',[0, freq_seasoned]);        % lag operator seasonality


traffic_season = ones(ns, T - freq_seasoned);
for x = 1:ns
    traffic_season(x,:) = filter(D, traffic.Y{1}(x,s_data:end));
end



dtraffic.Y{1} = traffic_season(:,:);
dtraffic.Y_name{1} = 'traffic';
dtraffic.weekend{1} = traffic.weekend{1}(s_data:end - freq_seasoned,1);                                                       
dtraffic.holiday{1} = traffic.holiday{1}(s_data:end - freq_seasoned,1);
dtraffic.mean_prec{1} = traffic.mean_prec{1}(s_data:end - freq_seasoned,1); 
dtraffic.mean_temp{1} = traffic.mean_temp{1}(s_data:end - freq_seasoned,1);
dtraffic.latitude = traffic.latitude;
dtraffic.longitude = traffic.longitude;
dtraffic.route = traffic.route;
dtraffic.station_id = traffic.station_id;
dtraffic.time_ini = '01-01-2022 9:00';
dtraffic.time_fin = '30-01-2022 23:00';


% Creating X_beta
T = size(dtraffic.Y{1}, 2);

X_beta = zeros(ns, 7, T);
X_beta(:,1,:) = repelem(dtraffic.weekend{1}', ns, 1);
X_beta(:,2,:) = repelem(dtraffic.holiday{1}', ns, 1);
X_beta(:,3,:) = repelem(dtraffic.mean_temp{1}', ns, 1);
X_beta(:,4,:) = repelem(dtraffic.mean_prec{1}', ns, 1);
X_beta(:,5,:) = repelem(traffic.route_type{1}(:,1), 1, T);
X_beta(:,6,:) = repelem(traffic.route_type{1}(:,2), 1, T);
X_beta(:,7,:) = repelem(traffic.route_type{1}(:,3), 1, T);

dtraffic.X_beta{1} = X_beta;  
dtraffic.X_beta_name{1} = {'weekend', 'holidays', 'mean temp', 'mean prec', 'interstate', 'US', 'RS'};

dtraffic.X_spa = traffic.route_type;
dtraffic.X_spa_name{1}  = {'interstate', 'US', 'RS'};

nexttile
hold on
plot(dtraffic.Y{1}(st,:));
title(['Seasoned/cleaned ',num2str(st),'rd station time series'])

nexttile
hold on
autocorr(dtraffic.Y{1}(st,:), 40);
title(['Seasoned/cleaned ',num2str(st),'rd station autocorrelation'])

nexttile
hold on
plot(diff(dtraffic.Y{1}(st,:)));
title(['Seasoned/cleaned ',num2str(st),'rd station + diff time series'])

nexttile
hold on
autocorr(diff(dtraffic.Y{1}(st,:)), 40);
title(['Seasoned/cleaned ',num2str(st),'rd station + diff autocorrelation'])

clear traffic s_data ns T st freq_seasoned D12 traffic_season x D X_beta
save('dtraffic.mat');
% clear('dtraffic.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                kriging                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

krig_lat = [40.771278; 40.771018; 40.771538];
krig_lon = [-112.144589; -112.133259; -112.088627];
krig.coordinates = [krig_lat(:) krig_lon(:)];

% generate covariates data
krig.coordinates_data = zeros(3, 7, size(dtraffic.Y{1},2));
krig.coordinates_name = dtraffic.X_beta_name;

obj_stem_krig_grid = stem_grid(krig.coordinates, 'deg', 'sparse', 'point', [], 'square', 0.5, 0.5);
obj_stem_krig_data = stem_krig_data(obj_stem_krig_grid, krig.covariates_data, krig_covariates.names,[]);
obj_stem_krig = stem_krig(obj_stem_model,obj_stem_krig_data);

obj_stem_krig_options = stem_krig_options();
obj_stem_krig_options.block_size = 500;

obj_stem_krig_result = obj_stem_krig.kriging(obj_stem_krig_options);
obj_stem_krig_result{1}.plot(1)


% location of stations 
gs = geoscatter(dtraffic.latitude, dtraffic.longitude);
geobasemap("topographic") 
geolimits([40 41],[-112 -111.60]) 
gs.MarkerFaceColor = [0, 0.270, 0.2410]

kriging