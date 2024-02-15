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
traffic.time_ini = '2022-01-01 00:00:00';
traffic.time_fin = '2022-01-31 23:00:00';

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

traffic.dates = [datetime('01-01-2022 00:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-31 23:00:00','Format','dd-MM-yyyy HH:mm:ss')];
traffic_on = [datetime('2022-01-03 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-03 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-04 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-04 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-05 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-05 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-06 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-06 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-07 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-07 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-10 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-10 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-11 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-11 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-12 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-12 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-13 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-13 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-14 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-14 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-17 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-17 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-18 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-18 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-19 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-19 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-20 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-20 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-21 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-21 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-24 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-24 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-25 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-25 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-26 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-26 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-27 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-27 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-28 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-28 17:00:00','Format','dd-MM-yyyy HH:mm:ss')];


traffic_on_full = ones(1,size(traffic.Y{1},2));
for x = 1:size(traffic.Y{1},2)
    if(ismember(traffic.dates(x), traffic_on))
        traffic_on_full(1,x) = 1;
    else
        traffic_on_full(1,x) = 0;
    end
end
traffic.traffic_on{1} = traffic_on_full;

% Creating X_beta
X_beta = zeros(ns, 8, T);
X_beta(:,1,:) = repelem(traffic.weekend{1}', ns, 1);
X_beta(:,2,:) = repelem(traffic.holiday{1}', ns, 1);
X_beta(:,3,:) = repelem(traffic.mean_temp{1}', ns, 1);
X_beta(:,4,:) = repelem(traffic.mean_prec{1}', ns, 1);
X_beta(:,5,:) = repelem(traffic.traffic_on{1}, ns, 1);
X_beta(:,6,:) = repelem(utah_meta(:,5), 1, T);
X_beta(:,7,:) = repelem(utah_meta(:,6), 1, T);
X_beta(:,8,:) = repelem(utah_meta(:,7), 1, T);

traffic.X_beta{1} = X_beta;  
traffic.X_beta_name{1} = {'weekend', 'holidays', 'mean temp', 'mean prec', 'traffic on', 'interstate', 'US', 'RS'};

traffic.X_spa = traffic.route_type;
traffic.X_spa_name{1}  = {'interstate', 'US', 'RS'};

clear c m mean_prec mean_temp route_type
clear utah_traffic utah_temp utah_prec utah_meta
clear X_beta ns T

figure
plot(datetime('2022-01-01 00:00:00','Format','dd-MM-yyyy hh:mm:ss'):hours(1):datetime('2022-01-31 23:00:00','Format','dd-MM-yyyy hh:mm:ss'), traffic.Y{1}(5, :))

Y_mean = zeros(1,size(traffic.Y{1},2));
for x = 1:size(traffic.Y{1},2)
    Y_mean(1,x) = mean(traffic.Y{1}(:,x), "omitnan");
end
dtraffic.Y_mean{1} = Y_mean;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     Data cleaning and seasoned                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PARAMETERS TO SET
seasonality = true;                                         % enable seasonality
d1 = false;                                                 % enable the Seasonal Differencing
nanTo = true;                                              % switch NaN into traffic to 0  
s_data = 1;                                                % data initial drop 
freq_seasoned = 7;                                     % seasonality frequency


ns = size(traffic.Y{1}, 1);                                 % number of stations    
T = size(traffic.Y{1}, 2) - s_data + 1;                     % number of time steps - data drop + 1 partenza al 10*


if seasonality == true
    if d1 == true
        D1 = LagOp({1 -1},'Lags',[0,1]);                    % Differencing
        Dx = LagOp({1 -1},'Lags',[0, freq_seasoned]);       % lag operator seasonality
        D = D1 * Dx;                                        % total lag operator
        freq_seasoned = freq_seasoned + 1; 
    else
        D = LagOp({1 -1},'Lags',[0, freq_seasoned]);        % lag operator seasonality
    end
else
    D = LagOp({1 -1},'Lags',[0, 0]);
    freq_seasoned = 0;
end


if nanTo == true
    dtraffic.Y{1}(isnan(dtraffic.Y{1})) = 0;
end


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
dtraffic.time_ini = datetime('01-01-2022 00:00:00','Format','dd-MM-yyyy HH:mm:ss') + hours(s_data - 1);
dtraffic.time_fin = datetime('31-01-2022 00:00:00','Format','dd-MM-yyyy HH:mm:ss') + hours(23) - hours(freq_seasoned);


dtraffic.dates = [datetime(dtraffic.time_ini):hours(1):datetime(dtraffic.time_fin)];
traffic_on_full = ones(1,size(dtraffic.Y{1},2));
for x = 1:size(dtraffic.Y{1},2)
    if(ismember(dtraffic.dates(x), traffic_on))
        traffic_on_full(1,x) = 1;
    else
        traffic_on_full(1,x) = 0;
    end
end
dtraffic.traffic_on{1} = traffic_on_full;

Y_mean_seas = zeros(1,size(dtraffic.Y{1},2));
for x = 1:size(dtraffic.Y{1},2)
    Y_mean_seas(1,x) = mean(dtraffic.Y{1}(:,x), "omitnan");
end
dtraffic.Y_mean_seas{1} = Y_mean_seas;


% Creating X_beta
T = size(dtraffic.Y{1}, 2);

X_beta = zeros(ns, 8, T);
X_beta(:,1,:) = repelem(dtraffic.weekend{1}', ns, 1);
X_beta(:,2,:) = repelem(dtraffic.holiday{1}', ns, 1);
X_beta(:,3,:) = repelem(dtraffic.mean_temp{1}', ns, 1);
X_beta(:,4,:) = repelem(dtraffic.mean_prec{1}', ns, 1);
X_beta(:,5,:) = repelem(dtraffic.traffic_on{1}, ns, 1);
X_beta(:,6,:) = repelem(traffic.route_type{1}(:,1), 1, T);
X_beta(:,7,:) = repelem(traffic.route_type{1}(:,2), 1, T);
X_beta(:,8,:) = repelem(traffic.route_type{1}(:,3), 1, T);

dtraffic.X_beta{1} = X_beta;  
dtraffic.X_beta_name{1} = {'weekend', 'holidays', 'mean temp', 'mean prec', 'traffic on', 'interstate', 'US', 'RS'};

dtraffic.X_spa = traffic.route_type;
dtraffic.X_spa_name{1}  = {'interstate', 'US', 'RS'};


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
tiledlayout(3,2)

nexttile
hold on
plot(dtraffic.Y_mean{1});
title("Y mean traffic");

nexttile
hold on
autocorr(dtraffic.Y_mean{1})
title("autocorr Y mean traffic");

nexttile
hold on
plot(diff(dtraffic.Y_mean{1}));
title("diff Y mean traffic");

nexttile
hold on
autocorr(diff(dtraffic.Y_mean{1}));
title("Autocorr diff Y mean traffic");

nexttile
hold on
plot(dtraffic.Y_mean_seas{1});
title("Seasonality Y mean traffic");

nexttile
hold on
autocorr(dtraffic.Y_mean_seas{1});
title("Autocorr seasonality Y mean traffic");

clear traffic s_data ns T st freq_seasoned D12 traffic_season x D X_beta Y_mean d1 seasonality nanTo traffic_on traffic_on_full
save('dtraffic.mat');
