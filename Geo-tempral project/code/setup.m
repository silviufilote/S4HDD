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
% traffic_on = [datetime('2022-01-03 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-03 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-04 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-04 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-05 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-05 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-06 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-06 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-07 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-07 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-10 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-10 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-11 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-11 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-12 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-12 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-13 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-13 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-14 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-14 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-17 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-17 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-18 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-18 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-19 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-19 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-20 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-20 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-21 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-21 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-24 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-24 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-25 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-25 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-26 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-26 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-27 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-27 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-28 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-28 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
%               datetime('2022-01-31 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-31 17:00:00','Format','dd-MM-yyyy HH:mm:ss')];


traffic_on = [datetime('2022-01-01 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-01 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-02 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-02 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-03 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-03 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-04 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-04 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-05 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-05 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-06 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-06 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-07 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-07 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-08 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-08 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-09 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-09 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-10 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-10 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-11 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-11 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-12 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-12 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-13 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-13 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-14 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-14 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-15 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-15 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-16 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-16 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-17 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-17 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-18 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-18 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-19 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-19 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-20 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-20 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-21 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-21 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-22 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-22 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-23 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-23 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-24 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-24 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-25 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-25 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-26 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-26 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-27 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-27 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-28 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-28 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-29 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-29 17:00:00','Format','dd-MM-yyyy HH:mm:ss') datetime('2022-01-30 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-30 17:00:00','Format','dd-MM-yyyy HH:mm:ss')...
              datetime('2022-01-31 7:00:00','Format','dd-MM-yyyy HH:mm:ss'):hours(1):datetime('2022-01-31 17:00:00','Format','dd-MM-yyyy HH:mm:ss')];



traffic.traffic_on_date = traffic_on;
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


Y_mean = zeros(1,size(traffic.Y{1},2));
for x = 1:size(traffic.Y{1},2)
    Y_mean(1,x) = mean(traffic.Y{1}(:,x), "omitnan");
end

traffic.Y_mean{1} = Y_mean;

clear traffic_on_full traffic_on x Y_mean

save('traffic.mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Kriging                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear traffic

load("data\krig_prec.mat")
load("data\krig_route.mat")
load("data\krig_temp.mat")

krig_lat = [40:0.01:41];
krig_lon = [-112.5:0.01:-111.5];

% krig_lat = [40.771379; 40.544919; 40.383103; 40.108343];
% krig_lon = [-112.140558; -111.895082; -111.959112; -111.677759];

[LAT,LON] = meshgrid(krig_lat, krig_lon);
krig.coordinates = [LAT(:) LON(:)];
krig.lat = LAT;
krig.lon = LON;

mean_temp = zeros(1, size(krig_temp,2));
mean_prec = zeros(1, size(krig_temp,2));
for x = 1:size(krig_temp,2)
    mean_temp(x) = mean(krig_temp(:,x));
    mean_prec(x) = mean(krig_prec(:,x));
end

krig.temp_mean = mean_temp;
krig.prec_mean = mean_prec;
krig.route = krig_ruote;

clear krig_lat krig_lon mean_prec mean_temp x
clear krig_temp krig_ruote krig_prec

clear LAT LON
save("krig.mat")

