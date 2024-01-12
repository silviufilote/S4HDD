clc
clearvars

addpath('../D-STEM/Src/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%        Dimensions       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%      Data  building     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

clear c m mean_prec mean_temp route_type
clear utah_traffic utah_temp utah_prec utah_meta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%          Renaming       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

traffic.Y_name{1} = {'traffic'};

% Creating X_beta
X_beta = zeros(ns, 4, T);
X_beta(:,1,:) = repelem(traffic.weekend{1}', ns, 1);
X_beta(:,2,:) = repelem(traffic.holiday{1}', ns, 1);
X_beta(:,3,:) = repelem(traffic.mean_temp{1}', ns, 1);
X_beta(:,4,:) = repelem(traffic.mean_prec{1}', ns, 1);
traffic.X_beta{1} = X_beta;  
traffic.X_beta_name{1} = {'weekend', 'holdays', 'mean temp', 'mean prec'};

traffic.X_p = traffic.route_type;
traffic.X_p_name{1}  = {'interstate','US','RS'};

clear X_beta ns T
save('traffic.mat');
