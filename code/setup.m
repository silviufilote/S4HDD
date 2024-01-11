clc
clearvars
addpath('../D-STEM/Src/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%      Data  building     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("data\utah_traffic.mat")
load("data\utah_prec.mat")
load("data\utah_meta.mat")
load("data\utah_temp.mat")

traffic.Y{1} = utah_traffic(:,3:end);
traffic.weekend{1} = utah_traffic(:,1);
traffic.holiday{1} = utah_traffic(:,2);
traffic.prec{1} = utah_prec(:,:);
traffic.temp{1} = utah_temp(:,:);
traffic.latitude = utah_MetaData(:,2);
traffic.longitude = utah_MetaData(:,1);
traffic.route = utah_MetaData(:,3);
traffic.station_id = utah_MetaData(:,4);
traffic.route_type{1} = [utah_MetaData(:,5) utah_MetaData(:,6) utah_MetaData(:,7)];
traffic.time_ini = '01-01-2022 00:00';
traffic.time_fin = '31-01-2022 23:00';

mean_prec = zeros(size(utah_prec,1),1);

for c = 1:size(utah_prec,1)
    m = mean(utah_prec(c,:));
    mean_prec(c) = m;
end

mean_temp = zeros(size(utah_temp,1),1);

for c = 1:size(utah_temp,1)
    m = mean(utah_temp(c,:));
    mean_temp(c) = m;
end

traffic.mean_prec{1} = mean_prec;
traffic.mean_temp{1} = mean_temp;

clear c m mean_prec mean_temp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%          Renaming       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


traffic.X_beta{1} = [traffic.weekend{1} traffic.holiday{1} traffic.mean_temp{1} traffic.mean_prec{1}];
traffic.X_p = traffic.route_type;

save("traffic.mat")
