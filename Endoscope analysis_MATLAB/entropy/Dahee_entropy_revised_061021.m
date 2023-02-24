clc; clear all; close all;

%%Load the files

cd 'F:\entropy\sal'
Files = dir('F:\entropy\sal');
%All_name = [];
All_name = {};
Total_time = [];
for k = 3:length(Files)
    FileNames = Files(k).name;
    temp_table = readtable(FileNames);
    temp_file_name = strcat('sal', num2str(k-2));
%     temp_Xcor_name = strcat('Xcor_sal', num2str(k-2));
%     temp_Ycor_name = strcat('Ycor_sal', num2str(k-2));
%     X_sal = [X_sal ; temp_Xcor_name];
%     Y_sal = [Y_sal ; temp_Ycor_name];
    temp_Xcor = temp_table.XCenter;
    temp_Ycor = temp_table.YCenter;
    temp_Xcor(isnan(temp_Xcor)) = [];
    temp_Ycor(isnan(temp_Ycor)) = [];
    Data.(temp_file_name) = horzcat(temp_Xcor, temp_Ycor);
    %All_name = [All_name; temp_file_name];
    All_name(k-2) = cellstr(temp_file_name);
    Total_time = [Total_time; length(temp_Xcor)];
%     Data.(temp_Xcor_name) = temp_Xcor;
%     Data.(temp_Ycor_name) = temp_Ycor;
    
end
sal_all_name = All_name;

temp_allname_length = length(All_name);
cd 'F:\entropy\cno'
Files = dir('F:\entropy\cno');
for k = 3:length(Files)
    FileNames = Files(k).name;
    temp_table = readtable(FileNames);
    temp_file_name = strcat('cno', num2str(k-2));
%     temp_Xcor_name = strcat('Xcor_cno', num2str(k-2));
%     temp_Ycor_name = strcat('Ycor_cno', num2str(k-2));
%     X_cno = [X_cno ; temp_Xcor_name];
%     Y_cno = [Y_cno ; temp_Ycor_name];
    
    temp_Xcor = temp_table.XCenter;
    temp_Ycor = temp_table.YCenter;
    temp_Xcor(isnan(temp_Xcor)) = [];
    temp_Ycor(isnan(temp_Ycor)) = [];
    Data.(temp_file_name) = horzcat(temp_Xcor, temp_Ycor);
    All_name(temp_allname_length+k-2) = cellstr(temp_file_name);
    Total_time = [Total_time; length(temp_Xcor)];
%     Data.(temp_Xcor_name) = temp_Xcor;
%     Data.(temp_Ycor_name) = temp_Ycor;
end
cno_all_name = All_name(temp_allname_length+1:end);


%wt_time = Data.wt_time;
% cno_x = Data.cno_x;
% cno_y = Data.cno_y;
%cno_time = Data.cno_time;
%r_wt = Data.wt_area.^1/2;
%r_cno = Data.cno_area.^1/2;

%Let's fit together
% cno_x1 = cno_x+median(wt_x);
% cno_y1 = cno_y+median(wt_y);

%Let's make integer coordinates
min_num= [];
max_num = [];
for k = 1:length(All_name)
    %temp_name = All_name(k, :);
    temp_name = char(All_name(k));
    Now.(temp_name) = round(Data.(temp_name), 0);
    min_num = [min_num ; min(Now.(temp_name))];
    max_num = [max_num ; max(Now.(temp_name))];
end

%Let's make grids

grid = min(min_num):1:max(max_num);
% grid_wt_x = min(wt_cor(:, 1)):1:max(wt_cor(:, 1));
% grid_wt_y = min(wt_cor(:, 2)):1:max(wt_cor(:, 2));
% 
% grid_cno_x = min(cno_cor(:, 1)):1:max(cno_cor(:, 1));
% grid_cno_y = min(cno_cor(:, 2)):1:max(cno_cor(:, 2));

%Count how many the points in certain grid point %The size of grid should
%be same for all groups
% for i = 1:length(grid)
%     for j = 1:length(grid)
%         candi = end
%     end
% end
for k = 1:length(All_name)
    temp_name = char(All_name(k));
    for i = 1:length(grid)
        for j = 1:length(grid)
            candi.(temp_name)(i, j) = nnz(sum(Now.(temp_name)==[grid(i), grid(j)], 2)==2);
        end
    end
end

% for i = 1:length(grid_cno_x)
%     for j = 1:length(grid_wt_y)
%         wt_candi(i, j) = nnz(sum([wt_cor==[grid_cno_x(i), grid_wt_y(j)]], 2)==2);
%     end
% end
% 
% for i = 1:length(grid_cno_x)
%     for j = 1:length(grid_wt_y)
%         cno_candi(i, j) = nnz(sum([cno_cor==[grid_cno_x(i), grid_wt_y(j)]], 2)==2);
%     end
% end

% total_len = length(grid)*2;

%Probability
for k = 1:length(All_name)
    temp_name = char(All_name(k));
    candi.(temp_name) = candi.(temp_name)/Total_time(k);
    log_candi.(temp_name) = log2(candi.(temp_name));
    log_candi.(temp_name)(isinf(log_candi.(temp_name))|isnan(log_candi.(temp_name))) = 0;
    
    H.(temp_name) = sum(-sum(candi.(temp_name).*log_candi.(temp_name))); %Entropy
end

% %Log2 of the probability
% wt_log = log2(wt_prob);
% wt_log(isinf(wt_log)|isnan(wt_log)) = 0; %Since log(0) = -inf, I will convert that into zero
% cno_log = log2(cno_prob);
% cno_log(isinf(cno_log)|isnan(cno_log)) = 0;

%Entropy Calculation
% wt_h = -sum(wt_prob.*wt_log);
% wt_h = sum(wt_h);
% cno_h = -sum(cno_prob.*cno_log);
% cno_h = sum(cno_h);

%% Probability Plot
figure(1)
for k = 1:length(sal_all_name)
    temp_name = char(All_name(k));
    subplot(round(sqrt(length(sal_all_name)))+1, round(sqrt(length(sal_all_name))), k);
    heatmap1 = heatmap(flip(transpose(-candi.(temp_name).*log_candi.(temp_name))), 'ColorMap', jet(100));
    title(sprintf('%s', temp_name));
%     caxis(heatmap1, [0 0.5]);
end

figure(2)
for kk = 1:length(cno_all_name)
    temp_name = char(All_name(kk+length(sal_all_name)));
    subplot(round(sqrt(length(cno_all_name)))+1, round(sqrt(length(cno_all_name))), kk);
    heatmap2 = heatmap(flip(transpose(-candi.(temp_name).*log_candi.(temp_name))), 'ColorMap', jet(100));
    title(sprintf('%s', temp_name));
%     caxis(heatmap1, [0 0.5]);
end

% figure(1)
% heatmap1 = heatmap(flip(transpose(-wt_prob.*wt_log)), 'ColorMap', jet(100))
% caxis(heatmap1, [0 0.05]); %You can manage the color range
% 
% figure(2)
% heatmap2 = heatmap(flip(transpose(-cno_prob.*cno_log)), 'ColorMap', jet(100))
% caxis(heatmap2, [0 0.05]);


%Let's fit together
%cno_x1 = cno_x+median(wt_x);
%cno_y1 = cno_y+median(wt_y);

%Polar coordinates 
%[theta_wt, rho_wt] = cart2pol(wt_x, wt_y);
%[theta_cno, rho_cno] = cart2pol(cno_x1, cno_y1);
%wt_xp = r_wt.*cos(theta_wt);
%wt_yp = r_wt.*sin(theta_wt);
%R_wt = theta_wt.*rho_wt;
%R_cno = theta_cno.*rho_cno;

%Calculate the velocities
% for i =1:length(wt_time)-1
%   wt_v(i) = (wt_x(i)^2+wt_y(i)^2)^1/2/(wt_time(i+1)-wt_time(i));
% end
% for i =1:length(cno_time)-1
%   cno_v(i) = (cno_x(i)^2+cno_y(i)^2)^1/2/(cno_time(i+1)-cno_time(i));
% end

