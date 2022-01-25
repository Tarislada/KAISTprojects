clear;
close all;

Data = varargout('mouse3_t1_trace.csv');%%분석할 trace 이름
tmp = Data(2:end, 2:end);
Data_num = cell2mat(tmp);

time = Data(3:end,1);
data_time = str2double(time);
last = data_time(end,1);

Data_accepted = zeros(size(Data_num));
for i=1:size(Data_num,2)
    if Data_num(1,i) == 1
        Data_accepted(:,i) = Data_num(:,i);
    end
end

Data_accepted(:,all(Data_accepted == 0))=[];
 
%movie frame to time(수동)

%behavior case 별 time 분류 정리


%movie and trace alignment
data_frame=xlsread('m3_t1_frame.xlsx'); %% 분석할 behavior file
frame=data_frame-5; %% trace 와 movie 사이의 시간차이 보정 (end sound video < trace  + / > -)
% case -1 = 1 / case 1 = 2 / case 2 = 3 / case 3 = 4 
times(1).case1 = frame(:,1:2);
times(1).case1 = rmmissing(times(1).case1);
times(1).case2 = frame(:,3:4);
times(1).case2 = rmmissing(times(1).case2);     
times(1).case3 = frame(:,5:6);
times(1).case3 = rmmissing(times(1).case3);
times(1).case4 = frame(:,7:8);
times(1).case4 = rmmissing(times(1).case4);

count = -1;
for ii =  1:size(Data_accepted,2)
    if rem(ii, 10) == 1
        figure ()
        count = count + 1;
    end
    position = ii - (count*10);
    subplot(5,2,position)
    x = data_time;
    plot(x, Data_accepted(2:end,ii))
    xlim([10 150]);%각 data 맞게 수정 필요(보통 10s start / 2:30 end 가 용이)
    xlabel('time')
    ylabel('dF/F')
    hold on
    Top = max(Data_accepted(:,ii));
    Bottom = min(Data_accepted(:,ii));
%(behavior1) blue
for j = 1:length(times(1).case1)
    b1_Start = times(1).case1(j,1);
    b1_End = times(1).case1(j,2); 
    area([b1_Start... 
        b1_End], [Top  Top], ...
        Bottom, 'FaceColor', [0,0,1], 'edgecolor', 'none', 'FaceAlpha', 0.3);    
end
%(behavior2) green
for j = 1:length(times(1).case2)
    b2_Start = times(1).case2(j,1);
    b2_End = times(1).case2(j,2);
    area([b2_Start... 
        b2_End], [Top  Top], ...
        Bottom, 'FaceColor', [0,1,0], 'edgecolor', 'none', 'FaceAlpha', 0.3); 
end
%(behavior3) yellow
for j = 1:length(times(1).case3)
    b3_Start = times(1).case3(j,1);
    b3_End = times(1).case3(j,2);
    area([b3_Start... 
        b3_End], [Top  Top], ...
        Bottom, 'FaceColor', [1,1,0], 'edgecolor', 'none', 'FaceAlpha', 0.3); 
end
%(behavior4) red
for j = 1:length(times(1).case4)
    b4_Start = times(1).case4(j,1);
    b4_End = times(1).case4(j,2);
    area([b4_Start... 
        b4_End], [Top  Top], ...
        Bottom, 'FaceColor', [1,0,0], 'edgecolor', 'none', 'FaceAlpha', 0.3); 
end

end