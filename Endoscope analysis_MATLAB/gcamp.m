clear all;
close all;

Dir = pwd;
ascfile=dir([Dir,'\*.asc']);
numFile=length(ascfile);
sampling_rate=10;
numdatalength=2048;%% 3 blcok = 6144, 9 block = 18432

%% raw data 읽어 들이기.
for z=1:numFile
    filename=ascfile(z,1).name;
    fid=fopen(filename);
    for i=1:10
        buffer=fgetl(fid);
    end
    raw_data=textscan(fid, '%d %d');
    photon(:,1)=raw_data{1};
    numblock=length(photon)/1024;
    for n=1:numblock
        block(:,n)=photon(1024*(n-1)+1:1024*n,1); %block 행렬에 각 block 데이터를 저장
        F(n,z)=sum(block(:,n));
    end
    F0(:,z)=mean(mean(F(:,z)));   
end
for ii=1:numFile
    total_photon(numblock*(ii-1)+1:numblock*ii,1)=double(F(:,ii));
    result(numblock*(ii-1)+1:numblock*ii,1)=(double(F(:,ii))-F0(:,ii))/F0(:,ii);
end

%% TSPCS 검증
photon_count_time=linspace(0, 16, 1024);
photon_count_time=photon_count_time';
photon_count(:,1)=mean(block,2);

%% smoothing
GCaMP_smooth=smooth(result, 5);
result = GCaMP_smooth;


%% threshold search
mean_result=mean(result);
std_result=std(result);
threshold=mean_result+2.91*std_result;
% [r,c]=find(result>=threshold); %transient search가 on이면 이게 off vice versa


%% transient search_GCaMP activity
baseline_window=100;
for iii=1:length(result)-baseline_window
    mu=mean(result(iii:iii+baseline_window-1, 1));
    sigma=std(result(iii:iii+baseline_window-1,1));
    istransient=result(iii+baseline_window,1)-(mu+3*sigma);
    if istransient>=0
        result(iii+baseline_window,2)=true;
    end
end
[r,c]=find(result(:,2)==true);

%% spike sorting of GCaMP activity (result)
spike_times=[r];
diff_spike=spike_times(2:end, 1)-spike_times(1:end-1,1);
sorting_address=find(diff_spike>5); % end of spikes , 얼마나 떨어져 있어야 spike 분리 할 것인지 설정.
start_of_spikes=spike_times(sorting_address+1);
start_of_spikes(2:length(start_of_spikes)+1,1)=start_of_spikes;
start_of_spikes(1,1)=min(spike_times);
end_of_spikes=spike_times(sorting_address);

for p=1:length(end_of_spikes)
    peak(p,1)=max(result(start_of_spikes(p,1):end_of_spikes(p,1),1)); 
    peak_times(p,1)=find(result(start_of_spikes(p,1):end_of_spikes(p,1),1)==peak(p,1))+start_of_spikes(p,1)-1;
end
figure();
plot(result(:,1));


%% statistics
%%openExample('stats/PairedSampletTestExample')

%%x1=result(167:178,1);
%%y1=result(178:188,1);
%%[h,p]=ttest(x1,y1)
