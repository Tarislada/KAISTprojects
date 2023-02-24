function [aup,auppt] = aup_behav(behavfile,data,vid_time,pho_time)
% Calculates the area under the photometry curve data
% input argument
% 1) behavfile = behavior file directory without extension. ex) 'C:\Users\endyd\Downloads\180816#15_t2_behavior'
% 2) data = photometry file directory without extionsion ex) 'C:\Users\endyd\Downloads\result_filtered'
% output argument
% a single vector with area under the photometry curve data in the following order
% (1) chasing
% (2) attack
% (3) grooming
% (4) (if it does exists) walking



%% Load and read data 
    raw_behav = readmatrix(behavfile);
    txtid = strcat(data,'.txt');
    fid = fopen(txtid);
    raw_filtered = cell2mat(textscan(fid,'%f'));
    fclose(fid);

    correction = vid_time-pho_time;
    
%% Correct the photometry file to sync with video
    if correction<0
        filtered = raw_filtered(10:end);
    elseif correction>0
        filtered = raw_filtered((correction)*10:end);
    else 
        filtered = raw_filtered;
    end
    
    
    
%% Check for walking behavior
    walkflag = 0;
    aup = zeros(1,5);
    auppt = zeros(1,5);
    if size(raw_behav,2) > 10
        walkflag = 1;
        walk = round(raw_behav(:,11:12)./6);
        walk = reshape(rmmissing(walk),[],2);
        walk(walk>length(filtered)) = length(filtered);
        
        aup = zeros(1,6);
        aup_pt = zeros(1,6);
        aup_walk = zeros(1,size(walk,1));
    end

%% Allocate the behaviors and polish the data for the following process
    
    baseline = round(raw_behav(:,2)./6);


    chasing = round(raw_behav(:,3:4)./6);
    chasing = reshape(rmmissing(chasing),[],2);
    chasing(chasing>length(filtered)) = length(filtered);
    
    attack = round(raw_behav(:,5:6)./6);
    attack = reshape(rmmissing(attack),[],2);
    attack(attack>length(filtered)) = length(filtered);
    
    groom = round(raw_behav(:,9:10)./6);
    groom = reshape(rmmissing(groom),[],2);
    groom(groom>length(filtered)) = length(filtered);
    
    consum = round(raw_behav(:,7:8)./6);
    consum = reshape(rmmissing(consum),[],2);
    consum(consum>length(filtered)) = length(filtered);
    
%% Initialize variables 
    aup_chasing = zeros(1,size(chasing,1));
    aup_attack = zeros(1,size(attack,1));
    aup_groom = zeros(1,size(groom,1));
    aup_consum = zeros(1,size(groom,1));
    aup_base = zeros(1,size(baseline,1));

%% Calculate AUC
    for i = size(chasing,1)
        aup_chasing(i) = trapz(filtered(chasing(i,1):chasing(i,2)));
    end
    aup(1) = sum(aup_chasing);
    auppt(1) = aup(1)/sum(diff(chasing,1,2));
    
    for i = size(attack,1)
        aup_attack(i) = trapz(filtered(attack(i,1):attack(i,2)));
    end
    aup(2) = sum(aup_attack);
    auppt(2) = aup(2)/sum(diff(attack,1,2));

    
    for i = size(groom,1)
        aup_groom(i) = trapz(filtered(groom(i,1):groom(i,2)));
    end
    aup(3) = sum(aup_groom);
    auppt(3) = aup(3)/sum(diff(groom,1,2));


    for i = size(consum,1)
        aup_consum(i) = trapz(filtered(consum(i,1):consum(i,2)));
    end
    aup(4) = sum(aup_consum);
    auppt(4) = aup(4)/sum(diff(consum,1,2));

    
    if walkflag == 1
        for i = size(walk,1)
            aup_walk(i) = trapz(filtered(walk(i,1):walk(i,2)));
        end
        aup(5) = sum(aup_walk);
        auppt(5) = aup(5)/sum(diff(walk,1,2));

    end
    
    for i = size(baseline,1)
        aup_base(i) = trapz(filtered(1:baseline));
    end
    aup(6) = sum(aup_base);
    auppt(6) = aup(6)/length(1:baseline);

    
end        