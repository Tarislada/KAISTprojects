function [indcell, cmat, idmat] = peakcat(time1,countmat,inputpoint,conpoint)
% time1 = m3t110_time;
% countmat = m3t110_count;
% inputpoint = 140;
% conpoint = 593;

%% Trimming data - before cricket enterence and consummation
for i = 1:length(time1)
    time1{i,1}(time1{i,1}(:,1)<inputpoint,:)= [];
    time1{i,1}(time1{i,1}(:,2)>conpoint,:)= [];
end

%% Calculate the area of each behavior band - expected probability
timesum = zeros(1,3);
for ii = 1:3
%     timesum(ii) = sum(time1{ii+1,1}(:,1) - time1{ii+1,1}(:,2));
     timesum(ii) = abs(sum(diff(time1{ii+1,1},1,2)));
end
expectedprob = [timesum(1)/sum(timesum) timesum(2)/sum(timesum) timesum(3)/sum(timesum)];

%% Run multinomial exact test - calculate p value and take results
pvalvec = zeros(1,length(countmat));
for n_cell  = 1:length(countmat)
    [~,~,pval]=multinomials(countmat(2:4,n_cell)',expectedprob);
    pvalvec(n_cell) = pval;
end
idmat = zeros(1,length(countmat));
Maxmat= -1*ones(1,length(countmat));

[M,id] = max(countmat(2:4,pvalvec<0.05));
idmat(pvalvec<0.05) = id;
Maxmat(pvalvec<0.05) = M;

tmpmat = [Maxmat; idmat];
checkmat = tmpmat(:,Maxmat(1,:)>1);

%% Count how many cells show significance for each behavior
cmat = zeros(3,1);
cmat(1) = sum(checkmat(2,:)==1);
cmat(2) = sum(checkmat(2,:)==2);
cmat(3) = sum(checkmat(2,:)==3);

%% Find the significant cells
indcell = cell(3,1);
ttmp = tmpmat;
ttmp(:,ttmp(1,:)<2) = -1;

indcell{1} = find(ttmp(2,:)==1);
indcell{2} = find(ttmp(2,:)==2);
indcell{3} = find(ttmp(2,:)==3);
