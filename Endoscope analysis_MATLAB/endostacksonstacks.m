function [comstackm,comstacksd,comstackn] = endostacksonstacks(totalstackmean,totalstacksd,totalstackn)
% stacking all trials 
% mean fluoresence signal response to specific behavior in a mouse


% [stackmean_m6t1,stacksd_m6t1,stackn_m6t1] = endostacks(timecell_m6t1,inccell_m6t1,normonset_m6t1);
% [stackmean_m6t2,stacksd_m6t2,stackn_m6t2] = endostacks(timecell_m6t2,inccell_m6t2,normonset_m6t2);
% [stackmean_m6t3,stacksd_m6t3,stackn_m6t3] = endostacks(timecell_m6t3,inccell_m6t3,normonset_m6t3);
% [stackmean_m7t1,stacksd_m7t1,stackn_m7t1] = endostacks(timecell_m7t1,inccell_m7t1,normonset_m7t1);
% [stackmean_m7t2,stacksd_m7t2,stackn_m7t2] = endostacks(timecell_m7t2,inccell_m7t2,normonset_m7t2);
% [stackmean_m7t3,stacksd_m7t3,stackn_m7t3] = endostacks(timecell_m7t3,inccell_m7t3,normonset_m7t3);
% stackmean_m6t1{5} = [];
% stacksd_m6t1{5} = [];
% stackn_m6t1{5} = [];
% totalstackmean = [stackmean_m6t1;stackmean_m6t2;stackmean_m6t3;stackmean_m7t1;stackmean_m7t2;stackmean_m7t3];
% totalstacksd = [stacksd_m6t1;stacksd_m6t2;stacksd_m6t3;stacksd_m7t1;stacksd_m7t2;stacksd_m7t3];
% totalstackn = [stackn_m6t1;stackn_m6t2;stackn_m6t3;stackn_m7t1;stackn_m7t2;stackn_m7t3];

%% initialize variables
[~,behavn]=size(totalstackmean);
comstackm = cell(1,behavn);
comstacksd = cell(1,behavn);
comstackn = cell(1,behavn);

%% Calculate stacked mean, sem, n

for n_behav = 1:behavn   % for all behaviors
    meanmat = nan(max(cellfun(@length,totalstackmean(:,n_behav))),behavn); % initialize mean matrix with nan for easier calculations
    sdmat = nan(max(cellfun(@length,totalstackmean(:,n_behav))),behavn); 
    nmat = nan(max(cellfun(@length,totalstackmean(:,n_behav))),behavn);
    for ii = 1:length(totalstackmean(:,n_behav)) % for each stack, 
         meanmat(1:length(totalstackmean{ii,n_behav}),ii) = totalstackmean{ii,n_behav}; % collect means 
         sdmat(1:length(totalstackmean{ii,n_behav}),ii) = totalstacksd{ii,n_behav}; % sd
         nmat(1:length(totalstackmean{ii,n_behav}),ii) = totalstackn{ii,n_behav};   % and n
    end
    comstackm{n_behav} = combm(meanmat,nmat);   % then calculate combined mean
    comstacksd{n_behav} = combsd(meanmat,nmat,sdmat);   % sd
    comstackn{n_behav} = sum(nmat,2,'omitnan'); % and nb
end


function m = combm(mmat,nmat)   % custom function for calcuation of combinedmean
m = sum(mmat.*nmat,2,'omitnan')./sum(nmat,2,'omitnan');
end

function sd = combsd(mmat,nmat,sdmat)% custom function for calcuation of combinedsd
varmat = sdmat.^2;
m = sum(mmat.*nmat,2,'omitnan')./sum(nmat,2,'omitnan');
cvar = (sum((nmat-1).*varmat,2,'omitnan')+sum(nmat.*((mmat-m).^2),2,'omitnan'))./(sum(nmat,2,'omitnan')-1);
sd = sqrt(cvar);
end



% g(7) = endostacks(timecell_m6t1,deccell_m6t1,normonset_m6t1);
% sgtitle('M6t1 Decrease cells stacked')
% g(8) = endostacks(timecell_m6t2,deccell_m6t2,normonset_m6t2);
% sgtitle('M6t2 Decrease cells stacked')
% g(9) = endostacks(timecell_m6t3,deccell_m6t3,normonset_m6t3);
% sgtitle('M6t3 Decrease cells stacked')
% g(10) = endostacks(timecell_m7t1,deccell_m7t1,normonset_m7t1);
% sgtitle('M7t1 Decrease cells stacked')
% g(11) = endostacks(timecell_m7t2,deccell_m7t2,normonset_m7t2);
% sgtitle('M7t2 Decrease cells stacked')
% g(12) = endostacks(timecell_m7t3,deccell_m7t3,normonset_m7t3);
% sgtitle('M7t3 Decrease cells stacked')
end
