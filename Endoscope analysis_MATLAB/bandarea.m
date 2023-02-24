function[areacube,areamean] = bandarea(analmat,timecell,inputpoint,conpoint,signflag)
% analmat  = m3t110_mat;
% timecell = m3t110_time;
% timestamp = round(m3t1_0810(2:end,1),1);
% 
% inputpoint = 140;
% conpoint = 593;
%% Trim time data - before cricket input and after catch
for i = 1:length(timecell)
    if ~isempty(timecell{i,1})
        timecell{i,1}(timecell{i,1}(:,1)<inputpoint,:)= [];
        timecell{i,1}(timecell{i,1}(:,2)>conpoint,:)= [];
    else 
        continue
    end
    
end

[~,leng] = size(analmat);

if signflag == 1
    analmat(analmat<0) = 0;
elseif signflag == 0
    analmat(analmat>0) = 0;
else
    error('Signflag must be 1 for positive, 0 for negative');
end
    

onsetact = {};

for i = 1:leng
        tmponset = {};
        % 매번 같은 time period를 돌지 말고, 
        
    for ii = 1:size(timecell{2,1},1)
        tmponset{2,ii} = analmat(timecell{2,1}(ii,2):timecell{2,1}(ii,1),i);
    end
        onsetact{2,i} = tmponset(2,:);
        
    for ii = 1:size(timecell{1,1},1)
        tmponset{1,ii} = analmat(timecell{1,1}(ii,2):timecell{1,1}(ii,1),i);
    end
        onsetact{1,i} = tmponset(1,:);
        
    for ii = 1:size(timecell{3,1},1)
        tmponset{3,ii} = analmat(timecell{3,1}(ii,2):timecell{3,1}(ii,1),i);
    end
        onsetact{3,i} = tmponset(3,:);
        
    for ii = 1:size(timecell{4,1},1)
        tmponset{4,ii} = analmat(timecell{4,1}(ii,2):timecell{4,1}(ii,1),i);
    end
        onsetact{4,i} = tmponset(4,:);     
end

areamean = zeros(3,leng);
for n_cell = 1:leng
    tmp1 = zeros(length(onsetact{2,n_cell}),1);
    for ii = 1:length(onsetact{2,n_cell})
        tmp1(ii) = trapz(abs(cell2mat(onsetact{2,n_cell}(ii))));
    end
    t1 = [tmp1,ones(length(tmp1),1)];
    areamean(1,n_cell) = mean(tmp1);
    
    tmp2 = zeros(length(onsetact{3,n_cell}),1);
    for ii = 1:length(onsetact{3,n_cell})
        tmp2(ii) = trapz(abs(cell2mat(onsetact{3,n_cell}(ii))));
    end
    t2 = [tmp2,2*ones(length(tmp2),1)];
    areamean(2,n_cell) = mean(tmp2);
    
    tmp3 = zeros(length(onsetact{4,n_cell}),1);
    for ii = 1:length(onsetact{4,n_cell})
        tmp3(ii) = trapz(abs(cell2mat(onsetact{4,n_cell}(ii))));
    end
    t3 = [tmp3,3*ones(length(tmp3),1)];
    areamean(3,n_cell) = mean(tmp3);
    
    areacube(:,:,n_cell) = [t1;t2;t3];
end
end
