function [onsetmeancell,onsetsdcell,b4onsetmeancell,b4onsetsdcell] = onsetalign(analmat,timecell,inputpoint,conpoint)
% align onset behavior
% set baseline to before cricket enterence.
% take the vector before cricket and calculate M and SD
% subtract M from target vector, then divide by SD.
% 
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

onsetact = {};
b4onsact = {};

timewindow = 0.5*10;  % amount of frames before onset of the behavior - baseline

for i = 1:leng
        tmponset = {};
        b4onset = {};
        
    for ii = 1:size(timecell{2,1},1)
        tmponset{2,ii} = analmat(timecell{2,1}(ii,2):timecell{2,1}(ii,1),i);
        b4onset{2,ii} = analmat(timecell{2,1}(ii,2)-timewindow:timecell{2,1}(ii,2)-1,i);
    end
        onsetact{2,i} = tmponset(2,:);
        b4onsact{2,i} = b4onset(2,:);
        
    for ii = 1:size(timecell{1,1},1)
        tmponset{1,ii} = analmat(timecell{1,1}(ii,2):timecell{1,1}(ii,1),i);
        b4onset{1,ii} = analmat(timecell{1,1}(ii,2)-timewindow:timecell{1,1}(ii,2)-1,i);
    end
        onsetact{1,i} = tmponset(1,:);
        b4onsact{1,i} = b4onset(1,:);
        
    for ii = 1:size(timecell{3,1},1)

        tmponset{3,ii} = analmat(timecell{3,1}(ii,2):timecell{3,1}(ii,1),i);
        b4onset{3,ii} = analmat(timecell{3,1}(ii,2)-timewindow:timecell{3,1}(ii,2)-1,i);
    end
        onsetact{3,i} = tmponset(3,:);
        b4onsact{3,i} = b4onset(3,:);
        
    for ii = 1:size(timecell{4,1},1)
        tmponset{4,ii} = analmat(timecell{4,1}(ii,2):timecell{4,1}(ii,1),i);
        b4onset{4,ii} = analmat(timecell{4,1}(ii,2)-timewindow:timecell{4,1}(ii,2)-1,i);
    end
        onsetact{4,i} = tmponset(4,:);     
        b4onsact{4,i} = b4onset(4,:);
end
   %% mechanism for onsetalign
   onsetmeancell = cell(4,leng);
   onsetsdcell = cell(4,leng);
   for n_cell = 1:leng
        for n_behav = 1:4
       
        maxLength=max(cellfun(@(x)numel(x),onsetact{n_behav,n_cell}));
        out=cell2mat(cellfun(@(x)cat(1,x,zeros(maxLength-length(x),1)),onsetact{n_behav,n_cell},'UniformOutput',false));
    
        onsetmeancell{n_behav,n_cell} = sum(out,2)./sum(out~=0,2);

%         = out(out~=0))-
        onsetsdcell{n_behav,n_cell} = sqrt(sum(out.^2,2)./sum(out~=0,2)-onsetmeancell{n_behav,n_cell}.^2)./sqrt(sum(out~=0,2));
        end
   end
   b4onsetmeancell = cell(4,leng);
   b4onsetsdcell = cell(4,leng);
   for n_cell = 1:leng
        for n_behav = 1:4
       
    

        maxLength=max(cellfun(@(x)numel(x),b4onsact{n_behav,n_cell}));
        out2=cell2mat(cellfun(@(x)cat(1,x,zeros(maxLength-length(x),1)),b4onsact{n_behav,n_cell},'UniformOutput',false));
    
        b4onsetmeancell{n_behav,n_cell} = sum(out2,2)./sum(out2~=0,2);
%         
        b4onsetsdcell{n_behav,n_cell} = sqrt(sum(out2.^2,2)./sum(out2~=0,2)-b4onsetmeancell{n_behav,n_cell}.^2)./sqrt(sum(out2~=0,2));
        end
   end
   
