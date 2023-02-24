function [normonset,tinccell,tdeccell]=walltaskanalysis(dn_array,timecell)
% function [normonset,tidcell]=walltaskanalysis(dn_array,timecell)
% dn_array = dnsigcell{1};
% timecell = timecellcell{1};
leng = size(dn_array,2);
behavn = length(timecell);
binaryflag = 0;
if behavn == 3
    binaryflag = 1;
end


%% onset align
analmat = dn_array;
fixedact = {};
onsetact = {};
b4onsact = {};

timewindow = 2*10;  % amount of frames before onset of the behavior - baseline

for i = 1:size(analmat,2) % without detecting signals with events
        tmponset = {};
        b4onset = {};
    for ii = 1:size(timecell{2,1},1)
        tmponset{2,ii} = analmat(timecell{2,1}(ii,2):timecell{2,1}(ii,1),i);
        b4onset{2,ii} = analmat(timecell{2,1}(ii,2)-timewindow:timecell{2,1}(ii,2)-1,i);
    end
        onsetact{2,i} = tmponset(2,:);
        b4onsact{2,i} = b4onset(2,:);
        
    for ii = 1:size(timecell{3,1},1)
        tmponset{3,ii} = analmat(timecell{3,1}(ii,2):timecell{3,1}(ii,1),i);
        b4onset{3,ii} = analmat(timecell{3,1}(ii,2)-timewindow:timecell{3,1}(ii,2)-1,i);
    end
        onsetact{3,i} = tmponset(3,:);
        b4onsact{3,i} = b4onset(3,:);
          
        
    if binaryflag == 0
        for ii = 1:size(timecell{4,1},1)
            tmponset{4,ii} = analmat(timecell{4,1}(ii,2):timecell{4,1}(ii,1),i);
            b4onset{4,ii} = analmat(timecell{4,1}(ii,2)-timewindow:timecell{4,1}(ii,2)-1,i);
        end
            onsetact{4,i} = tmponset(4,:);     
            b4onsact{4,i} = b4onset(4,:);
    end
    
%     
%         
%     effective2 = find(actstrip{2,i}); % find if there is any non-zero event rate cell to reduce time consumption - pursuit
%     if numel(effective2)~=0
%         for ii = 1:numel(effective2)
%             tmponset{2,ii} = analmat(timecell{2,1}(effective2(ii),2):timecell{2,1}(effective2(ii),1),i); % find non-zero event rate behavior reps 
%             b4onset{2,ii} = analmat(timecell{2,1}(effective2(ii),2)-timewindow:timecell{2,1}(effective2(ii),2)-1,i);
%         end
%         onsetact{2,i} = tmponset(2,:);
%         b4onsact{2,i} = b4onset(2,:);
%     else 
%         onsetact{2,i} = {};
%         b4onsact{2,i} = {};
%     end
%     
%     effective3 = find(actstrip{3,i}); %forelimb
%     if numel(effective3)~=0
%         for ii = 1:numel(effective3)
%             tmponset{3,ii} = analmat(timecell{3,1}(effective3(ii),2):timecell{3,1}(effective3(ii),1),i);
%             b4onset{3,ii} = analmat(timecell{3,1}(effective3(ii),2)-timewindow:timecell{3,1}(effective3(ii),2)-1,i);
%         end
%         onsetact{3,i} = tmponset(3,:);
%         b4onsact{3,i} = b4onset(3,:);
%     else 
%         onsetact{3,i} = {};
%         b4onsact{3,i} = {};
%     end
%     
%     effective4 = find(actstrip{4,i}); % jaw
%     if numel(effective4)~=0
%         for ii = 1:numel(effective4)
%             tmponset{4,ii} = analmat(timecell{4,1}(effective4(ii),2):timecell{4,1}(effective4(ii),1),i);
%             b4onset{4,ii} = analmat(timecell{4,1}(effective4(ii),2)-timewindow:timecell{4,1}(effective4(ii),2)-1,i);
%         end
%         onsetact{4,i} = tmponset(4,:);     
%         b4onsact{4,i} = b4onset(4,:);
%     else 
%         onsetact{4,i} = {};
%         b4onsact{4,i} = {};
%     end
%     
% %        
% %     effective5 = find(actstrip{5,i}); %consume
% %     if numel(effective5)~=0
% %         for ii = 1:numel(effective5)
% %             tmponset{5,ii} = analmat(timecell{5,1}(effective5(ii),2):timecell{5,1}(effective5(ii),1),i);
% %             b4onset{5,ii} = analmat(timecell{5,1}(effective5(ii),2)-timewindow:timecell{5,1}(effective5(ii),2)-1,i);
% %         end
% %         onsetact{5,i} = tmponset(5,:);     
% %         b4onsact{5,i} = b4onset(5,:);
% %     else 
% %         onsetact{5,i} = {};
% %         b4onsact{5,i} = {};
% %     end
%     
% if searchflag ==1       % search
% %     for ii = 1:size(timecell{6,1},1)
% %         tmponset{6,ii} = analmat(timecell{6,1}(ii,2):timecell{6,1}(ii,1),i);
% %         b4onset{6,ii} = analmat(timecell{6,1}(ii,2)-timewindow:timecell{6,1}(ii,2)-1,i);
% %     end
% %         onsetact{6,i} = tmponset(6,:);
% %         b4onsact{6,i} = b4onset(6,:);
% %     
%     effective5 = find(actstrip{5,i});
%     if numel(effective5)~=0
%         for ii = 1:numel(effective5)
%             tmponset{5,ii} = analmat(timecell{5,1}(effective5(ii),2):timecell{5,1}(effective5(ii),1),i);
%             b4onset{5,ii} = analmat(timecell{5,1}(effective5(ii),2)-timewindow:timecell{5,1}(effective5(ii),2)-1,i);
%         end
%         onsetact{5,i} = tmponset(5,:);     
%         b4onsact{5,i} = b4onset(5,:);
%     else 
%         onsetact{5,i} = {};
%         b4onsact{5,i} = {};
%     end
% end    
end


%% normalize b4onset
newb4onset = cell(size(b4onsact));  % initialize variable containing normalized before onset aligned neural data
newonset = cell(size(onsetact));    % initialize variable containing normalized onset aligned neural data
normonset = cell(size(onsetact));   % initialize variable containing normalized aligned neural data
for n_cell = 1:leng     % for all cells
    for n_behav = 1:behavn  % for all behaviors
        if ~isempty(b4onsact{n_behav,n_cell})   % is b4onset is not empty,
            tmpb4ons = cellfun(@(x) mean(x), b4onsact{n_behav,n_cell});     % before onset means
            tmpb4sd = cellfun(@(x) std(x), b4onsact{n_behav,n_cell});       % and std
            newb4onset{n_behav,n_cell} = cellfun (@minus, b4onsact{n_behav,n_cell}, num2cell(tmpb4ons), 'UniformOutput', false);    % normalize with before onset means 
            newb4onset{n_behav,n_cell} = cellfun (@mrdivide, newb4onset{n_behav,n_cell}, num2cell(tmpb4sd), 'UniformOutput', false);    % and std
            newonset{n_behav,n_cell} = cellfun (@minus, onsetact{n_behav,n_cell}, num2cell(tmpb4ons), 'UniformOutput', false);      % normalize after onset signals with before onset
            newonset{n_behav,n_cell} = cellfun (@mrdivide, newonset{n_behav,n_cell}, num2cell(tmpb4sd), 'UniformOutput', false);
            tmpcell = cellfun (@vertcat, newb4onset{n_behav,n_cell},newonset{n_behav,n_cell},'UniformOutput',false);    % normalized peri-event histogram
            tmpraw = cellfun (@vertcat, b4onsact{n_behav,n_cell},onsetact{n_behav,n_cell},'UniformOutput',false);       % raw peri-event histogram
            idx = ~cellfun('isempty',tmpcell);
            normonset{n_behav,n_cell}(idx) = tmpcell(idx);      % normalized whole onset aligned signal
            rawonset{n_behav,n_cell}(idx) = tmpraw(idx);        % raw whole onset aligned signal
        end
    end
end

%% Onsetalign plot variables 
   onsetmeancell = cell(behavn,leng);   % initialize variables for onset-aligned mean
   onsetsdcell = cell(behavn,leng);     % standard error 
   onsetncell = cell(behavn,leng);      % and n
   for n_cell = 1:leng                  % for all cells 
        for n_behav = 1:behavn          % and behavior
        if ~isempty(newonset{n_behav,n_cell})   % if normalized onset-aligned peth is not empty,
            maxLength=max(cellfun(@(x)numel(x),newonset{n_behav,n_cell}));  
            out=cell2mat(cellfun(@(x)cat(1,x,nan(maxLength-length(x),1)),newonset{n_behav,n_cell},'UniformOutput',false)); %nan padding for easier means and sem calculation
            onsetmeancell{n_behav,n_cell} = nanmean(out,2);         % mean onset-aligned signal
            onsetsdcell{n_behav,n_cell} = nanstd(out,0,2)./ sqrt(sum(~isnan(out),2));   % stnadard error of means
            onsetncell{n_behav,n_cell} = sum(~isnan(out),2);        % n
        end
        
        end
   end
   % same for before-onset 
   b4onsetmeancell = cell(behavn,leng); 
   b4onsetsdcell = cell(behavn,leng);
   b4onsetncell = cell(behavn,leng);
   for n_cell = 1:leng
        for n_behav = 1:behavn
        if ~isempty(newb4onset{n_behav,n_cell}) 
            maxLength=max(cellfun(@(x)numel(x),newb4onset{n_behav,n_cell}));
            out2=cell2mat(cellfun(@(x)cat(1,x,nan(maxLength-length(x),1)),newb4onset{n_behav,n_cell},'UniformOutput',false));
            b4onsetmeancell{n_behav,n_cell} = nanmean(out2,2);
            b4onsetsdcell{n_behav,n_cell} = nanstd(out2,0,2)./ sqrt(sum(~isnan(out2),2));
            b4onsetncell{n_behav,n_cell} = sum(~isnan(out2),2);
        end
        
        end
   end
   
   
%% Cell identity check


% initiallize variables containing cell specificity and
% increasing/decreasing activity cells
tidcell = cell(behavn,1);
tinccell = cell(behavn,1);
tdeccell = cell(behavn,1);

   tcheck = nan(behavn,leng);
   tpval = nan(behavn,leng);
   for n_cell = 1:leng
       for n_behav = 1:behavn
           if ~isempty(newb4onset{n_behav,n_cell}) 
               [tcheck(n_behav,n_cell),tpval(n_behav,n_cell)]=ttest2(cell2mat(newonset{n_behav,n_cell}(:)),cell2mat(newb4onset{n_behav,n_cell}(:)),'Alpha',1-(1-0.05)^(1/sum(~cellfun(@isempty,newb4onset(:)))));
           end
       end
       
   end
%    tid = cell(5,1);
%    tinccell = cell(5,1);
%    tdeccell = cell(5,1);
   inccheck = cellfun(@mean,onsetmeancell)>cellfun(@mean,b4onsetmeancell);
   deccheck = cellfun(@mean,onsetmeancell)<cellfun(@mean,b4onsetmeancell);
       
   for ii = 1:behavn-1
       tid{ii} = find(tcheck(ii+1,:)==1);
       tinccell{ii} = find(tcheck(ii+1,:)==1 & inccheck(ii+1,:)==1);
       tdeccell{ii} = find(tcheck(ii+1,:)==1 & deccheck(ii+1,:)==1);       
   end
   
