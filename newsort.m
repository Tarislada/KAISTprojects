function varargout = newsort(workingarray)
% maybe use varargout to output several arrays?
n_input = unique(workingarray);
resultcell = cell(n_input+1);
for ii = 1:n_input+1
    resultcell{ii} = workingarray(:,1)==ii-2;
end


% 
% divind = find(diff(workingarray(:,1))~=0);
% 
%    for i = 1:length(divind)-1
%         if workingarray(divind(i)+1,1) == -1
%             tmpser = workingarray(divind(i)+1:divind(i+1),3);
%             resultcell{1}(end,:) = [max(tmpser) min(tmpser)];
%         elseif workingarray(divind(i)+1,1) == 1
%             tmpser = workingarray(divind(i)+1:divind(i+1),3);
%             resultcell{2}(end,:) = [max(tmpser) min(tmpser)];
%         elseif workingarray(divind(i)+1,1) == 2    
%             tmpser = workingarray(divind(i)+1:divind(i+1),3);
%             resultcell{3}(end,:) = [max(tmpser) min(tmpser)];
%         elseif workingarray(divind(i)+1,1) == 3    
%             tmpser = workingarray(divind(i)+1:divind(i+1),3);
%             resultcell{4}(end,:) = [max(tmpser) min(tmpser)];
%         else 
%             error("Unknown category");
%         end
%     end
    

for ii = 1:nargout
    varargout{ii} = resultcell{ii};
end
