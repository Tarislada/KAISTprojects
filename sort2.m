% sort file dealing with different version of the input data
function [array1, array2, array3, array4] = sort2(workingarray)
% This code will seperate accodring to first column and extract time stamp
% data. CAUTION: could be extravagent. WILL NOT FUCTION IF CATEGORY EXCEEDS
% A RANGE OF -1~4. Possible ways of improving this shameful code would be
% using other data structures, such as a cell or a struct
%% initialize
    divind = find(diff(workingarray(:,1))~=0);
    % divind is the end of a series. so, each element+1 is the start of new
    % series, and next element is the end.
    array1 = [];    % case: -1
    array2 = [];    % case: 0
    array3 = [];    % case: 1
    array4 = [];    % case: 2

%% Main seperation
    for i = 1:length(divind)-1
        if workingarray(divind(i)+1,1) == -1
            tmpser = workingarray(divind(i)+1:divind(i+1),3);
            array1 = [array1; max(tmpser) min(tmpser)];
        elseif workingarray(divind(i)+1,1) == 1
            tmpser = workingarray(divind(i)+1:divind(i+1),3);
            array2 = [array2; max(tmpser) min(tmpser)];
        elseif workingarray(divind(i)+1,1) == 2    
            tmpser = workingarray(divind(i)+1:divind(i+1),3);
            array3 = [array3; max(tmpser) min(tmpser)];
        elseif workingarray(divind(i)+1,1) == 3    
            tmpser = workingarray(divind(i)+1:divind(i+1),3);
            array4 = [array4; max(tmpser) min(tmpser)];
        else 
            error("Unknown category");
        end
    end
    
%% Dealing with remaining cases - when starting index is 1 & ending index
    if divind(1) == 1
        if workingarray(divind(1)) == -1
            array1 = [workingarray(1,3) workingarray(1,3); array1];
        elseif workingarray(divind(1)) == 1
            array2 = [workingarray(1,3) workingarray(1,3); array2];
        elseif workingarray(divind(1)) == 2
            array3 = [workingarray(1,3) workingarray(1,3); array3];
        elseif workingarray(divind(1)) == 3
            array4 = [workingarray(1,3) workingarray(1,3); array4];
        end
    end
    
    if workingarray(divind(end)+1) == -1
        tmpser = workingarray(divind(end)+1:length(workingarray),3);
        array1 = [array1; max(tmpser) min(tmpser)];
    elseif workingarray(divind(end)+1) == 1
        tmpser = workingarray(divind(end)+1:length(workingarray),3);
        array2 = [array2; max(tmpser) min(tmpser)];
    elseif workingarray(divind(end)+1) == 2
        tmpser = workingarray(divind(end)+1:length(workingarray),3);
        array3 = [array3; max(tmpser) min(tmpser)];
    elseif workingarray(divind(end)+1) == 3
        tmpser = workingarray(divind(end)+1:length(workingarray),3);
        array4 = [array4; max(tmpser) min(tmpser)];
    end
    
%resultcell = {array1, array2, array3, array4, array5, array6};

end
% try using unique and cell to create a better code


