%function [countmat, areamat] = peakcount(datamat, class1, class2, class3, class4)
% CK, 2022.01
% input: datamat including the timestamp in the leftmost column, and
% active/inactive logical on the top row. Each column will be a cell, each
% row will be response at the corresponding time. 
% input: class1~4 includes the beginning and end timeframe data of each behavior. If there are more/less than
% 4 classes of behavior, change all 4 level variables to corresponding
% number, and write/delete a few for loops.
% output - countmat: count of peaks. each row accounts for each behavior classes,
% and each colum represents an active cell. areamat: area under the curve
% of each behavior class bands. 

% Peak parameter is right under first for loop - change if needed.
% Area calculation is under each counting line. change if needed.

    % variable setting for testing. 
    datamat = m3_t1_210810;
    class1 = array1;
    class2 = array2;
    class3 = array3;
    class4 = array4;
    
    % trimming data to get timestamps and preparing to match frame data
	rawtime = datamat(2:end,1);
	timestamp = round(rawtime,1);

    % preparing frame data to match with each behavior type data, 
	tmp1 = round(class1,1);
	tmp2 = round(class2,1);
	tmp3 = round(class3,1);
	tmp4 = round(class4,1);
	
    
    % preprocess data - select only active data and normalize.
    workingmat = datamat(:,2:end);
	rawanalmat = workingmat(2:end,workingmat(1,:)==1);
    analmat = normalize(rawanalmat,1);
    
    % setup variables for calculation
    [wid,leng] = size(analmat);
	%indicecell = {};
	%peakcell = {};
	countmat = nan(4,leng);
	areamat =  nan(4,leng);
    newpeakdetect = cell(4,leng);
    newpeakcheck = cell(4,leng);
    
    for i = 1:leng
        
        tmpnewpeak1 = [];
        tmpnewpeak2 = [];
        tmpnewpeak3 = [];
        tmpnewpeak4 = [];
        
        tmppeakval1 = [];
        tmppeakval2 = [];
        tmppeakval3 = [];
        tmppeakval4 = [];
        
        for ii = 1:size(tmp1,1)
			tmpval1 = find(timestamp<=tmp1(ii,1) & timestamp>=tmp1(ii,2));
            if any(zscore(rawanalmat(tmpval1,i))>=1)
                % newpeak1 = tmpval1(zscore(rawanalmat(tmpval1,i))>=1);
                [rawpeakval1,rawnewpeak1] = findpeaks(analmat(tmpval1,i),'MinPeakProminence',0.2621, 'MinPeakHeight', 2,'MinPeakDistance',0.3);
                newpeak1 = tmpval1(rawnewpeak1);
                tmpnewpeak1 = [tmpnewpeak1; newpeak1];
                tmppeakval1 = [tmppeakval1; rawpeakval1];
            end

        end
        newpeakdetect{1,i} = tmpnewpeak1;
        newpeakcheck{1,i} = tmppeakval1;

        for ii = 1:size(tmp2,1)
            tmpval2 = find(timestamp<=tmp2(ii,1) & timestamp>=tmp2(ii,2));
            if any(zscore(rawanalmat(tmpval2,i))>=1)
                % newpeak2 = tmpval2(zscore(rawanalmat(tmpval2,i))>=1);
                [rawpeakval2,rawnewpeak2] = findpeaks(analmat(tmpval2,i),'MinPeakProminence',0.2621, 'MinPeakHeight', 2,'MinPeakDistance',0.3);
                newpeak2 = tmpval2(rawnewpeak2);
                tmpnewpeak2 = [tmpnewpeak2; newpeak2];
                tmppeakval2 = [tmppeakval2; rawpeakval2];
            end

        end
        newpeakdetect{2,i} = tmpnewpeak2;
        newpeakcheck{2,i} = tmppeakval2;

        for ii = 1:size(tmp3,1)
            tmpval3 = find(timestamp<=tmp3(ii,1) & timestamp>=tmp3(ii,2));
            if any(zscore(rawanalmat(tmpval3,i))>=1)
                % newpeak3 = tmpval3(zscore(rawanalmat(tmpval3,i))>=1);
                [rawpeakval3,rawnewpeak3] = findpeaks(analmat(tmpval3,i),'MinPeakProminence',0.2621, 'MinPeakHeight', 2,'MinPeakDistance',0.3);
                newpeak3 = tmpval3(rawnewpeak3);
                tmpnewpeak3 = [tmpnewpeak3; newpeak3];
                tmppeakval3 = [tmppeakval3; rawpeakval3];
            end

        end
        newpeakdetect{3,i} = tmpnewpeak3;
        newpeakcheck{3,i} = tmppeakval3;

        for ii = 1:size(tmp4,1)
            tmpval4 = find(timestamp<=tmp4(ii,1) & timestamp>=tmp4(ii,2));
            if any(zscore(rawanalmat(tmpval4,i))>=1)
                % newpeak4 = tmpval4(zscore(rawanalmat(tmpval4,i))>=1);
                [rawpeakval4,rawnewpeak4] = findpeaks(analmat(tmpval4,i),'MinPeakProminence',0.2621, 'MinPeakHeight', 2,'MinPeakDistance',0.3);
                newpeak4 = tmpval4(rawnewpeak4);
                tmpnewpeak4 = [tmpnewpeak4; newpeak4];
                tmppeakval4 = [tmppeakval4; rawpeakval4];
            end

        end
        newpeakdetect{4,i} = tmpnewpeak4;
        newpeakcheck{4,i} = tmppeakval4; 
    end
    countmat = cellfun(@length,newpeakdetect);
    
    