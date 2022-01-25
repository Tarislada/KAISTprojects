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
 
    for i = 1:leng
		[tmppeak,tmpindice]=findpeaks(analmat(:,i),'MinPeakHeight',1,'MinPeakProminence',0.8, 'Threshold', 0.1);  % find peaks in each cells with minimum heigth and prominence(how much it stands out from vicinity
		
        % initialize behavioral sets
		tmparray1 = [];
		tmparray2 = [];
		tmparray3 = [];
		tmparray4 = [];

        % find sections corresponding to each behavior and count peaks in
        % each corresponding section
		for ii = 1:size(tmp1,1)
			tmpval1 = find(timestamp<=tmp1(ii,1) & timestamp>=tmp1(ii,2));
			tmparray1 = [tmparray1; tmpval1];
		end
		count1 = sum(ismember(tmparray1,tmpindice));        % total number of peaks(tmpindice) in behavior 1(tmparray1)
        area1 = trapz(rawtime(tmparray1),abs(analmat(tmparray1,i)));   % total area under behavior class

		for ii = 1:size(tmp2,1)
			tmpval2 = find(timestamp<=tmp2(ii,1) & timestamp>=tmp2(ii,2));
			tmparray2 = [tmparray2; tmpval2];
		end
		count2 = sum(ismember(tmparray2,tmpindice));
		area2 = trapz(rawtime(tmparray2),abs(analmat(tmparray2,i)));

		for ii = 1:size(tmp3,1)
			tmpval3 = find(timestamp<=tmp3(ii,1) & timestamp>=tmp3(ii,2));
			tmparray3 = [tmparray3; tmpval3];
		end
		count3 = sum(ismember(tmparray3,tmpindice));
		area3 = trapz(rawtime(tmparray3),abs(analmat(tmparray3,i)));

		for ii = 1:size(tmp4,1)
			tmpval4 = find(timestamp<=tmp4(ii,1) & timestamp>=tmp4(ii,2));
			tmparray4 = [tmparray4; tmpval4];
		end
		count4 = sum(ismember(tmparray4,tmpindice));
		area4 = trapz(rawtime(tmparray4),abs(analmat(tmparray4,i)));

		% collect all counts into a matrix for later usage
        countmat(1,i) = count1;
		countmat(2,i) = count2;
		countmat(3,i) = count3;
		countmat(4,i) = count4;
        
        % collect all areas into a matrix for later usage
		areamat(1,i) = area1;
		areamat(2,i) = area2;
		areamat(3,i) = area3;
		areamat(4,i) = area4;        
        
	end
% band specification is bit off
% most likely due to rounding to use things such as the line34
% somehow, the source of time is different, not allowing me to extract the
% exact timeband of each behavior. this is making me bleed
% one possible solution is using the map function to forcibly match the
% two - not a good solution tho. 
% Actually, the behavior standard has changed - it does not match the
% original frame by frame behavioral bands. 

% Another peak finding method is using binary log ratio: activity while
% in action of interest / activity while not in action of interest need to
% be over M+-1SD

% use function 'trapz' to calculate area under the line
% cf) integral, integral2, cumtrapz
%end