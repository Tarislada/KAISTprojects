%% Call files and variables
dirname = uigetdir();

filepatx = fullfile(dirname,'*_x_*.mat');
filenamex = dir(filepatx);

filepaty = fullfile(dirname,'*_y_*.mat');
filenamey = dir(filepaty);

filepatt = fullfile(dirname,'LABEL*.mat');
filenamet = dir(filepatt);

xcell = cell(4,1);
ycell = cell(4,1);

for i = 1:4 
    xcell{i} = load(filenamex(i).name);
    ycell{i} = load(filenamey(i).name);
end

tab = load(filenamet.name);
wall = table2array(tab.tblLbls(9:11,4));
space = table2array(tab.tblLbls(5:8,4));
centerpoint=[960,540];
% 26233개? how to downscale to 10hz (endoscope)
% downscale & cut after conpoint
% issues: boundary? just enlarge?
%% Position data preprocessing - downscaling, centering, fliplr based on cricket position
space - space(1,:)

headpos = [cell2mat(xcell{2}.data); cell2mat(ycell{2}.data)];
foodpos = [cell2mat(xcell{3}.data); cell2mat(ycell{3}.data)];

xsize = 1200;
ysize = 1200;
%cellcount = ?


%% Create heatmap variable
heatcube = zeros(xsize,ysize,cellcount);
for ii = 1:length(cell2mat(xcell{1}.data))
    heatcube(headpos(ii,1),(headpos(ii,2)),cell(:,n_cell)) = heatcube(headpos(ii,1),(headpos(ii,2)),cell) + cellact(ii,n_cell);
end

%% Plot heatmap





% x3,y3 기준으로 x3가 mid point를 넘어서면
% x1 or x2로 측정하는 현재 쥐의 포지션을 반전 (가운데를 계산해서 그걸 기준으로)
% x1/x2가 행렬의 x축, y1/y2가 행렬의 y축이 되므로
% 정수화 
% 행렬의 element는 세포의 df/f로
% 3차원 행렬을 쓰자
% position x, y 
% cell num z
% df/f as element
% zeros(size(전체 apparatus))
% 
% dot product to cosine angle between two vectors
% unit vector to compare directions
% unit vector = divide itself by norm
% 
% # how to set a corner to 0?
% the apparatus is tilted.
% 
