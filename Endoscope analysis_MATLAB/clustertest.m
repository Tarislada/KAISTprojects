%cluster hyperparameter testing
function [F1,hiervec,kmeansvec,spectralvec,dbscanvec] = clustertest(testmat)
% testmat = newwtclcell{1}';
%% Declaring variables
leng = floor(size(testmat,1)/2.5);

F1 = figure();

%% PCA 
[~,score,~,~,explained] = pca(testmat);
compnum = find(cumsum(explained)>95,1);
pcadmat = score(:,1:compnum);
%% Hierarchy
hiervec = zeros(leng,1);
for ii = 3:leng
    S = silhouette(pcadmat,clusterdata(pcadmat,'MaxClust',ii));
    hiervec(ii) = mean(S);
end
[val,idx]=max(hiervec);
subplot(4,1,1);
h1 = plot(hiervec);
fprintf('Best performance of Hierarchical clustering is %f in %d number of clusters \n', val,idx);

%% Kmeans
kmeansvec = zeros(leng,1);
for ii = 3:leng
    S = silhouette(pcadmat,kmeans(pcadmat,ii));
    kmeansvec(ii) = mean(S);
end
[val,idx]=max(kmeansvec);
subplot(4,1,2);
k1 = plot(kmeansvec);
fprintf('Best performance of K-means clustering is %f in %d number of clusters \n', val,idx);
fprintf('\n');
%% Spectral
spectralvec = zeros(leng,1);
for ii = 3:leng
    try
        S = silhouette(pcadmat,spectralcluster(pcadmat,ii));
    catch
        continue
    end
    spectralvec(ii) = mean(S);
end
[val,idx]=max(spectralvec);
subplot(4,1,3);
s1 = plot(spectralvec);
fprintf('Best performance of Spectral clustering is %f in %d number of clusters \n', val,idx);

%% dBscan
% dbscan epsilon depends on the x and y scale. therefore, check the min &
% max of the plain you are clustering in. In this case, space is 1000 wide,
% so a suggestable value is 200, and minpts means how many points are
% needed within epsilon distance to be considered a cluster, so smaller the
% number, the more clusters you get. 2
dbscanvec = zeros(leng-2,1);
for ii = 2:leng-1
    S = silhouette(pcadmat,dbscan(pcadmat,ii,size(pcadmat,2)*2));
    dbscanvec(ii) = mean(S);
end
[val,idx]=max(dbscanvec);
subplot(4,1,4);
d1 = plot(dbscanvec);
fprintf('Best performance of dbscan clustering is %f in %d epsilon \n', val,idx);
% does not work - curse of dimensionality
%% GMM
% 
% gmmvec = zeros(leng-2,1);
% bothvec = cell(leng-2,1);
% for ii = 2:leng-1
%     gm = fitgmdist(testmat,ii);
%     threshold = [0.4 0.6];
%     P = posterior(gm,X);
%     S = silhouette(testmat,cluster(gm,X));
%     bothvec{ii} = find(P(:,1)>=threshold(1) & P(:,1)<=threshold(2)); 
%     gmmvec(ii) = mean(S);    
% end
% [val,idx]=max(gmmvec);
% subplot(5,1,5);
% g1 = plot(gmmvec);
% fprintf('Best performance of GMM clustering is %f in %d number of clusters \n', val,idx);
% does not work - curse of dimensionality
%% Spectral dbscan

[~,V,D] = spectralcluster(testmat,15);
% calculate 15 smallest eigenvectors from laplacian matrix of testmat
[idx] = dbscan(V,0.2,2);
% cluster that eigenvectors with dbscan method, with cluster defined as
% atleast 2 points within 0.2 distance