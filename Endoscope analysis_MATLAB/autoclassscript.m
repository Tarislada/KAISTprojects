%% Delete all 0s
% testcollmatcellidx = cellfun(@(x) find(~x(:,end)),collmatcell,'UniformOutput',false);

tmpmat = [];
tmpcell = collmatcell;
% for ii = 1:length(collmatcell)
% tmpcell{ii}(testcollmatcellidx{ii},:)=[];
% end
%% Delete after task idle 0s
collmatcell_n0 = collmatcell;
for ii = 1:length(collmatcell)
    collmatcell_n0{ii}(ceil(incon(ii,2)/15)+1:end,:)=[];
end

inccollmatcell_n0 = inccollmatcell;
for ii = 1:length(inccollmatcell)
    inccollmatcell_n0{ii}(ceil(incon(ii,2)/15)+1:end,:)=[];
end

deccollmatcell_n0 = deccollmatcell;
for ii = 1:length(deccollmatcell)
    deccollmatcell_n0{ii}(ceil(incon(ii,2)/15)+1:end,:)=[];
end

%% Delete 1
% testcollmatcellidx = cellfun(@(x) find(x(:,end)==1),collmatcell,'UniformOutput',false);
% tmpmat = [];
% for ii = 1:length(collmatcell)
% tmpcell{ii}(testcollmatcellidx{ii},:)=[];
% end
% 

dirname = 'C:\Users\endyd\OneDrive\Onedrive-CK\OneDrive\문서\카카오톡 받은 파일\walltaskclassifier\idle0del\allcells\';
allcellaccuracy = zeros(22,2);
errorvec = zeros(3,22);
for i = 1:22
%     accuracy(i,2) = autoclass(shufflematcell{i});
%     autoclass(collmatcell{i});
    try
        autoclass(collmatcell_n0{i});
        filename = append(dirname,list(i),'.fig');
        saveas(gcf,filename);
        close all
    catch
        errorvec(1,i) = 1;
        continue
    end
    
%     filename = append(dirname,list(i),'_sh','.fig');
    
end


dirname = 'C:\Users\endyd\OneDrive\Onedrive-CK\OneDrive\문서\카카오톡 받은 파일\walltaskclassifier\idle0del\inccells\';
incaccuracy = zeros(22,2);
for i = 1:22
    
%     accuracy(i,2) = autoclass(shufflematcell{i});
%     autoclass(collmatcell{i});
    try
        autoclass(inccollmatcell_n0{i});
%     filename = append(dirname,list(i),'_sh','.fig');
    filename = append(dirname,list(i),'.fig');
    saveas(gcf,filename);
    close all
    catch
        errorvec(2,i) = 1;
        continue
    end
end


dirname = 'C:\Users\endyd\OneDrive\Onedrive-CK\OneDrive\문서\카카오톡 받은 파일\walltaskclassifier\idle0del\deccells\';
deaccuracy = zeros(22,2);
for i = 1:22
%     accuracy(i,2) = autoclass(shufflematcell{i});
%     autoclass(collmatcell{i});
    try
        autoclass(deccollmatcell_n0{i});
%     filename = append(dirname,list(i),'_sh','.fig');
    filename = append(dirname,list(i),'.fig');
    saveas(gcf,filename);
    close all
    catch
        errorvec(3,i) = 1;
    end
end
