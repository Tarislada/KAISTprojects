%% M345 clustertest matrix generation
walltaskclasscell = cell(22,1);
walltaskshufflecell = cell(22,1);
for ii = 1:length(walltaskclasscell)
    [tmpclass,tmpshuffle] = class_prep(dnsigcell{ii},timecellcell{ii},1:size(dnsigcell{ii},2));
    walltaskclasscell{ii} = tmpclass;
    walltaskshufflecell{ii} = tmpshuffle;
end
newwtclcell = cell(22,1);
binlength = 5;
for ii = 1:length(newwtclcell)
    tmpmat = walltaskclasscell{ii};
    tmpmat2 = tmpmat(:,1:end-1);
    size1 = size(tmpmat2,1);
    size2 = size(tmpmat2,2);
    newwtclcell{ii} = reshape(mean(reshape(tmpmat2,size1,binlength,[]),2),size1,[]);
end
%shapetest = reshape(mean(reshape(testmat,81,15,94),2),81,94);




%%
[m6t1posmat,m6t1posmat_shuffle]=class_prep(dn_m6t1,timecell_m6t1,unique(horzcat(inccell_m6t1{:})));
[m6t2posmat,m6t2posmat_shuffle]=class_prep(dn_m6t2,timecell_m6t2,unique(horzcat(inccell_m6t2{:})));
[m6t3posmat,m6t3posmat_shuffle]=class_prep(dn_m6t3,timecell_m6t3,unique(horzcat(inccell_m6t3{:})));

[m7t1posmat,m7t1posmat_shuffle]=class_prep(dn_m7t1,timecell_m7t1,unique(horzcat(inccell_m7t1{:})));
[m7t2posmat,m7t2posmat_shuffle]=class_prep(dn_m7t2,timecell_m7t2,unique(horzcat(inccell_m7t2{:})));
[m7t3posmat,m7t3posmat_shuffle]=class_prep(dn_m7t3,timecell_m7t3,unique(horzcat(inccell_m7t3{:})));

[m6t1negmat,m6t1negmat_shuffle]=class_prep(dn_m6t1,timecell_m6t1,unique(horzcat(deccell_m6t1{:})));
[m6t2negmat,m6t2negmat_shuffle]=class_prep(dn_m6t2,timecell_m6t2,unique(horzcat(deccell_m6t2{:})));
[m6t3negmat,m6t3negmat_shuffle]=class_prep(dn_m6t3,timecell_m6t3,unique(horzcat(deccell_m6t3{:})));
 
[m7t1negmat,m7t1negmat_shuffle]=class_prep(dn_m7t1,timecell_m7t1,unique(horzcat(deccell_m7t1{:})));
[m7t2negmat,m7t2negmat_shuffle]=class_prep(dn_m7t2,timecell_m7t2,unique(horzcat(deccell_m7t2{:})));
[m7t3negmat,m7t3negmat_shuffle]=class_prep(dn_m7t3,timecell_m7t3,unique(horzcat(deccell_m7t3{:})));

minsize = min([size(m6t1posmat,2) size(m6t2posmat,2) size(m6t3posmat,2)])-1;
m6posmat = [m6t1posmat(:,randperm(size(m6t1posmat,2)-1,minsize));m6t2posmat(:,randperm(size(m6t2posmat,2)-1,minsize));m6t3posmat(:,randperm(size(m6t3posmat,2)-1,minsize))];
m6posmat(:,minsize+1) = [m6t1posmat(:,end);m6t2posmat(:,end);m6t3posmat(:,end)];
m6posmat_shuffle = [m6t1posmat_shuffle(:,randperm(size(m6t1posmat,2)-1,minsize));m6t2posmat_shuffle(:,randperm(size(m6t2posmat,2)-1,minsize));m6t3posmat_shuffle(:,1:end-1)];
 
minsize = min([size(m6t1negmat,2) size(m6t2negmat,2) size(m6t3negmat,2)])-1;
m6negmat = [m6t1negmat(:,randperm(size(m6t1negmat,2)-1,minsize));m6t2negmat(:,randperm(size(m6t2negmat,2)-1,minsize));m6t3negmat(:,randperm(size(m6t3negmat,2)-1,minsize))];
m6negmat(:,minsize+1) = [m6t1negmat(:,end);m6t2negmat(:,end);m6t3negmat(:,end)];
m6negmat_shuffle = [m6t1negmat_shuffle(:,randperm(size(m6t1negmat,2)-1,minsize));m6t2negmat_shuffle(:,randperm(size(m6t2negmat,2)-1,minsize));m6t3negmat_shuffle(:,1:end-1)];
 
minsize = min([size(m7t1posmat,2) size(m7t2posmat,2) size(m7t3posmat,2)])-1;
m7posmat = [m7t1posmat(:,randperm(size(m7t1posmat,2)-1,minsize));m7t2posmat(:,randperm(size(m7t2posmat,2)-1,minsize));m7t3posmat(:,randperm(size(m7t3posmat,2)-1,minsize))];
m7posmat(:,minsize+1) = [m7t1posmat(:,end);m7t2posmat(:,end);m7t3posmat(:,end)];
m7posmat_shuffle = [m7t1posmat_shuffle(:,randperm(size(m7t1posmat,2)-1,minsize));m7t2posmat_shuffle(:,randperm(size(m7t2posmat,2)-1,minsize));m7t3posmat_shuffle(:,1:end-1)];
 
minsize = min([size(m7t1negmat,2) size(m7t2negmat,2) size(m7t3negmat,2)])-1;
m7negmat = [m7t1negmat(:,randperm(size(m7t1negmat,2)-1,minsize));m7t2negmat(:,randperm(size(m7t2negmat,2)-1,minsize));m7t3negmat(:,randperm(size(m7t3negmat,2)-1,minsize))];
m7negmat(:,minsize+1) = [m7t1negmat(:,end);m7t2negmat(:,end);m7t3negmat(:,end)];
m7negmat_shuffle = [m7t1negmat_shuffle(:,randperm(size(m7t1negmat,2)-1,minsize));m7t2negmat_shuffle(:,randperm(size(m7t2negmat,2)-1,minsize));m7t3negmat_shuffle(:,1:end-1)];


tmplabel = m6posmat(:,end);
tmplabel(tmplabel>1) = 2;
m6posinitial = m6posmat;
m6posinitial(:,end) = tmplabel;

tmplabel = m7posmat(:,end);
tmplabel(tmplabel>1) = 2;
m7posinitial = m7posmat;
m7posinitial(:,end) = tmplabel;

tmplabel = m6negmat(:,end);
tmplabel(tmplabel>1) = 2;
m6neginitial = m6negmat;
m6neginitial(:,end) = tmplabel;

tmplabel = m7negmat(:,end);
tmplabel(tmplabel>1) = 2;
m7neginitial = m7negmat;
m7neginitial(:,end) = tmplabel;

yfit = m6posinitialtrainer.predictFcn(m6posinitial(:,1:end-1));
huntingidx = find(yfit==2);
tmpmat = m6posinitial(huntingidx,:);
m6possecondary = m6posmat(huntingidx,:);

yfit = m6neginitialtrain.predictFcn(m6neginitial(:,1:end-1));
huntingidx = find(yfit==2);
tmpmat = m6neginitial(huntingidx,:);
m6negsecondary = m6negmat(huntingidx,:);

yfit = m7neginitialtrainer.predictFcn(m7neginitial(:,1:end-1));
huntingidx = find(yfit==2);
tmpmat = m7neginitial(huntingidx,:);
m7negsecondary = m7negmat(huntingidx,:);

yfit = m7posinitialtrainer.predictFcn(m7posinitial(:,1:end-1));
huntingidx = find(yfit==2);
tmpmat = m7posinitial(huntingidx,:);
m7possecondary = m7posmat(huntingidx,:);