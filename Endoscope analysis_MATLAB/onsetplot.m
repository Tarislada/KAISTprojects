function [] =  onsetplot(onsetmeancell,onsetsdcell,cellnumvec,behavnumvec,b4onsetmeancell,b4onsetsdcell)

% cellnumvec = m3t110ind{2,1};
% behavnumvec = 3*ones(length(m3t110ind{2,1}),1);


b4length = length(b4onsetmeancell{3,1});

figure()

for i = 1:length(cellnumvec)
   n_cell = cellnumvec(i);
   n_behav = behavnumvec(i);

   meancurve = onsetmeancell{n_behav,n_cell};
   stdercurve1 = onsetmeancell{n_behav,n_cell}+onsetsdcell{n_behav,n_cell};
   stdercurve2 = onsetmeancell{n_behav,n_cell}-onsetsdcell{n_behav,n_cell};
   
   basemeancurve = b4onsetmeancell{n_behav,n_cell};
   basestdcurve1 = b4onsetmeancell{n_behav,n_cell} + b4onsetsdcell{n_behav,n_cell};
   basestdcurve2 = b4onsetmeancell{n_behav,n_cell} - b4onsetsdcell{n_behav,n_cell};
   
   subplot(ceil(length(cellnumvec)/2),2,i)
   
   plot(-b4length+1:0,basemeancurve)
   hold on
   plot(-b4length+1:0,basestdcurve1)
   plot(-b4length+1:0,basestdcurve2)
   fill([-b4length+1:0 fliplr(-b4length+1:0)],[basestdcurve2; flipud(basestdcurve1)]','k','FaceAlpha',0.3)
   
%    plot(1:inputpoint,baseline(n_cell)*ones(1,inputpoint))
%    hold on
%    plot(1:inputpoint,baseline(n_cell)+baselineSD(n_cell))
%    plot(1:inputpoint,baseline(n_cell)-baselineSD(n_cell))


   plot(0:length(meancurve)-1,meancurve')
   plot(0:length(meancurve)-1,stdercurve1')
   plot(0:length(meancurve)-1,stdercurve2')
   fill([0:length(meancurve)-1 fliplr(0:length(meancurve)-1)],[stdercurve2; flipud(stdercurve1)]',[0.4660 0.6740 0.1880],'FaceAlpha',0.3)
   title("Cell "+n_cell)
end
end
