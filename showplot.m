close all
cell_num = 5;
behav_num = 4;
targetbehav = tmp4;

plot(timestamp,analmat(:,cell_num))
hold on
plot(timestamp(newpeakdetect{behav_num,cell_num}),analmat(newpeakdetect{behav_num,cell_num},cell_num),'o')
for ii = 1:length(targetbehav)
    patch([targetbehav(ii,2), targetbehav(ii,1), targetbehav(ii,1), targetbehav(ii,2)],[-3, -3, 4, 4], [0.9290 0.6940 0.1250] ,'FaceAlpha',0.5,'EdgeColor','none')
end
