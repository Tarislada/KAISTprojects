% close all
%cell_num = 5;
% behav_num = 4;
% targetbehav = tmp4;
function [] = showplot3(analmat,timecell)
% analmat = all modified signals matrix
% newpeakdetect = cell containing the peak indices
% analmat = norm_analmat;
% newpeakdetect = norm_peakindcell;



for cell_num = 1:size(analmat,2)
        
        plot_pos=mod(cell_num,10);
        if plot_pos == 0
            file_name = ['C:\Users\endyd\Desktop\figures\' num2str(floor(cell_num/10)-1) '.fig'];
%             saveas(gcf,file_name)
            figure()
        end
        
        subplot(5,2,plot_pos+1)
        
        plot(analmat(:,cell_num))
        title("Cell "+cell_num)
        hold on
        %plot(timestamp(newpeakdetect{behav_num,cell_num}),analmat(newpeakdetect{behav_num,cell_num},cell_num),'o')
 
        for ii = 1:length(timecell{1,1})% behavior coding-1 yellow #EDB120
            patch([timecell{1,1}(ii,2), timecell{1,1}(ii,1), timecell{1,1}(ii,1), timecell{1,1}(ii,2)],[-3, -3, 80, 80], [0.9290 0.6940 0.1250] ,'FaceAlpha',0.5,'EdgeColor','none')
        end
    
        for ii = 1:length(timecell{2,1})% behavior coding1 orange '#D95319'
            patch([timecell{2,1}(ii,2), timecell{2,1}(ii,1), timecell{2,1}(ii,1), timecell{2,1}(ii,2)],[-3, -3, 80, 80], [0.8500 0.3250 0.0980] ,'FaceAlpha',0.5,'EdgeColor','none')
        end

        for ii = 1:length(timecell{3,1}) % behavior coding2 green '#77AC30'	
            patch([timecell{3,1}(ii,2), timecell{3,1}(ii,1), timecell{3,1}(ii,1), timecell{3,1}(ii,2)],[-3, -3, 80, 80], [0.4660 0.6740 0.1880] ,'FaceAlpha',0.5,'EdgeColor','none')
        end

        for ii = 1:length(timecell{4,1}) % behavior coding3 purple '#7E2F8E'	
            patch([timecell{4,1}(ii,2), timecell{4,1}(ii,1), timecell{4,1}(ii,1), timecell{4,1}(ii,2)],[-3, -3, 80, 80], [0.4940 0.1840 0.5560] ,'FaceAlpha',0.5,'EdgeColor','none')
        end      
    
end
% filename = ['C:\Users\endyd\Desktop\figures\' num2str(floor(size(analmat,2)/10)-1) '.fig'];
% saveas(gcf,filename)
%end
%end
