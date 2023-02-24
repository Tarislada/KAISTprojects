function [modevent,casemean1,casemean2,casestd] = bandrate(event_array,timecell)

% modevent = zeros(size(event_array));
% for ii = 1:size(event_array,2)
%     wrkingvec = event_array(:,ii);
%     threshold = mean(nonzeros(wrkingvec)) - std(nonzeros(wrkingvec));
%     wrkingvec(wrkingvec<threshold) = 0;
%     modevent(:,ii) = wrkingvec;
% end
modevent = event_array;


casemean1 = zeros(4,size(event_array,2));
casemean2 = zeros(4,size(event_array,2));
casestd = zeros(4,size(event_array,2));
ratecell = cell(4,size(event_array,2));

for ii = 1:size(event_array,2)
case1count = 0;
case2count = 0;
case3count = 0;
    for i = 1:length(timecell{2,1})
        tmpcount = sum(nonzeros(modevent(timecell{2,1}(i,2):timecell{2,1}(i,1),ii)));
        case1count = tmpcount+case1count;
        ratecell{2,ii}(i)=tmpcount/(length(timecell{2,1}(i,2):timecell{2,1}(i,1))/10);
    end
    casemean1(2,ii) = case1count/(sum(timecell{2,1}(:,1)-timecell{2,1}(:,2))/10);
    casemean2(2,ii) = mean(ratecell{2,ii});
    casestd(2,ii) = std(ratecell{2,ii})/sqrt(length(timecell{2,1}));

    for i = 1:length(timecell{3,1})
        tmpcount = sum(nonzeros(modevent(timecell{3,1}(i,2):timecell{3,1}(i,1))));
        case2count = tmpcount+case2count;
        ratecell{3,ii}(i)=tmpcount/(length(timecell{3,1}(i,2):timecell{3,1}(i,1))/10);
    end
    casemean1(3,ii) = case2count/(sum(timecell{3,1}(:,1)-timecell{3,1}(:,2))/10);
    casemean2(3,ii) = mean(ratecell{3,ii});
    casestd(3,ii) = std(ratecell{3,ii})/sqrt(length(timecell{3,1}));

    for i = 1:length(timecell{4,1})
        tmpcount = sum(nonzeros(modevent(timecell{4,1}(i,2):timecell{4,1}(i,1))));
        case3count = tmpcount+case3count;
        ratecell{4,ii}(i)=tmpcount/(length(timecell{4,1}(i,2):timecell{4,1}(i,1))/10);
    end
    casemean1(4,ii) = case3count/(sum(timecell{3,1}(:,1)-timecell{3,1}(:,2))/10);
    casemean2(4,ii) = mean(ratecell{4,ii});
    casestd(4,ii) = std(ratecell{4,ii})/sqrt(length(timecell{4,1}));
    
end

% 현재 event 자체가 너무 적음
