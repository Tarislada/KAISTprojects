function newtimecell = combinetime(timecell)
%     timecell = testtimecell;
    newtimecell = cell(3,1);
    workingmat = flipud(sort([timecell{3}; timecell{2}])');
    realigned = workingmat(:);
    deleteidx = [];
    for ii = 2:2:length(realigned)-1
        if realigned(ii)+1 == realigned(ii+1)
            deleteidx = [deleteidx; ii];
        end
    end
    deleteidx = [deleteidx; deleteidx+1];
    realigned(deleteidx) = [];
    newmat = fliplr(reshape(realigned,2,[])');
    newtimecell{1} = timecell{1};
    newtimecell{2} = newmat;
    newtimecell{3} = timecell{4};
end