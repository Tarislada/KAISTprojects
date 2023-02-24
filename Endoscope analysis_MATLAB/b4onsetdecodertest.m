pursuitmat = [];
tmppursuitmat = [];
for celln = 1:length(m6order(:,1))
    tmppursuitmat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t1{2,m6order(celln,1)}, 'UniformOutput', false))';
    pursuitmat = [pursuitmat tmppursuitmat];
end
pursuitmat = [pursuitmat ones(size(pursuitmat,1),1)];

attackmat = [];
tmpattackmat = [];
for celln = 1:length(m6order(:,1))
    tmpattackmat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t1{3,m6order(celln,1)}, 'UniformOutput', false))';
    attackmat = [attackmat tmpattackmat];
end
attackmat = [attackmat 2*ones(size(attackmat,1),1)];

consumemat = [];
tmpconsumemat = [];
for celln = 1:length(m6order(:,1))
    tmpconsumemat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t1{4,m6order(celln,1)}, 'UniformOutput', false))';
    consumemat = [consumemat tmpconsumemat];
end
consumemat = [consumemat 3*ones(size(consumemat,1),1)];
% pursuitmat = [];
% tmppursuitmat = [];
% for celln = m6order(:,1)
%     tmppursuitmat = cell2mat(cellfun(@(x) x(1:15), testrawonset{2,celln}, 'UniformOutput', false))';
%     pursuitmat = [pursuitmat tmpursuitmat];
% end
m6t1before = [pursuitmat;attackmat;consumemat];

pursuitmat = [];
tmppursuitmat = [];
for celln = 1:length(m6order(:,2))
    tmppursuitmat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t2{2,m6order(celln,2)}, 'UniformOutput', false))';
    pursuitmat = [pursuitmat tmppursuitmat];
end
pursuitmat = [pursuitmat ones(size(pursuitmat,1),1)];

attackmat = [];
tmpattackmat = [];
for celln = 1:length(m6order(:,2))
    tmpattackmat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t2{3,m6order(celln,2)}, 'UniformOutput', false))';
    attackmat = [attackmat tmpattackmat];
end
attackmat = [attackmat 2*ones(size(attackmat,1),1)];

consumemat = [];
tmpconsumemat = [];
for celln = 1:length(m6order(:,2))
    tmpconsumemat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t2{4,m6order(celln,2)}, 'UniformOutput', false))';
    consumemat = [consumemat tmpconsumemat];
end
consumemat = [consumemat 3*ones(size(consumemat,1),1)];
m6t2before = [pursuitmat;attackmat;consumemat];

pursuitmat = [];
tmppursuitmat = [];
for celln = 1:length(m6order(:,3))
    tmppursuitmat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t3{2,m6order(celln,3)}, 'UniformOutput', false))';
    pursuitmat = [pursuitmat tmppursuitmat];
end
pursuitmat = [pursuitmat ones(size(pursuitmat,1),1)];

attackmat = [];
tmpattackmat = [];
for celln = 1:length(m6order(:,3))
    tmpattackmat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t3{3,m6order(celln,3)}, 'UniformOutput', false))';
    attackmat = [attackmat tmpattackmat];
end
attackmat = [attackmat 2*ones(size(attackmat,1),1)];

consumemat = [];
tmpconsumemat = [];
for celln = 1:length(m6order(:,3))
    tmpconsumemat = cell2mat(cellfun(@(x) x(1:15), rawonset_m6t3{4,m6order(celln,3)}, 'UniformOutput', false))';
    consumemat = [consumemat tmpconsumemat];
end
consumemat = [consumemat 3*ones(size(consumemat,1),1)];
m6t3before = [pursuitmat;attackmat;consumemat];

m6_b4ons_class = [m6t1before;m6t2before;m6t3before];