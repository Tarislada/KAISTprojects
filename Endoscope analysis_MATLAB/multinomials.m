function [event,pObs,pval] = multinomials(targetvec,probvec)

K = sum(targetvec)+length(targetvec)-1;
n = length(targetvec);
c = nchoosek(1:K,n-1);
pObs = mnpdf(targetvec,probvec);
m = size(c,1);
A = zeros(m,n);
for ix = 1:m
    A(ix,:) = diff([1,c(ix,:),K+1]);
end
B = zeros(size(A));
B(:,1) = A(:,1);
B(:,2:end) = A(:,2:end)-1;
r = mnpdf(B,probvec);
pval = sum(r(r<pObs));
event = m;
