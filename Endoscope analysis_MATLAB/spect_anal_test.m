function [spect] = spect_anal_test(address)
[workingsignal,~,~]=abfload(address);
% Fs = 2000;
spect = zeros(2,4);
% testsignal = mouse1_normal(:,1);
for ii = 1:2
testsignal = workingsignal(:,ii);

L = length(testsignal);

Y = fft(testsignal);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;

testtheta = sum(P1(7261:14520));
testdelta = sum(P1(1:7260));
testalpha = sum(P1(14521:21779));
testbeta = sum(P1(23595:54447));

spect(ii,:) = [testdelta,testtheta,testalpha,testbeta];
end
% testxax = categorical({'Delta','Theta','Alpha','Beta'});
% testxax = reordercats(testxax,{'Delta','Theta','Alpha','Beta'});
% bar(testxax,testyax)
% figure();
% bar(f(1:10:72597),P1(1:10:72597))