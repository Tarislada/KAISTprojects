N = 2000;

f1 = [2.5 4.5];
f2 = [10.5 12.5];
f3 = [15 18];


[testsignal]=abfload('C:\Users\endyd\OneDrive\Onedrive-CK\OneDrive\바탕 화면\EEG_rawdata_YW\alleegdata\22727006.abf');workingsig = testsignal(:,1);
T = length(testsignal);
ch1 = testsignal(:,1);
ch2 = testsignal(:,2);

ch1f1 = cwt(ch1,'amor',2000,'FrequencyLimits',f1);
ch1f2 = cwt(ch1,'amor',2000,'FrequencyLimits',f2);
ch1f3 = cwt(ch1,'amor',2000,'FrequencyLimits',f3);

ch2f1 = cwt(ch2,'amor',2000,'FrequencyLimits',f1);
ch2f2 = cwt(ch2,'amor',2000,'FrequencyLimits',f2);
ch2f3 = cwt(ch2,'amor',2000,'FrequencyLimits',f3);

energych1f1 = abs(ch1f1).^2;
energych1f2 = abs(ch1f2).^2;
energych1f3 = abs(ch1f3).^2;

energych2f1 = abs(ch2f1).^2;
energych2f2 = abs(ch2f2).^2;
energych2f3 = abs(ch2f3).^2;

% energych1f1 = sqrt(sum(abs(ch1f1).^2));
% energych1f2 = sqrt(sum(abs(ch1f2).^2));
% energych1f3 = sqrt(sum(abs(ch1f3).^2));

totenergych1f1 = sum(energych1f1);
totenergych1f2 = sum(energych1f2);
totenergych1f3 = sum(energych1f3);

totenergych2f1 = sum(energych2f1);
totenergych2f2 = sum(energych2f2);
totenergych2f3 = sum(energych2f3);

deltat = 0.5*N;

intenergych1f1 = movmean(totenergych1f1,0.5*deltat);
intenergych1f2 = movmean(totenergych1f2,0.5*deltat);
intenergych1f3 = movmean(totenergych1f3,0.5*deltat);

intenergych2f1 = movmean(totenergych2f1,0.5*deltat);
intenergych2f2 = movmean(totenergych2f2,0.5*deltat);
intenergych2f3 = movmean(totenergych2f3,0.5*deltat);

% energych1f1 = sqrt(sum(abs(ch1f1).^2));
% energych1f2 = sqrt(sum(abs(ch1f2).^2));
% energych1f3 = sqrt(sum(abs(ch1f3).^2));

% energych1f1 = sqrt(sum(abs(ch1f1).^2,2));
% energych1f2 = sqrt(sum(abs(ch1f2).^2,2));
% energych1f3 = sqrt(sum(abs(ch1f3).^2,2));
% 
% energych2f1 = sqrt(sum(abs(ch2f1).^2));
% energych2f2 = sqrt(sum(abs(ch2f2).^2));
% energych2f3 = sqrt(sum(abs(ch2f3).^2));

% energych2f1 = sqrt(sum(abs(ch2f1).^2,2));
% energych2f2 = sqrt(sum(abs(ch2f2).^2,2));
% energych2f3 = sqrt(sum(abs(ch2f3).^2,2));

% energyratioch1 = energych1f3 ./ (energych1f1+energych1f2);
% energyratioch2 = energych2f3 ./ (energych2f1+energych2f2);

energyratioch1 = intenergych1f1 ./ (intenergych1f3+intenergych1f2);
energyratioch2 = intenergych2f1 ./ (intenergych2f3+intenergych2f2);


energychar = (energyratioch1+energyratioch2) / 2;

tau = 3*N;
integralenergychar = movmean(energychar,tau*2*1.5);



thresh1 = mean(integralenergychar)*1.75;
thresh2 = mean(integralenergychar)*1.55;

% based on
% Spike-Wave Seizures, NREM Sleep and Micro-Arousals in WAG/Rij Rats with 
% Genetic Predisposition to Absence Epilepsy: Developmental Aspects

% lessons: 
% 1) on and off threshold based on average 
% 2) sum of energy and ratio - 3~5 vs rest
% 3) using both channels