function [filtered] = custom_gaussfilt(z,sigma)
% Custom gaussian filter for photometry dataset
% removed edge effect from convolution
% input argument
% z: raw signal that needs to be filtered
% sigma: level of filtering
% output argument
% filtered: filtered signal

dt = 1;
n = length(z);
t = 1:n;
a = 1/(sqrt(2*pi)*sigma);  
custfilter = dt*a*exp(-0.5*((t - mean(t)).^2)/(sigma^2));
i = custfilter < dt*a*1.e-6;
custfilter(i) = [];
filtered = conv(z,custfilter,'same');
onesToFilt = ones(size(z));     % remove edge effect from conv 
onesFilt = conv(onesToFilt,custfilter,'same'); 
filtered = filtered./onesFilt; 