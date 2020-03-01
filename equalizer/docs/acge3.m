function [numsopt,densopt] = acge3(Gdb)
% acge3.m
% 
% Design third-octave EQ according to the method presented by J. Liski and
% V. Valimaki in "The Quest for the Best Graphic Equalizer," in Proc. 
% DAFx-17, Edinburgh, UK, Sep. 2017.
% 
% Input parameters:
% Gdb  = command gains in dB, size 31x1
% 
% Output:
% numsopt = numerator parts of the 31 filters
% densopt = denominator parts of the 31 filters
%
% Uses pareq.m and interactionMatrix.m
%
% Created by Juho Liski and Vesa Valimaki, Otaniemi, Espoo, Finland, 21 June 2017
% Modified by Juho Liski, Otaniemi, Espoo, Finland, 6 May 2019
%
% Aalto University, Dept. of Signal Processing and Acoustics

fs  = 44.1e3;  % Sample rate
fc1 = [19.69,24.80,31.25,39.37,49.61,62.50,78.75,99.21,125.0,157.5,198.4, ...
    250.0,315.0,396.9,500.0,630.0,793.7,1000,1260,1587,2000,2520,3175,4000, ...
    5040,6350,8000,10080,12700,16000,20160]; % Log center frequencies for filters
fc2 = zeros(1,61); % Center frequencies and intermediate points between them
fc2(1:2:61) = fc1;
for k = 2:2:61
    fc2(k) = sqrt(fc2(k-1)*fc2(k+1));  % Extra points are at geometric mean frequencies
end
wg = 2*pi*fc1/fs;  % Command gain frequencies in radians
wc = 2*pi*fc2/fs;  % Center frequencies in radians for iterative design with extra points
gw = 0.4; % Gain factor at bandwidth (parameter c)
bw = 2*pi/fs*[9.178 11.56 14.57 18.36 23.13 29.14 36.71 46.25 58.28 73.43 ...
    92.51 116.6 146.9 185.0 233.1 293.7 370.0 466.2 587.4 740.1 932.4 ...
    1175 1480 1865 2350 2846 3502 4253 5038 5689 5570]; % EQ filter bandwidths

leak = interactionMatrix(10^(17/20)*ones(1,31),gw,wg,wc,bw); % Estimate leakage b/w bands
Gdb2 = zeros(61,1);
Gdb2(1:2:61) = Gdb;
for k = 2:2:61
    Gdb2(k) = (Gdb2(k-1)+Gdb2(k+1))/2; % Interpolate target gains linearly b/w command gains
end
Goptdb = leak'\Gdb2;      % Solve first estimate of dB gains based on leakage
Gopt = 10.^(Goptdb/20);    % Convert to linear gain factors

% Iterate once
leak2 = interactionMatrix(Gopt,gw,wg,wc,bw); % Use previous gains
G2optdb = leak2'\Gdb2;     % Solve optimal dB gains based on leakage
G2opt = 10.^(G2optdb/20);   % Convert to linear gain factors
G2woptdb = gw*G2optdb;      % Gain at bandwidth wg
G2wopt = 10.^(G2woptdb/20); % Convert to linear gain factor

% Design filters with optimized gains
numsopt = zeros(3,31);  % 3 num coefficients for each 10 filters
densopt = zeros(3,31);  % 3 den coefficients for each 10 filters
for k = 1:31,
    [num,den] = pareq(G2opt(k), G2wopt(k), wg(k), bw(k)); % Design filters
    numsopt(:,k) = num;
    densopt(:,k) = den;
end

% %%% Evaluation and plotting of the frequency response
% Nfreq = 2^12;  % Number of frequency points for frequency response evaluation
% w = logspace(log10(9),log10(22050),Nfreq);  % Log frequency points
% % Evaluate frequency responses
% Hopt = ones(Nfreq,31);   % Frequency response of individual filters
% Hopttot = ones(Nfreq,1); % Frequency response of the cascade EQ
% for k = 1:31
%     Hopt(:,k) = freqz(numsopt(:,k),densopt(:,k),w,fs);
%     Hopttot = Hopt(:,k) .* Hopttot;
% end
% % Plot responses for the proposed optimized design
% figure(1); clf;
% semilogx(w,db(Hopttot),'k','linewidth',3); hold on % Total response
% plot(fc2,Gdb2,'ro','linewidth',2) % Command gains
% set(gca,'fontname','Times','fontsize',16);
% xlabel('Frequency (Hz)');ylabel('Magnitude (dB)')
% set(gca,'XTick',[10 30 100 300 1000 3000 10000],'YTick',-20:5:20)
% grid on
% axis([15 22050 -20 20])


