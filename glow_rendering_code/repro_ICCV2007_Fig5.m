clc; clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reproduce Fig. 5 in ICCV 2007 paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% q is forward scattering parameter
% 0.0-0.2 air
% 0.2-0.7 aerosol
% 0.7-0.8 haze
% 0.8-0.85 mist
% 0.85-0.9 fog
% 0.9-1.0 rain
q = 0.2;

% T is optical thickness 
% According to Narasimhan in CVPR03 paper, 
% T = sigma * R (extinctiontion coefficition * distance or depth)
% which is the same \beta d in haze modeling
T = 1.2;    % possibly [0.7 1.2 4]

% k is conversion parameter from original solution to approximate kernel
k = 0.5;



sprow = 2; spcol = 4;
fig = figure('Position',[9   722   908   394]);

% clean_path = './img/ICCV2007/5a.bmp';
clean_path = './img/ICCV2007/0009.png';

clean_img = im2double(imread(clean_path));
[R,C,Z] = size(clean_img);

subplot(sprow,spcol,1), imshow(clean_img); title('clean');


%%
% q_cand = [0.68,0.5,0.3];
q_cand = [0.70,0.55,0.3];

for qidx=1:length(q_cand)
    q = q_cand(qidx);
    
    %%

    % Approximated convolution kernel for APSF
    % S. Metari and F. Deschenes, 
    % A New Convolution Kernel for Atmospheric Point Spread Function Applied to Computer Vision, ICCV 2007

    p = k*T;            % eq (9)
    sigma = (1-q)/q;    % eq (1)
    mu = 0; % assuming mean is equal to zero
    A = @(p,sigma) sqrt(sigma.^2 * gamma(1/p) / gamma(3/p));    % scale parameter

    x = -6:12/100:6;
        
    [XX,YY] = meshgrid(x,x);
    APSF2D = exp(-(XX.^2+YY.^2).^(p/2)/abs(A(p,sigma))^p) / (2*gamma(1+1/p)*A(p,sigma))^2;
    APSF2D = APSF2D/sum(APSF2D(:));
    
    img = imfilter(clean_img,APSF2D);
    
    %%
    subplot(sprow,spcol,qidx+1);
    imshow(img);
    title(sprintf('T=%2.1f q=%3.2f',T,q));
    
    subplot(sprow,spcol,spcol+qidx+1);
    plot(x,APSF2D(:,round(size(APSF2D,1)/2))/sum(APSF2D(:)),'linewidth',1.5);
    
end


