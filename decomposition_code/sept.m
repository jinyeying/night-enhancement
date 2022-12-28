
function [L1 L2] = septRelSmo(I,lambda,lb,hb,L1_0)
% I: input
% lambda: smoothness regularization parameter on the smoother layer Layer 2 
% lb: lower bound of the Layer 1,need to be same dimention with input I 
% hb: upper bound of the Layer 1,need to be same dimention with input I
% L1_0: initialization of Layer 1, default as the input I

if ~exist('I0','var')
    L1_0 = I;
end

[N,M,D] = size(I);

% filters
f1 = [1, -1];
f2 = [1; -1];
f3 = [0, -1, 0; 
      -1, 4, -1;
      0, -1, 0];

sizeI2D = [N,M];
otfFx = psf2otf(f1,sizeI2D);
otfFy = psf2otf(f2,sizeI2D);
otfL = psf2otf(f3,sizeI2D);

Normin1 = repmat(abs(otfL),[1,1,D]).^2.*fft2(I);
Denormin1 = abs(otfL).^2 ;
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;

if D>1
    Denormin1 = repmat(Denormin1,[1,1,D]);
    Denormin2 = repmat(Denormin2,[1,1,D]);
end

eps = 1e-16;
L1 = L1_0;

thr = 0.05;

for i = 1:3
    beta = 2^(i-1)/thr;
    Denormin   = lambda*Denormin1 + beta*Denormin2;
    gFx = -imfilter(L1,f1,'circular');
    gFy = -imfilter(L1,f2,'circular');
    t = repmat(sum(abs(gFx),3)<1/beta,[1 1 D]);
    gFx(t) = 0;
    t = repmat(sum(abs(gFy),3)<1/beta,[1 1 D]);
    gFy(t) = 0;
    %% compute L1
    Normin2 = [gFx(:,end,:) - gFx(:, 1,:), -diff(gFx,1,2)];
    Normin2 = Normin2 + [gFy(end,:,:) - gFy(1, :,:); -diff(gFy,1,1)];
    FL1 = (lambda*Normin1 + beta*fft2(Normin2))./(Denormin+eps);
    L1 = real(ifft2(FL1));
    %% normalize L1
    for c = 1:D
        L1t = L1(:,:,c);
        for k = 1:500
        dt = (sum(L1t(L1t<lb(:,:,c)) )+ sum(L1t(L1t>hb(:,:,c)) ))*2/numel(L1t);
        L1t = L1t-dt;
        if abs(dt)<1/numel(L1t) 
            break; 
        end
        end
        L1(:,:,c) = L1t;
    end
    t = L1<lb;
    L1(t) = lb(t);
    t = L1>hb;
    L1(t) = hb(t);
    %% L2
    L2 = I-L1;
end

