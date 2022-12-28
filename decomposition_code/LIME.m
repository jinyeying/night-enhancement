function [I, T_ini,T_ref] = LIME(L,para)
    T_ini = max(L,[],3)+0.02;
    [wx, wy] = computeTextureWeights(T_ini, para.sigma);
    T_ref = solveLinearEquation(T_ini, wx, wy, para.lambda);
    hgamma = vision.GammaCorrector(1/para.gamma,'Correction','Gamma');
    T_ref = step(hgamma, T_ref);
    I(:,:,1) = L(:,:,1)./T_ref;
    I(:,:,2) = L(:,:,2)./T_ref;
    I(:,:,3) = L(:,:,3)./T_ref;
end

function [retx, rety] = computeTextureWeights(fin, sigma)
   fx = diff(fin,1,2);
   fx = padarray(fx, [0 1 0], 'post');
   fy = diff(fin,1,1);
   fy = padarray(fy, [1 0 0], 'post');
   
   vareps_s = 0.02;
   vareps = 0.001;
   wto = max(sum(sqrt(fx.^2+fy.^2),3)/size(fin,3),vareps_s).^(-1); 
   fbin = lpfilter(fin, sigma);
   gfx = diff(fbin,1,2);
   gfx = padarray(gfx, [0 1], 'post');
   gfy = diff(fbin,1,1);
   gfy = padarray(gfy, [1 0], 'post');     
   wtbx = max(sum(abs(gfx),3)/size(fin,3),vareps).^(-1); 
   wtby = max(sum(abs(gfy),3)/size(fin,3),vareps).^(-1);   
   retx = wtbx.*wto;
   rety = wtby.*wto;
   retx(:,end) = 0;
   rety(end,:) = 0;
end

function ret = conv2_sep(im, sigma)
  ksize = bitor(round(5*sigma),1);
  g = fspecial('gaussian', [1,ksize], sigma); 
  ret = conv2(im,g,'same');
  ret = conv2(ret,g','same');  
end

function FBImg = lpfilter(FImg, sigma)     
    FBImg = FImg;
    for ic = 1:size(FBImg,3)
        FBImg(:,:,ic) = conv2_sep(FImg(:,:,ic), sigma);
    end   
end
function OUT = solveLinearEquation(IN, wx, wy, lambda)
% 
% The code for constructing inhomogenious Laplacian is adapted from 
% the implementaion of the wlsFilter. 
% 
% For color images, we enforce wx and wy be same for three channels
% and thus the pre-conditionar only need to be computed once. 
% 
    [r,c,ch] = size(IN);
    k = r*c;
    dx = -lambda*wx(:);
    dy = -lambda*wy(:);
    B(:,1) = dx;
    B(:,2) = dy;
    d = [-r,-1];
    A = spdiags(B,d,k,k);
    e = dx;
    w = padarray(dx, r, 'pre'); w = w(1:end-r);
    s = dy;
    n = padarray(dy, 1, 'pre'); n = n(1:end-1);
    D = 1-(e+w+s+n);
    A = A + A' + spdiags(D, 0, k, k); 
    if exist('ichol','builtin')
        L = ichol(A,struct('michol','on'));    
        OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            [tout, flag] = pcg(A, tin(:),0.1,100, L, L'); 
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
    else
        OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            tout = A\tin(:);
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
    end   
end
