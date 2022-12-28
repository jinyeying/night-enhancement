clc;close all;clear all;
save_dirG = './results_G/';
save_dirJ = './results_J/';
if ~exist([save_dirG],'dir')
	mkdir([save_dirG])
end
if ~exist([save_dirJ],'dir')
	mkdir([save_dirJ])
end
imgname = 'DSC00637'
% imgname = 'DSC01257_input'
% imgname = 'DSC01607_input'
% imgname = 'g'
input_i = im2double(imread(['.\light-effects\' imgname '.JPG']));
[H W D] = size(input_i);
[J g] = sept(input_i, 50, zeros(H,W,D), input_i);
[j G] = sept(input_i, 25000, zeros(H,W,D), input_i);
if(strcmp(imgname, 'g'))
    [j G] = septRelSmo(input_i, 500, zeros(H,W,D), input_i);
end

post = false; % Three layers?
%post = true
if post
    para.lambda = .15; 
    para.sigma = 2; 
    para.gamma = 0.8;
    para.solver = 1; 
    para.strategy = 3;
    [I,~,L] = LIME(J*2,para);
    L = cat(3,L,L,L); 
    figure(3);imshow([input_i G*2 J*2 L]);
end

figure(1),
imshow([input_i G*2 J*2]);
%imwrite(G*2,  [save_dirG imgname '_G.jpg']);
%imwrite(J*2,  [save_dirJ imgname '_J.jpg']);
