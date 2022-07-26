% Script to check how to merge the three channels of an image
% to get the "best" possible information
clearvars;
close all;
addpath(genpath('./Exposure_Fusion/'));

% Read the image
imgname = 'DSC01607';
img    = im2double(imread('./results_VGGfeatures/DSC01607_input.jpg'));
img_r = img(:, :, 1);
img_g = img(:, :, 2);
img_b = img(:, :, 3);
        
% Crit-2 (using Merten's fusion approach)
img_stack = zeros(size(img_r,1), size(img_r, 2), 3, 3);
img_stack(:, :, :, 1) = cat3(img_r);
img_stack(:, :, :, 2) = cat3(img_g);
img_stack(:, :, :, 3) = cat3(img_b);

[img_gray_weights_2, img_gray_best2_2]   = exposure_fusion(img_stack,[0 1 0]);
figure(1), imshow([img, img_gray_best2_2]);
% figure(2), imshow([(img_gray_weights_1(:, :, 1)), ...
%                    (img_gray_weights_1(:, :, 2)), ...
%                    (img_gray_weights_1(:, :, 3))]);
% imwrite(img_gray_best2_2, ['./results_VGGfeatures/', imgname, '_GrayBest.png']);
% imwrite([img, img_gray_best2_2], ['./results_VGGfeatures/', imgname, '_I_GrayBest.png']);

function y = cat3(x)
    y = cat(3, x, x, x);
end

function y = norm(x)
    max_x = max(x(:));
    min_x = min(x(:));
    y = x - min_x/(max_x - min_x);
end

