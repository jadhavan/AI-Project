% Demo for keypoint detection
%
% Aug. 23, 2016
%
% For further details, visit http://www.robots.ox.ac.uk/~vgg/software/keypoint_detection/

close all; clear; clc

% Update these according to your requirements
USE_GPU = 0; % 1 for GPU
img_fn = 'sample_img.png';

DEMO_BASEDIR = pwd;
DEMO_MODEL_FN = fullfile(DEMO_BASEDIR,'data','keypoint-v2.mat');
MATCONVNET_DIR = fullfile(DEMO_BASEDIR, 'lib', 'matconvnet-custom');

%
% Compile matconvnet
% http://www.vlfeat.org/matconvnet/install/
%
if ~exist( fullfile(MATCONVNET_DIR, 'matlab', 'mex'), 'dir' )
  disp('Compiling matconvnet ...')
  addpath('./lib/matconvnet-custom/matlab');
  if ( USE_GPU )
    vl_compilenn('enableGpu', true);
  else
    vl_compilenn('enableGpu', false);
  end
  fprintf(1, '\n\nMatcovnet compilation finished.');
end

% setup matconvnet path variables
matconvnet_setup_fn = fullfile(MATCONVNET_DIR, 'matlab', 'vl_setupnn.m');
run(matconvnet_setup_fn) ;

% Initialize keypoint detector
keypoint_detector = KeyPointDetector(DEMO_MODEL_FN, MATCONVNET_DIR, USE_GPU);

% Detect keypoints
fprintf(1, '\nDetecting keypoints in image : %s', img_fn);
[kpx, kpy, kpname] = get_all_keypoints(keypoint_detector, img_fn);

% Display the keypoints
img = imread(img_fn);
figure('Name', 'Detected Keypoints');
imshow(img); hold on;
plot(kpx, kpy, 'r.', 'MarkerSize', round(size(img,2)/10)); hold on;

voffset = -10;
for i=1:length(kpname)
  text(double(kpx(i)), double(kpy(i) + voffset), ...
    kpname{i}, 'color', 'yellow', 'FontSize', 8, ...
    'backgroundcolor', 'black'); 
  hold on;
  voffset = voffset * -1; % to prevent cluttering of annotations
end
hold off;

fprintf(1, '\nShowing detected keypoints:');
for i=1:length(kpname)
  fprintf(1, '\n%s\tat\t(%d,%d)', kpname{i}, kpx(i), kpy(i));
end
fprintf(1, '\n');