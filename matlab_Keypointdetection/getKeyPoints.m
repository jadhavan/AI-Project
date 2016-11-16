% This function takes an image file and returns its keypoint data. 
% This is adapted from demo.m 
function [kpx, kpy, kpname] = getKeyPoints(filename)

% Update these according to your requirements
USE_GPU = 0; % 1 for GPU
img_fn = filename; %sample_img.png

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

%close all; clear; clc
end