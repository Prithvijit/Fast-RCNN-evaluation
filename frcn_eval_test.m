%fast RCNN performance Evaluation--on PASCAL VOC2007%
clc
addpath('/home/prithv1/fast-rcnn/matlab');
addpath('/home/prithv1/VOCdevkit');
addpath('/home/prithv1/VOCdevkit/VOCcode');
%addpath('--add all your required paths here---');
addpath('/home/prithv1/fast-rcnn/lib/datasets/VOCdevkit-matlab-wrapper');
% initialize VOC options
VOCinit;

[folder, name, ext] = fileparts('/home/prithv1/fast-rcnn/matlab/frcn_eval_test.m');

caffe_path = fullfile(folder, '..', 'caffe-fast-rcnn', 'matlab', 'caffe');
addpath(caffe_path);

use_gpu = true;
% You can try other models here:
def = fullfile(folder, '..', 'models', 'VGG16', 'test.prototxt');;
net = fullfile(folder, '..', 'data', 'fast_rcnn_models', ...
               'vgg16_fast_rcnn_iter_40000.caffemodel');
model = fast_rcnn_load_net(def, net, use_gpu);

%--perform detection and store results
%input image
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
%D=dir('/home/prithv1/VOCdevkit/VOC2007/JPEGImages/*.jpg');
%imcell=cell(1,numel(D));
namecell=cell(1,length(ids));
for i=1:length(ids)
        namecell{i}=ids{i};
        %namecell{i}=strtok(namecell{i},'.');
        %D(i).name=strcat('/home/prithv1/VOCdevkit/VOC2007/JPEGImages/',D(i).name);
        %imcell{i}=imread(D(i).name);
end
length(ids)
%input boxes
ld=load('/home/prithv1/fast-rcnn/data/selective_search_data/voc_2007_test.mat');
cls_inds=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
cls_names={'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','plant','sheep','sofa','train','tvmonitor'};
%detection

THRESH=0.5;
detect=cell(1,length(ids));
for i=1:length(ids)
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        box=ld.boxes{i};
        box=single(box)+1;
        dets=fast_rcnn_im_detect(model,I,box);
        detect{i}=dets;
        clear box;
end

%for b=1:30
%for c=1:20
%[x,y]=size(detect{b}{c});
%for b=1:30
%for c=1:20
%[x,y]=size(detect{b}{c});
%x
%y
%end
%end
for a=1:VOCopts.nclasses
    cls{a}=VOCopts.classes{a};
    nama=strcat(cls{a},'.txt');
    nama=strcat('/home/prithv1/VOCdevkit/results/VOC2007/Main/comp3_det_test_',nama);
    fid=fopen(nama,'w');
    for j=1:length(ids)
        [x,y]=size(detect{j}{a});
        x
        y
        for k=1:x
            %----write in the file----%
            %if (detect{j}{a}(k,end)>=0.5)
            fprintf(fid,'%s %f %d %d %d %d\n',namecell{j},detect{j}{a}(k,end),int16(detect{j}{a}(k,1:end-1)));
            %end
        end
    end
    fclose(fid);
end





output_dir=['/home/prithv1/results'];
path=['/home/prithv1/VOCdevkit/'];
tst_set='test';
res1=voc_eval(path,'comp3',tst_set,output_dir,true);
