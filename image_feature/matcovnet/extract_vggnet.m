clear

data_folder = '/fishtank/urix/survey/';   % corresponds to SURVEY_DATA
ds = 'train10k';                          % dataset to be processed
%ds = 'train1m';
%ds = 'nuswide';
%ds = 'train10k';
%ds = 'train100k';
%ds = 'mirflickr08';
%ds = 'imagenet166';
relu = 1;                                 % Do relu after fc7

% Job splitting
this_part = 1;                            % Part to be processed. One could get this from command line or environment.
parts = 1;                                % Total parts.
gpuDevice(this_part);                     % Use this GPU id

cnn_nets_folder = 'cnn_models/';
data_output_folder = 'FeatureData';
img_folder = 'ImageData';
                                                                                
%dataset_path = [img_folder, ds, '/images/'];
                                                                                                    
%% --------------------SELECT PRETRAINED MODEL-------------------
%net_name = 'vgg-m-128';
%net_name = 'vgg-f';
%net_name = 'vgg-s';
%net_name = 'vgg-verydeep-16';
net_name = 'vgg-verydeep-16';

if strfind(net_name, 'verydeep-16')
    layer_selected = 35 + relu; %fc7 verydeep-16
    chunk_size = 40;
elseif strfind(net_name, 'verydeep-19')
    layer_selected = 41 + relu; %fc7 verydeep-19
    chunk_size = 20;    
else
    layer_selected = 19 + relu; %fc7
    chunk_size = 150;
end
if strfind(net_name, 'm-128')
    descrSize=128;
else
    descrSize=4096;
end

preTrainedModel=['imagenet-', net_name, '.mat'];
preTrainedModel = [cnn_nets_folder, preTrainedModel];

%% --------------------SELECT DATASET-------------------
%load file list
images_paths = importdata([data_folder, ds, '/', img_folder, '/', ds, '.txt']);
datasetSize=length(images_paths);
images_paths = cellfun(@(x) [data_folder, ds, '/', img_folder, '/', x], images_paths, 'UniformOutput', false);

if parts > 1
    part_size = floor(length(images_paths) / parts);
    if this_part == parts
        images_paths = images_paths((this_part-1) * part_size + 1 : end);
    else
        images_paths = images_paths((this_part-1) * part_size + 1 : (this_part) * part_size);
    end
end

fc7 = zeros(descrSize,length(images_paths),'single');

%% --------------------LOAD MODEL-------------------
run matconvnet-1.0-beta8/matlab/vl_setupnn
net = load(preTrainedModel);
net=vl_simplenn_move(net,'gpu');

%% --------------- NUMBER OF CORES IN THE MACHINE-----------------
numCores=12;

%% --------------- MEMORY TO USE IN GB-----------------
mem2use=5;
%% --------------- ESTIMATED IMAGE SIZE IN GB-----------------
imageSize=0.0011;
numImgsForTurn=floor(mem2use/imageSize);
numTurns=ceil(length(images_paths)/numImgsForTurn);

nextBatch=cellstr(images_paths(1:min(numImgsForTurn, length(images_paths))));
vl_imreadjpeg(nextBatch,'numThreads',numCores,'Prefetch');

failedToRead = {};
failed = [];
k=1;
for j=1:numTurns
    fprintf('%d TURN of %d\n*******************\n', j, numTurns);
  
    tic;
    images=vl_imreadjpeg(nextBatch,'numThreads',numCores);
    previousBatch = nextBatch;
    if j ~= numTurns
        if j==(numTurns-1)
           nextBatch=cellstr(images_paths(((j)*numImgsForTurn+1):end));
        else
           nextBatch=cellstr(images_paths(((j)*numImgsForTurn+1):numImgsForTurn*(j+1)));
        end
        vl_imreadjpeg(nextBatch,'numThreads',numCores,'Prefetch');
    end
    toc;
  
    n_chunks = ceil(length(images) / chunk_size);
    for i = 1 : n_chunks
        fprintf('%d chunk of %d\n', i, n_chunks);
        batch = zeros(net.normalization.imageSize(1), net.normalization.imageSize(2), 3, chunk_size, 'single');
        
        tic;
        start_idx = (i-1)*chunk_size+1;
        if i ~= n_chunks
            end_idx = i*chunk_size;
        else
            end_idx = length(images);
        end
        for k = start_idx : end_idx;
            im = images{k};
            if isempty(im)
                try
                    im = imread(previousBatch{k});
                catch
                    fprintf('failed to read: %s\n', previousBatch{k});
                    failedToRead = [failedToRead, previousBatch{k}];
                    failed = [failed, ((j-1) * numImgsForTurn) + start_idx + k];
                    continue
                end
            end
            if size(im,3)~=3
                im=gray2rgb(im);
            end            
            im_ = single(im) ; % note: 255 range
            im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
            im_ = im_ - net.normalization.averageImage ;           
            batch(:,:,:,k-start_idx+1) = im_;
        end
        toc;
        
        tic;            
        % run the CNN
        batch = gpuArray(batch);
        res = vl_simplenn(net, batch) ;
        feat_fc7 = squeeze(gather(res(layer_selected).x));

        fc7(:, ((j-1) * numImgsForTurn) + start_idx : ((j-1) * numImgsForTurn) + end_idx) = single(feat_fc7(:,1:end_idx-start_idx+1));
        toc;
    end

    clear images
end

if ~isempty(failed)
    fprintf('trimming failed images...\n');
    
    assert(sum(sum(fc7(:,failed))) == 0);
    fc7(:, failed) = [];
    
    datasetSize = datasetSize - length(failed);
    
    images_paths(failed) = [];
end


%% save data fc7

suffix_name = 'fc7';
if relu
    suffix_name = [suffix_name, 'relu'];
end

mkdir([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name]);

% binary data
if parts > 1
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/feature-', num2str(this_part), '.bin'], 'w');
else
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/feature.bin'], 'w');
end
fwrite(fid, fc7, 'single');
fclose(fid);

% shape.txt
if parts > 1
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/shape-', num2str(this_part), '.txt'], 'w');
else
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/shape.txt'], 'w');
end
fprintf(fid, '%d %d', datasetSize, size(fc7,1));
fclose(fid);

% minmax.txt
if parts > 1
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/minmax-', num2str(this_part), '.txt'], 'w');
else
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/minmax.txt'], 'w');
end
fprintf(fid, '%f ', min(fc7, [], 2));
fprintf(fid, '\n');
fprintf(fid, '%f ', max(fc7, [], 2));
fprintf(fid, '\n');
fclose(fid);

% id.txt
if parts > 1
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/id-', num2str(this_part), '.txt'], 'w');
else
    fid = fopen([data_folder, ds, '/', data_output_folder, '/', net_name, suffix_name, '/id.txt'], 'w');
end
for i = 1:length(images_paths)
    idfile = strsplit(images_paths{i}, '/');
    idfile = idfile{end};
    idfile = strsplit(idfile,'.');
    idfile = idfile{1};
        
    fprintf(fid, '%s ', idfile);
end
fclose(fid);
