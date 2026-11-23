%% ========================================================================
%  Project: Automated Quantification of Myocardial Infarct Size
%  Course:  EGR 532 - Biomedical Imaging and Image Processing (Class Project #1)
%  Author:  Maaz Bin Fazal
%  Instructor: Dr. Samhita Rhodes
%  Institution: Grand Valley State University
%  Date:    November 19, 2026
%  ========================================================================

clear; clc; close all;
fprintf('\n=== AUTOMATED TTC HEART INFARCT ANALYSIS ===\n');

%% STAGE 0 - CONFIGURATION
cfg.projectRoot = fileparts(fileparts(mfilename('fullpath')));
cfg.trainingFolder = '';   % Optional: enter full Training folder path
cfg.testFolder = '';       % Optional: enter full Test folder path

cfg.outputFolderName = ['Infarct_Results_' datestr(now,'yyyymmdd_HHMMSS')];
cfg.heartMaskMethod = 'backgroundDistance'; % or 'saturation'
cfg.backgroundDistanceMin = 0.06;
cfg.heartClosingRadius = 5;
cfg.minimumHeartAreaFraction = 0.03;
cfg.minimumHeartComponentPixels = 100;
cfg.secondaryComponentFraction = 0.02;

% Optional score preprocessing. Keep off initially to preserve color data.
cfg.useMedianFilterForScores = false;
cfg.medianWindow = [3 3];
cfg.useGaussianSmoothing = false;
cfg.gaussianSigma = 0.8;

% Experiments: four color spaces x two automatic threshold methods.
cfg.runOtsu = true;
cfg.runAdaptive = true;
cfg.runFixedThreshold = false; % Enable only after training calibration.
cfg.adaptiveSensitivity = 0.50;
% Provisional starting values ONLY; not validated for this dataset.
cfg.fixedThresholds = [0.55 0.45 0.65 0.60]; % RGB, HSV, Lab, Gray

% Morphological refinement of infarct candidates.
cfg.removeSmallInfarctObjects = true;
cfg.minimumInfarctObjectFraction = 0.0005;
cfg.infarctClosingRadius = 1;
cfg.infarctOpeningRadius = 0; % 0 disables opening.

cfg.saveMasks = true;
cfg.saveDiagnostics = true;
cfg.saveHistograms = true;
cfg.makeEdgeDiagnostics = true;
% Manual masks are loaded automatically by image basename from this folder.
cfg.manualTracingFolder = fullfile(cfg.projectRoot,'05_Manual_Tracing');
% Optional percentage-only fallback; pixel metrics require TIFF/PNG masks.
cfg.manualReferenceCSV = fullfile(cfg.projectRoot,'ManualReference.csv');

extensions = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'};

%% STAGE 1 - FIND TRAINING AND TEST DATA
fprintf('\nSTAGE 1: Locating datasets...\n');
if isempty(cfg.trainingFolder) || ~isfolder(cfg.trainingFolder)
    cfg.trainingFolder = findFolder(cfg.projectRoot,{'Training','Training Data','Train'});
end
if isempty(cfg.testFolder) || ~isfolder(cfg.testFolder)
    cfg.testFolder = findFolder(cfg.projectRoot,{'Test','Testing','Test Data'});
end
if isempty(cfg.trainingFolder)
    p = uigetdir(cfg.projectRoot,'Select TRAINING dataset folder');
    if isequal(p,0), error('Training folder not selected. Set cfg.trainingFolder.'); end
    cfg.trainingFolder = p;
end
if isempty(cfg.testFolder)
    p = uigetdir(cfg.projectRoot,'Select TEST dataset folder');
    if isequal(p,0), error('Test folder not selected. Set cfg.testFolder.'); end
    cfg.testFolder = p;
end
fprintf('Training: %s\nTest: %s\n',cfg.trainingFolder,cfg.testFolder);

%% STAGE 2 - OUTPUT FOLDERS AND FILE INVENTORY

fprintf('\nSTAGE 2: Building file inventory...\n');

outRoot = fullfile(cfg.projectRoot,cfg.outputFolderName);

mkdirIfNeeded(outRoot);
mkdirIfNeeded(fullfile(outRoot,'HeartMasks'));
mkdirIfNeeded(fullfile(outRoot,'InfarctMasks'));
mkdirIfNeeded(fullfile(outRoot,'Diagnostics'));
mkdirIfNeeded(fullfile(outRoot,'ManualValidation'));

trainFiles = listImages(cfg.trainingFolder,extensions);
testFiles = listImages(cfg.testFolder,extensions);

fprintf('Training images: %d | Test images: %d\n', ...
    numel(trainFiles),numel(testFiles));

if isempty(trainFiles) && isempty(testFiles)
    error('No supported images found. Check the selected folders.');
end
%% STAGE 3 - EXPERIMENT DESIGN
colorNames = {'RGB','HSV','CIELAB','Grayscale'};
thresholdNames = {};
if cfg.runOtsu, thresholdNames{end+1} = 'Otsu'; end %#ok<SAGROW>
if cfg.runAdaptive, thresholdNames{end+1} = 'Adaptive'; end %#ok<SAGROW>
if cfg.runFixedThreshold
    thresholdNames{end+1} = 'Fixed'; %#ok<SAGROW>
    warning(['Fixed thresholds are provisional. Calibrate them using ' ...
        'training references before final test evaluation.']);
end
if isempty(thresholdNames), error('Enable at least one threshold method.'); end
fprintf('\nSTAGE 3: Color spaces = RGB, HSV, CIELAB, Grayscale\n');
fprintf('Threshold methods = %s\n',strjoin(thresholdNames,', '));

%% STAGE 4 - AUTOMATIC IMAGE PROCESSING
% Each image is processed automatically; no drawing or per-image clicks.
fprintf('\nSTAGE 4: Processing images...\n');
allRows = struct([]);
featureRows = struct([]);

sets = {'Training',trainFiles; 'Test',testFiles};
for ds = 1:size(sets,1)
    datasetName = sets{ds,1};
    files = sets{ds,2};
    fprintf('\n--- %s (%d images) ---\n',datasetName,numel(files));

    for k = 1:numel(files)
        imagePath = files{k};
        [~,baseName,ext] = fileparts(imagePath);
        fileName = [baseName ext];
        fprintf('Image %d/%d: %s\n',k,numel(files),fileName);

        try
            %% STAGE 4A - LOAD AND STANDARDIZE IMAGE
            raw = imread(imagePath);
            I = toRGBDouble(raw);
            [nr,nc,~] = size(I);
            if nr < 20 || nc < 20
                warning('Image too small; skipped: %s',fileName);
                continue;
            end

            %% STAGE 4B - PREPROCESSING / QUALITY-CONTROL IMAGES
            gray = rgb2gray(I);
            medianGray = medfilt2(gray,cfg.medianWindow);
            if exist('imgaussfilt','file') == 2
                gaussianGray = imgaussfilt(gray,cfg.gaussianSigma);
            else
                gaussianGray = gray;
            end
            if exist('imsharpen','file') == 2
                sharpenedGray = imsharpen(gray);
            else
                sharpenedGray = gray;
            end

            %% STAGE 4C - AUTOMATIC HEART EXTRACTION
            [heartMask,maskInfo] = makeHeartMask(I,cfg);
            tissuePixels = nnz(heartMask);
            maskPercent = 100*tissuePixels/numel(heartMask);
            if tissuePixels == 0 || maskPercent < 100*cfg.minimumHeartAreaFraction ...
                    || maskPercent > 95
                warning('Check heart mask for %s: %.1f%% of image.',fileName,maskPercent);
            end

            %% STAGE 4D - LOAD MANUAL REFERENCE MASKS
            % Manual masks are matched by the original image basename.
            % Expected location: 05_Manual_Tracing/<baseName>/
            manualRef = loadManualReferenceMasks( ...
                cfg.manualTracingFolder,baseName,nr,nc);

            manualHeartMask = manualRef.heartMask;
            manualInfarctMask = manualRef.infarctMask;
            manualReferenceFound = ~isempty(manualHeartMask) && ...
                ~isempty(manualInfarctMask);

            manualHeartPixels = NaN;
            manualInfarctPixels = NaN;
            manualInfarctPercent = NaN;
            manualInfarctWeight = NaN;
            manualHeartPercentOfImage = NaN;
            manualOutsidePixels = manualRef.outsideInfarctPixels;

            heartMetrics = emptyBinaryMetrics();
            if ~isempty(manualHeartMask)
                manualHeartPixels = nnz(manualHeartMask);
                manualHeartPercentOfImage = ...
                    100*manualHeartPixels/numel(manualHeartMask);
                heartMetrics = binaryMetrics(heartMask,manualHeartMask);
            end

            if ~isempty(manualHeartMask) && ~isempty(manualInfarctMask)
                manualInfarctPixels = nnz(manualInfarctMask);
                manualInfarctPercent = ...
                    100*manualInfarctPixels/max(manualHeartPixels,1);
                sliceWeightForManual = weightFromName(fileName);
                if isfinite(sliceWeightForManual)
                    manualInfarctWeight = ...
                        (manualInfarctPercent/100)*sliceWeightForManual;
                end
                fprintf('  Manual masks found: %s | Manual infarct %.3f%%\n', ...
                    baseName,manualInfarctPercent);
                if manualOutsidePixels > 0
                    fprintf('  Note: %d manual infarct pixels were outside the manual heart and are excluded from infarct scoring.\n', ...
                        manualOutsidePixels);
                end
            end

            %% STAGE 4E - COLOR SPACE CONVERSION AND FEATURES
            [scores,features] = makeColorScores(I,heartMask);

            %% STAGE 4F - THRESHOLDING AND MORPHOLOGICAL REFINEMENT
            imageRows = struct([]);
            masks = struct();
            thresholds = struct();

            for c = 1:numel(colorNames)
                cname = colorNames{c};
                score = scores.(cname);
                if cfg.useMedianFilterForScores
                    score = medfilt2(score,cfg.medianWindow);
                end
                if cfg.useGaussianSmoothing && exist('imgaussfilt','file') == 2
                    score = imgaussfilt(score,cfg.gaussianSigma);
                end
                score = min(max(score,0),1);

                for t = 1:numel(thresholdNames)
                    tname = thresholdNames{t};
                    thresholdValue = NaN;

                    switch tname
                        case 'Otsu'
                            vals = score(heartMask);
                            vals = vals(isfinite(vals));
                            if isempty(vals), thresholdValue = 1;
                            else, thresholdValue = graythresh(vals);
                            end
                            candidate = score > thresholdValue;

                        case 'Adaptive'
                            if exist('adaptthresh','file') ~= 2
                                warning('adaptthresh unavailable; skipping adaptive method.');
                                continue;
                            end
                            localT = adaptthresh(score,cfg.adaptiveSensitivity, ...
                                'ForegroundPolarity','bright');
                            candidate = score > localT;
                            thresholdValue = cfg.adaptiveSensitivity;

                        case 'Fixed'
                            thresholdValue = cfg.fixedThresholds(c);
                            candidate = score > thresholdValue;
                    end

                    candidate = logical(candidate) & heartMask;
                    infarctMask = cleanInfarctMask(candidate,heartMask,cfg);
                    infarctPixels = nnz(infarctMask);
                    infarctFraction = infarctPixels/max(tissuePixels,1);
                    infarctPercent = 100*infarctFraction;
                    sliceWeight = weightFromName(fileName);
                    if isfinite(sliceWeight)
                        infarctWeight = infarctFraction*sliceWeight;
                    else
                        infarctWeight = NaN;
                    end

                    % For a fair percentage comparison, calculate the
                    % automatic infarct fraction inside the SAME manually
                    % outlined heart ROI used for the reference percentage.
                    infarctPercentWithinManualHeart = NaN;
                    infarctWeightWithinManualHeart = NaN;
                    automaticInfarctPixelsWithinManualHeart = NaN;
                    if ~isempty(manualHeartMask) && manualHeartPixels > 0
                        automaticInfarctPixelsWithinManualHeart = ...
                            nnz(infarctMask & manualHeartMask);
                        infarctPercentWithinManualHeart = ...
                            100*automaticInfarctPixelsWithinManualHeart/manualHeartPixels;
                        if isfinite(sliceWeight)
                            infarctWeightWithinManualHeart = ...
                                (infarctPercentWithinManualHeart/100)*sliceWeight;
                        end
                    end

                    % Pixel-level infarct metrics are also calculated only
                    % inside the manually outlined heart ROI.
                    infarctMetrics = emptyBinaryMetrics();
                    if ~isempty(manualHeartMask) && ~isempty(manualInfarctMask)
                        infarctMetrics = binaryMetrics( ...
                            infarctMask,manualInfarctMask,manualHeartMask);
                    end

                    signedError = NaN;
                    absoluteError = NaN;
                    if isfinite(manualInfarctPercent) && ...
                            isfinite(infarctPercentWithinManualHeart)
                        signedError = infarctPercentWithinManualHeart-manualInfarctPercent;
                        absoluteError = abs(signedError);
                    end

                    row = struct( ...
                        'Dataset',datasetName, ...
                        'Heart',heartID(imagePath,datasetName), ...
                        'Filename',fileName, ...
                        'FullPath',imagePath, ...
                        'SliceWeight_mg',sliceWeight, ...
                        'ColorSpace',cname, ...
                        'ThresholdMethod',tname, ...
                        'ThresholdValue',thresholdValue, ...
                        'ImageRows',nr, ...
                        'ImageColumns',nc, ...
                        'TissuePixels',tissuePixels, ...
                        'InfarctPixels',infarctPixels, ...
                        'InfarctFraction',infarctFraction, ...
                        'InfarctPercent',infarctPercent, ...
                        'EstimatedInfarctWeight_mg',infarctWeight, ...
                        'HeartMaskPercentOfImage',maskPercent, ...
                        'ManualReferenceFound',manualReferenceFound, ...
                        'ManualReferenceSource',manualRef.source, ...
                        'ManualHeartPixels',manualHeartPixels, ...
                        'ManualHeartPercentOfImage',manualHeartPercentOfImage, ...
                        'ManualInfarctPixels',manualInfarctPixels, ...
                        'ManualInfarctPercent',manualInfarctPercent, ...
                        'ManualInfarctWeight_mg',manualInfarctWeight, ...
                        'AutomaticInfarctPixelsWithinManualHeart',automaticInfarctPixelsWithinManualHeart, ...
                        'InfarctPercentWithinManualHeart',infarctPercentWithinManualHeart, ...
                        'EstimatedInfarctWeightWithinManualHeart_mg',infarctWeightWithinManualHeart, ...
                        'ManualInfarctOutsideHeartPixels',manualOutsidePixels, ...
                        'SignedError_percent',signedError, ...
                        'AbsoluteError_percent',absoluteError, ...
                        'HeartDice',heartMetrics.Dice, ...
                        'HeartIoU',heartMetrics.IoU, ...
                        'HeartPrecision',heartMetrics.Precision, ...
                        'HeartSensitivity',heartMetrics.Sensitivity, ...
                        'HeartSpecificity',heartMetrics.Specificity, ...
                        'HeartAccuracy',heartMetrics.Accuracy, ...
                        'InfarctDice',infarctMetrics.Dice, ...
                        'InfarctIoU',infarctMetrics.IoU, ...
                        'InfarctPrecision',infarctMetrics.Precision, ...
                        'InfarctSensitivity',infarctMetrics.Sensitivity, ...
                        'InfarctSpecificity',infarctMetrics.Specificity, ...
                        'InfarctAccuracy',infarctMetrics.Accuracy);

                    if isempty(imageRows), imageRows = row;
                    else, imageRows(end+1) = row; %#ok<SAGROW>
                    end

                    key = matlab.lang.makeValidName([cname '_' tname]);
                    masks.(key) = infarctMask;
                    thresholds.(key) = thresholdValue;

                    if cfg.saveMasks
                        md = fullfile(outRoot,'InfarctMasks',datasetName);
                        mkdirIfNeeded(md);
                        imwrite(uint8(infarctMask)*255,fullfile(md,[baseName '_' key '.png']));
                    end
                end
            end

            if ~isempty(imageRows)
                if isempty(allRows), allRows = imageRows;
                else, allRows = [allRows imageRows]; %#ok<AGROW>
                end
            end

            %% STAGE 4G - IMAGE FEATURE TABLE
            features.Dataset = datasetName;
            features.Heart = heartID(imagePath,datasetName);
            features.Filename = fileName;
            features.ImageRows = nr;
            features.ImageColumns = nc;
            features.TissuePixels = tissuePixels;
            features.HeartMaskPercentOfImage = maskPercent;
            features.HeartMaskMethod = maskInfo.method;
            features.BackgroundDistanceThreshold = maskInfo.threshold;
            if isempty(featureRows), featureRows = features;
            else, featureRows(end+1) = features; %#ok<AGROW>
            end

            %% STAGE 4H - SAVE MASKS AND VISUAL DIAGNOSTICS
            if cfg.saveMasks
                imwrite(uint8(heartMask)*255,fullfile(outRoot,'HeartMasks', ...
                    [datasetName '_' baseName '_HeartMask.png']));
            end
            if cfg.saveDiagnostics
                saveDiagnostics(I,gray,medianGray,gaussianGray,sharpenedGray, ...
                    heartMask,maskInfo,scores,masks,thresholds,datasetName,baseName,outRoot);
            end
            if cfg.makeEdgeDiagnostics
                saveEdges(gray,datasetName,baseName,outRoot);
            end
            if cfg.saveHistograms
                saveHistograms(scores,heartMask,datasetName,baseName,outRoot);
            end
            if ~isempty(manualHeartMask) || ~isempty(manualInfarctMask)
                saveManualReferenceDiagnostics(I,heartMask,manualHeartMask, ...
                    manualInfarctMask,datasetName,baseName,outRoot);
            end

        catch ME
            warning('FAILED: %s | %s',imagePath,ME.message);
            fid = fopen(fullfile(outRoot,'Processing_Errors.txt'),'a');
            if fid >= 0
                fprintf(fid,'\n%s\n%s\n',imagePath,ME.message);
                fclose(fid);
            end
        end
    end
end

if isempty(allRows)
    error('No images were successfully processed. See Processing_Errors.txt.');
end

%% STAGE 5 - BUILD TABLES AND MANUAL VALIDATION REFERENCES
fprintf('\nSTAGE 5: Building slice and feature tables...\n');
sliceTable = struct2table(allRows);
featureTable = struct2table(featureRows);

% Convert grouping columns to string for robust grouping in recent MATLAB.
sliceTable.Dataset = string(sliceTable.Dataset);
sliceTable.Heart = string(sliceTable.Heart);
sliceTable.ColorSpace = string(sliceTable.ColorSpace);
sliceTable.ThresholdMethod = string(sliceTable.ThresholdMethod);
sliceTable.ManualReferenceSource = string(sliceTable.ManualReferenceSource);
featureTable.Dataset = string(featureTable.Dataset);
featureTable.Heart = string(featureTable.Heart);
featureTable.HeartMaskMethod = string(featureTable.HeartMaskMethod);

% Manual TIFF masks are the primary reference. The optional CSV is used
% only to fill percentage values for images that do not have manual masks.
if isfile(cfg.manualReferenceCSV)
    try
        M = readtable(cfg.manualReferenceCSV);
        if all(ismember({'Filename','ManualInfarctPercent'},M.Properties.VariableNames))
            for r = 1:height(sliceTable)
                if ~isfinite(sliceTable.ManualInfarctPercent(r))
                    ix = strcmpi(string(M.Filename),string(sliceTable.Filename(r)));
                    if any(ix)
                        v = M.ManualInfarctPercent(find(ix,1,'first'));
                        if isfinite(v)
                            sliceTable.ManualInfarctPercent(r) = v;
                            sliceTable.ManualReferenceSource(r) = "CSV";
                            if isfinite(sliceTable.SliceWeight_mg(r))
                                sliceTable.ManualInfarctWeight_mg(r) = ...
                                    sliceTable.SliceWeight_mg(r)*v/100;
                            end
                        end
                    end
                end
            end
            fprintf('Optional ManualReference.csv fallback checked.\n');
        else
            warning('ManualReference.csv needs Filename and ManualInfarctPercent columns.');
        end
    catch ME
        warning('Manual reference CSV could not be read: %s',ME.message);
    end
end

automaticPercentForValidation = sliceTable.InfarctPercent;
hasCommonManualROI = isfinite(sliceTable.InfarctPercentWithinManualHeart);
automaticPercentForValidation(hasCommonManualROI) = ...
    sliceTable.InfarctPercentWithinManualHeart(hasCommonManualROI);
sliceTable.SignedError_percent = automaticPercentForValidation - ...
    sliceTable.ManualInfarctPercent;
sliceTable.AbsoluteError_percent = abs(sliceTable.SignedError_percent);

manualRows = isfinite(sliceTable.InfarctDice);
fprintf('Images with pixel-level manual masks: %d of %d processed images.\n', ...
    numel(unique(sliceTable.Filename(manualRows))), ...
    numel(unique(sliceTable.Filename)));
%% STAGE 6 - HEART-LEVEL WEIGHTED INFARCT QUANTIFICATION
fprintf('\nSTAGE 6: Calculating weighted heart-level infarct sizes...\n');
heartSummary = heartLevelSummary(sliceTable);

%% STAGE 7 - COLOR SPACE / THRESHOLD METHOD COMPARISON
fprintf('\nSTAGE 7: Calculating method comparison statistics...\n');
methodSummary = comparisonSummary(sliceTable);

%% STAGE 8 - EXCEL AND CSV EXPORT
fprintf('\nSTAGE 8: Exporting tables...\n');
xlsx = fullfile(outRoot,'Infarct_Analysis_Results.xlsx');
manualValidationTable = sliceTable(:,{ ...
    'Dataset','Heart','Filename','ColorSpace','ThresholdMethod', ...
    'InfarctPercent','AutomaticInfarctPixelsWithinManualHeart', ...
    'InfarctPercentWithinManualHeart','ManualHeartPixels', ...
    'ManualHeartPercentOfImage','ManualInfarctPixels','ManualInfarctPercent', ...
    'ManualInfarctWeight_mg','EstimatedInfarctWeightWithinManualHeart_mg', ...
    'SignedError_percent','AbsoluteError_percent','InfarctDice','InfarctIoU', ...
    'InfarctPrecision','InfarctSensitivity','InfarctSpecificity', ...
    'InfarctAccuracy','HeartDice','HeartIoU','HeartPrecision', ...
    'HeartSensitivity','HeartSpecificity','HeartAccuracy', ...
    'ManualReferenceFound','ManualReferenceSource', ...
    'ManualInfarctOutsideHeartPixels'});
try
    writetable(sliceTable,xlsx,'Sheet','SliceResults');
    writetable(heartSummary,xlsx,'Sheet','HeartSummary');
    writetable(methodSummary,xlsx,'Sheet','MethodComparison');
    writetable(featureTable,xlsx,'Sheet','ImageFeatures');
    writetable(manualValidationTable,xlsx,'Sheet','ManualValidation');
    fprintf('Workbook: %s\n',xlsx);
catch ME
    warning('Excel workbook export failed: %s',ME.message);
end
writetable(sliceTable,fullfile(outRoot,'SliceResults.csv'));
writetable(heartSummary,fullfile(outRoot,'HeartSummary.csv'));
writetable(methodSummary,fullfile(outRoot,'MethodComparison.csv'));
writetable(featureTable,fullfile(outRoot,'ImageFeatures.csv'));
writetable(manualValidationTable,fullfile(outRoot,'ManualValidationResults.csv'));

%% STAGE 9 - SUMMARY GRAPHS
fprintf('\nSTAGE 9: Saving summary graphs...\n');
saveSummaryGraphs(sliceTable,heartSummary,outRoot);
saveManualValidationGraphs(sliceTable,outRoot);

%% STAGE 10 - SAVE SETTINGS AND FINISH
fid = fopen(fullfile(outRoot,'Run_Configuration.txt'),'w');
if fid >= 0
    fprintf(fid,'Training folder: %s\n',cfg.trainingFolder);
    fprintf(fid,'Test folder: %s\n',cfg.testFolder);
    fprintf(fid,'Manual tracing folder: %s\n',cfg.manualTracingFolder);
    fprintf(fid,'Manual pixel-mask images found: %d\n',numel(unique(sliceTable.Filename(isfinite(sliceTable.InfarctDice)))));
    fprintf(fid,'Heart mask method: %s\n',cfg.heartMaskMethod);
    fprintf(fid,'Threshold methods: %s\n',strjoin(thresholdNames,', '));
    fprintf(fid,'Adaptive sensitivity: %.3f\n',cfg.adaptiveSensitivity);
    fprintf(fid,'Fixed threshold enabled: %d\n',cfg.runFixedThreshold);
    fprintf(fid,'Infarct closing radius: %d\n',cfg.infarctClosingRadius);
    fprintf(fid,'Small-object fraction: %.6f\n',cfg.minimumInfarctObjectFraction);
    fprintf(fid,'\nInspect heart masks and segmentation figures before reporting results.\n');
    fprintf(fid,'Tune using training data only; freeze settings before test evaluation.\n');
    fclose(fid);
end

fprintf('\n=== PROCESSING COMPLETE ===\n');
fprintf('Result rows: %d\n',height(sliceTable));
fprintf('All outputs saved in:\n%s\n',outRoot);
fprintf('\nInspect first: HeartMasks, InfarctMasks, Diagnostics, ManualValidation, then Excel.\n');
fprintf('Manual validation uses manual infarct masks clipped to the manual heart ROI.\n');
fprintf('Numerical results are provisional until segmentation masks are verified.\n');

%% =========================== LOCAL FUNCTIONS ===========================

function folder = findFolder(root,names)
folder = '';
for q = 1:numel(names)
    p = fullfile(root,names{q});
    if isfolder(p), folder = p; return; end
end
end

function mkdirIfNeeded(p)
if ~exist(p,'dir'), mkdir(p); end
end

function files = listImages(root,extensions)
files = {};
for q = 1:numel(extensions)
    a = dir(fullfile(root,'**',['*' extensions{q}]));
    b = dir(fullfile(root,'**',['*' upper(extensions{q})]));
    both = [a; b];
    for j = 1:numel(both)
        if ~both(j).isdir
            p = fullfile(both(j).folder,both(j).name);
            files{end+1,1} = p; %#ok<AGROW>
        end
    end
end
if ~isempty(files), files = unique(files,'stable'); end
end

function I = toRGBDouble(raw)
if ndims(raw) == 2
    raw = repmat(raw,1,1,3);
elseif size(raw,3) > 3
    raw = raw(:,:,1:3);
end
I = im2double(raw);
I = min(max(I,0),1);
end

function [mask,info] = makeHeartMask(I,cfg)
% Estimate background color from a border band in CIELAB.
% NEW: detect and exclude long dark horizontal/vertical background grid lines.

lab = rgb2lab(I);
[h,w,~] = size(lab);

%% 1) Detect black/dark ruler or grid lines in the photograph
% The supplied TTC images contain long, nearly straight black lines on the
% paper background. These are not heart tissue and must be excluded.
grayForLines = rgb2gray(I);
darkThreshold = 0.35;  % normalized intensity; dark grid lines are below this
lineLength = max(25,round(0.08*min(h,w)));
lineDilationRadius = 3;

darkPixels = grayForLines < darkThreshold;
horizontalLines = imopen(darkPixels,strel('line',lineLength,0));
verticalLines   = imopen(darkPixels,strel('line',lineLength,90));
gridLineMask = horizontalLines | verticalLines;
if any(gridLineMask(:))
    gridLineMask = imdilate(gridLineMask,strel('disk',lineDilationRadius,0));
end

%% 2) Estimate the paper/background color from the image border
band = max(2,round(0.04*min(h,w)));
border = false(h,w);
border(1:band,:) = true;
border(end-band+1:end,:) = true;
border(:,1:band) = true;
border(:,end-band+1:end) = true;

% Do not let detected dark grid lines bias the estimated background color.
backgroundPixels = border & ~gridLineMask;
if nnz(backgroundPixels) < 20
    backgroundPixels = border;
end
P = reshape(lab,[],3);
bg = median(P(backgroundPixels(:),:),1);
D = lab - reshape(bg,1,1,3);
distance = sqrt(sum(D.^2,3));
distance = mat2gray(distance);
distThreshold = max(cfg.backgroundDistanceMin,graythresh(distance));

%% 3) Initial heart candidate
switch lower(cfg.heartMaskMethod)
    case 'saturation'
        hsv = rgb2hsv(I);
        S = hsv(:,:,2);
        candidate = S > max(0.08,graythresh(S));
        method = 'saturation';
    otherwise
        candidate = distance > distThreshold;
        method = 'backgroundDistance';
end

% Remove grid-line pixels before and after closing so morphology cannot
% reconnect them to the heart candidate.
candidate = logical(candidate) & ~gridLineMask;
candidate = imclose(candidate,strel('disk',cfg.heartClosingRadius,0));
candidate = candidate & ~gridLineMask;
candidate = imfill(candidate,'holes');
candidate = candidate & ~gridLineMask;
candidate = bwareaopen(candidate,cfg.minimumHeartComponentPixels);

%% 4) Retain plausible connected heart components
cc = bwconncomp(candidate);
mask = false(size(candidate));
if cc.NumObjects > 0
    areas = cellfun(@numel,cc.PixelIdxList);
    largest = max(areas);
    minKeep = max(cfg.minimumHeartComponentPixels, ...
        cfg.secondaryComponentFraction*largest);
    for j = 1:cc.NumObjects
        if areas(j) >= minKeep
            mask(cc.PixelIdxList{j}) = true;
        end
    end
end

%% 5) Saturation fallback if the primary method found too little tissue
if nnz(mask) < cfg.minimumHeartAreaFraction*numel(mask)
    hsv = rgb2hsv(I);
    S = hsv(:,:,2);
    fallback = S > max(0.08,graythresh(S));
    fallback = fallback & ~gridLineMask;
    fallback = imclose(fallback,strel('disk',cfg.heartClosingRadius,0));
    fallback = fallback & ~gridLineMask;
    fallback = imfill(fallback,'holes');
    fallback = fallback & ~gridLineMask;
    fallback = bwareaopen(fallback,cfg.minimumHeartComponentPixels);
    cc = bwconncomp(fallback);
    if cc.NumObjects > 0
        areas = cellfun(@numel,cc.PixelIdxList);
        [~,ix] = max(areas);
        mask = false(size(fallback));
        mask(cc.PixelIdxList{ix}) = true;
        method = 'saturationFallback';
    end
end

%% 6) Final cleanup; explicitly exclude grid lines after hole filling
mask = imclose(mask,strel('disk',max(1,round(cfg.heartClosingRadius/2)),0));
mask = imfill(mask,'holes');
mask = mask & ~gridLineMask;
mask = bwareaopen(mask,cfg.minimumHeartComponentPixels);

% If cleanup split the mask, keep the largest remaining heart component.
cc = bwconncomp(mask);
if cc.NumObjects > 1
    areas = cellfun(@numel,cc.PixelIdxList);
    [~,ix] = max(areas);
    cleanMask = false(size(mask));
    cleanMask(cc.PixelIdxList{ix}) = true;
    mask = cleanMask;
end

info.method = method;
info.threshold = distThreshold;
info.distance = distance;
info.initialCandidate = candidate;
info.gridLineMask = gridLineMask;
info.backgroundLab = bg;
end

function [scores,F] = makeColorScores(I,mask)
R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
redRatio = R./(R+G+B+eps);
rgbScore = 1-redRatio;

hsv = rgb2hsv(I);
H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);
hsvScore = V.*(1-S);

lab = rgb2lab(I);
L = lab(:,:,1); a = lab(:,:,2); b = lab(:,:,3);
Lnorm = min(max(L/100,0),1);
redness = min(max(a/127,0),1);
labScore = 0.5*Lnorm + 0.5*(1-redness);

gray = rgb2gray(I);
scores.RGB = mat2gray(rgbScore);
scores.HSV = mat2gray(hsvScore);
scores.CIELAB = mat2gray(labScore);
scores.Grayscale = mat2gray(gray);

F.Mean_R = mean(R(mask),'omitnan');
F.Mean_G = mean(G(mask),'omitnan');
F.Mean_B = mean(B(mask),'omitnan');
F.Median_R = median(R(mask),'omitnan');
F.Median_G = median(G(mask),'omitnan');
F.Median_B = median(B(mask),'omitnan');
F.Std_R = std(R(mask),0,'omitnan');
F.Std_G = std(G(mask),0,'omitnan');
F.Std_B = std(B(mask),0,'omitnan');
F.Mean_H = mean(H(mask),'omitnan');
F.Mean_S = mean(S(mask),'omitnan');
F.Mean_V = mean(V(mask),'omitnan');
F.Mean_Lstar = mean(L(mask),'omitnan');
F.Mean_astar = mean(a(mask),'omitnan');
F.Mean_bstar = mean(b(mask),'omitnan');
F.Mean_Gray = mean(gray(mask),'omitnan');
F.Mean_NormalizedRedRatio = mean(redRatio(mask),'omitnan');
F.Mean_RGBScore = mean(scores.RGB(mask),'omitnan');
F.Mean_HSVScore = mean(scores.HSV(mask),'omitnan');
F.Mean_LabScore = mean(scores.CIELAB(mask),'omitnan');
F.Mean_GrayScore = mean(scores.Grayscale(mask),'omitnan');
end

function BW = cleanInfarctMask(candidate,heartMask,cfg)
BW = logical(candidate) & logical(heartMask);
if cfg.removeSmallInfarctObjects
    minArea = max(3,round(cfg.minimumInfarctObjectFraction*nnz(heartMask)));
    BW = bwareaopen(BW,minArea);
end
if cfg.infarctClosingRadius > 0
    BW = imclose(BW,strel('disk',cfg.infarctClosingRadius,0));
end
if cfg.infarctOpeningRadius > 0
    BW = imopen(BW,strel('disk',cfg.infarctOpeningRadius,0));
end
BW = BW & heartMask;
end

function manual = loadManualReferenceMasks(root,baseName,nr,nc)
% Locate masks in 05_Manual_Tracing/<image basename>/.
% Also accepts masks placed directly in 05_Manual_Tracing for compatibility.
manual.heartMask = [];
manual.infarctMask = [];
manual.heartPath = '';
manual.infarctPath = '';
manual.outsideInfarctPixels = NaN;
manual.source = 'None';

if isempty(root) || ~isfolder(root)
    return;
end

searchFolders = {fullfile(root,baseName),root};
heartSuffix = '_Manual_HeartMask';
infarctSuffix = '_Manual_InfarctMask';
extensions = {'.tif','.tiff','.png','.jpg','.jpeg','.bmp'};

heartPath = findManualMaskPath(searchFolders,baseName,heartSuffix,extensions);
infarctPath = findManualMaskPath(searchFolders,baseName,infarctSuffix,extensions);

if ~isempty(heartPath)
    manual.heartMask = readBinaryMask(heartPath,nr,nc);
    manual.heartPath = heartPath;
end
if ~isempty(infarctPath)
    manual.infarctMask = readBinaryMask(infarctPath,nr,nc);
    manual.infarctPath = infarctPath;
end

if ~isempty(manual.heartMask) && ~isempty(manual.infarctMask)
    manual.outsideInfarctPixels = nnz(manual.infarctMask & ~manual.heartMask);
    % Pixels outside the manually outlined heart are excluded from scoring.
    manual.infarctMask = manual.infarctMask & manual.heartMask;
    manual.source = 'Masks';
elseif ~isempty(manual.heartMask) || ~isempty(manual.infarctMask)
    manual.source = 'PartialMasks';
end
end

function pathFound = findManualMaskPath(folders,baseName,suffixes,extensions)
pathFound = '';
for f = 1:numel(folders)
    if ~isfolder(folders{f}), continue; end
    for e = 1:numel(extensions)
        candidate = fullfile(folders{f},[baseName suffixes extensions{e}]);
        if isfile(candidate)
            pathFound = candidate;
            return;
        end
        candidateUpper = fullfile(folders{f},[baseName suffixes upper(extensions{e})]);
        if isfile(candidateUpper)
            pathFound = candidateUpper;
            return;
        end
    end
end

% Last-resort recursive lookup if the user has an extra folder level.
if isfolder(folders{1})
    for e = 1:numel(extensions)
        hits = dir(fullfile(folders{1},'**',[baseName suffixes extensions{e}]));
        if ~isempty(hits)
            pathFound = fullfile(hits(1).folder,hits(1).name);
            return;
        end
    end
end
end

function M = readBinaryMask(pathName,nr,nc)
M = [];
try
    [A,map] = imread(pathName);
    if ~isempty(map)
        A = ind2gray(A,map);
    elseif ndims(A) == 3
        A = rgb2gray(A);
    end
    if size(A,1) ~= nr || size(A,2) ~= nc
        warning('Manual mask dimensions do not match source image; ignored: %s',pathName);
        return;
    end
    M = A > 0;
catch ME
    warning('Could not read manual mask %s: %s',pathName,ME.message);
end
end

function M = emptyBinaryMetrics()
M = struct('Dice',NaN,'IoU',NaN,'Precision',NaN,'Sensitivity',NaN, ...
    'Specificity',NaN,'Accuracy',NaN);
end

function M = binaryMetrics(pred,ref,roi)
M = emptyBinaryMetrics();
if nargin < 3 || isempty(roi)
    roi = true(size(pred));
end
if isempty(pred) || isempty(ref) || isempty(roi)
    return;
end
if ~isequal(size(pred),size(ref)) || ~isequal(size(pred),size(roi))
    return;
end

roi = logical(roi);
pred = logical(pred) & roi;
ref = logical(ref) & roi;
TP = nnz(pred & ref);
FP = nnz(pred & ~ref & roi);
FN = nnz(~pred & ref & roi);
TN = nnz(~pred & ~ref & roi);

M.Dice = safeDivide(2*TP,2*TP+FP+FN);
M.IoU = safeDivide(TP,TP+FP+FN);
M.Precision = safeDivide(TP,TP+FP);
M.Sensitivity = safeDivide(TP,TP+FN);
M.Specificity = safeDivide(TN,TN+FP);
M.Accuracy = safeDivide(TP+TN,TP+TN+FP+FN);
end

function value = safeDivide(a,b)
if b == 0
    value = NaN;
else
    value = a/b;
end
end

function w = weightFromName(name)
tok = regexp(name,'IS([0-9]+(?:\.[0-9]+)?)mg','tokens','once','ignorecase');
if isempty(tok), w = NaN; else, w = str2double(tok{1}); end
end

function h = heartID(path,dataset)
p = lower(path);
if contains(p,'108'), h = 'Heart108';
elseif contains(p,'208'), h = 'Heart208';
else
    parent = fileparts(path);
    [~,nm] = fileparts(parent);
    if strcmpi(nm,dataset), h = 'HeartUnspecified';
    else, h = matlab.lang.makeValidName(nm);
    end
end
end

function saveDiagnostics(I,gray,med,gauss,sharp,heart,info,scores,masks,thresholds,ds,name,outRoot)
outDir = fullfile(outRoot,'Diagnostics',ds);
mkdirIfNeeded(outDir);
fig = figure('Visible','off','Color','w','Position',[50 50 1500 1050]);
tiledlayout(4,3,'Padding','compact','TileSpacing','compact');
nexttile; imshow(I); title('Original RGB');
nexttile; imshow(heart); title('Automatic heart mask');
nexttile;
if exist('labeloverlay','file') == 2
    imshow(labeloverlay(I,heart)); title('Heart boundary overlay');
else
    imshow(I); hold on; visboundaries(heart,'Color','g'); title('Heart boundary');
end
nexttile; imshow(gray,[]); title('Grayscale');
nexttile; imshow(med,[]); title('Median-filter diagnostic');
nexttile; imshow(gauss,[]); title('Gaussian low-pass diagnostic');
nexttile; imshow(sharp,[]); title('Sharpening diagnostic');
nexttile; imshow(info.distance,[]); title('Distance from background');
nexttile; imshow(info.initialCandidate); title('Initial heart candidate');

fn = fieldnames(masks);
for q = 1:min(3,numel(fn))
    nexttile; imshow(masks.(fn{q}));
    title(strrep(fn{q},'_',' '),'Interpreter','none');
end
for q = numel(fn)+1:3
    nexttile; axis off;
end
sgtitle([ds ' - ' name ' | ' info.method],'Interpreter','none');
saveas(fig,fullfile(outDir,[name '_MainDiagnostics.png']));
close(fig);

% Save the detected background grid-line mask for quality control.
if isfield(info,'gridLineMask')
    figGrid = figure('Visible','off','Color','w','Position',[100 100 700 500]);
    imshow(info.gridLineMask);
    title('Detected dark background grid lines excluded from heart mask');
    saveas(figGrid,fullfile(outDir,[name '_ExcludedGridLines.png']));
    close(figGrid);
end

fig2 = figure('Visible','off','Color','w','Position',[50 50 1400 900]);
for q = 1:numel(fn)
    subplot(ceil(numel(fn)/2),2,q);
    imshow(masks.(fn{q}));
    tv = thresholds.(fn{q});
    title(sprintf('%s | %.3f',strrep(fn{q},'_',' '),tv),'Interpreter','none');
end
sgtitle([ds ' - ' name ' | Segmentation comparison'],'Interpreter','none');
saveas(fig2,fullfile(outDir,[name '_SegmentationComparison.png']));
close(fig2);
end

function saveManualReferenceDiagnostics(I,autoHeart,manualHeart,manualInfarct,ds,name,outRoot)
outDir = fullfile(outRoot,'ManualValidation',ds);
mkdirIfNeeded(outDir);
fig = figure('Visible','off','Color','w','Position',[80 80 1300 850]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

nexttile; imshow(I); title('Original image');

nexttile;
if ~isempty(manualHeart)
    imshow(manualHeart); title('Manual heart mask');
else
    axis off; title('Manual heart mask unavailable');
end

nexttile;
if ~isempty(manualInfarct)
    imshow(manualInfarct); title('Manual infarct mask (inside manual heart)');
else
    axis off; title('Manual infarct mask unavailable');
end

nexttile;
imshow(autoHeart); title('Automatic heart mask');

nexttile;
if ~isempty(manualHeart)
    imshow(I); hold on;
    visboundaries(manualHeart,'Color','g','LineWidth',0.8);
    visboundaries(autoHeart,'Color','r','LineWidth',0.8);
    title('Heart boundary: manual green / automatic red');
else
    axis off; title('Heart overlay unavailable');
end

nexttile;
if ~isempty(manualInfarct)
    imshow(I); hold on;
    visboundaries(manualInfarct,'Color','g','LineWidth',0.8);
    title('Manual infarct boundary');
else
    axis off; title('Infarct overlay unavailable');
end

sgtitle([ds ' - ' name ' | Manual-reference quality check'],'Interpreter','none');
saveas(fig,fullfile(outDir,[name '_ManualReferenceCheck.png']));
close(fig);
end

function saveEdges(gray,ds,name,outRoot)
try
    sobel = edge(gray,'Sobel');
    canny = edge(gray,'Canny');
    fig = figure('Visible','off','Color','w','Position',[100 100 1000 400]);
    subplot(1,3,1); imshow(gray,[]); title('Grayscale');
    subplot(1,3,2); imshow(sobel); title('Sobel edges');
    subplot(1,3,3); imshow(canny); title('Canny edges');
    sgtitle('Edge diagnostics only - not infarct classification');
    outDir = fullfile(outRoot,'Diagnostics',ds);
    mkdirIfNeeded(outDir);
    saveas(fig,fullfile(outDir,[name '_Edges.png']));
    close(fig);
catch
end
end

function saveHistograms(scores,mask,ds,name,outRoot)
names = fieldnames(scores);
fig = figure('Visible','off','Color','w','Position',[100 100 1100 800]);
for q = 1:numel(names)
    subplot(2,2,q);
    histogram(scores.(names{q})(mask),50);
    xlabel('Normalized infarct score'); ylabel('Pixel count');
    title([names{q} ' score'],'Interpreter','none'); grid on;
end
sgtitle('Color-space score distributions inside heart mask');
outDir = fullfile(outRoot,'Diagnostics',ds);
mkdirIfNeeded(outDir);
saveas(fig,fullfile(outDir,[name '_ScoreHistograms.png']));
close(fig);
end

function S = heartLevelSummary(T)
K = unique(T(:,{'Dataset','Heart','ColorSpace','ThresholdMethod'}),'rows');
n = height(K);
Dataset = strings(n,1); Heart = strings(n,1);
ColorSpace = strings(n,1); ThresholdMethod = strings(n,1);
NumberOfSlices = zeros(n,1);
TotalSliceWeight_mg = NaN(n,1);
EstimatedInfarctWeight_mg = NaN(n,1);
WeightedInfarctPercent = NaN(n,1);
MeanSliceInfarctPercent = NaN(n,1);

for q = 1:n
    ix = T.Dataset==K.Dataset(q) & T.Heart==K.Heart(q) & ...
        T.ColorSpace==K.ColorSpace(q) & T.ThresholdMethod==K.ThresholdMethod(q);
    sub = T(ix,:);
    Dataset(q)=K.Dataset(q); Heart(q)=K.Heart(q);
    ColorSpace(q)=K.ColorSpace(q); ThresholdMethod(q)=K.ThresholdMethod(q);
    NumberOfSlices(q)=height(sub);
    MeanSliceInfarctPercent(q)=mean(sub.InfarctPercent,'omitnan');
    valid = isfinite(sub.SliceWeight_mg) & isfinite(sub.EstimatedInfarctWeight_mg);
    if any(valid)
        TotalSliceWeight_mg(q)=sum(sub.SliceWeight_mg(valid));
        EstimatedInfarctWeight_mg(q)=sum(sub.EstimatedInfarctWeight_mg(valid));
        WeightedInfarctPercent(q)=100*EstimatedInfarctWeight_mg(q)/TotalSliceWeight_mg(q);
    end
end
S = table(Dataset,Heart,ColorSpace,ThresholdMethod,NumberOfSlices, ...
    TotalSliceWeight_mg,EstimatedInfarctWeight_mg,WeightedInfarctPercent, ...
    MeanSliceInfarctPercent);
end

function S = comparisonSummary(T)
K = unique(T(:,{'Dataset','ColorSpace','ThresholdMethod'}),'rows');
n = height(K);
Dataset = strings(n,1); ColorSpace = strings(n,1);
ThresholdMethod = strings(n,1); NumberOfImages = zeros(n,1);
MeanInfarctPercent = NaN(n,1); StdInfarctPercent = NaN(n,1);
MeanManualInfarctPercent = NaN(n,1);
MeanInfarctPercentWithinManualHeart = NaN(n,1);
NumberWithManualReference = zeros(n,1);
NumberWithManualPixelMasks = zeros(n,1);
MAE_percent = NaN(n,1); RMSE_percent = NaN(n,1);
MeanHeartDice = NaN(n,1); MeanHeartIoU = NaN(n,1);
MeanInfarctDice = NaN(n,1); MeanInfarctIoU = NaN(n,1);
MeanInfarctPrecision = NaN(n,1); MeanInfarctSensitivity = NaN(n,1);
MeanInfarctSpecificity = NaN(n,1); MeanInfarctAccuracy = NaN(n,1);

for q = 1:n
    ix = T.Dataset==K.Dataset(q) & T.ColorSpace==K.ColorSpace(q) & ...
        T.ThresholdMethod==K.ThresholdMethod(q);
    sub = T(ix,:);
    Dataset(q)=K.Dataset(q); ColorSpace(q)=K.ColorSpace(q);
    ThresholdMethod(q)=K.ThresholdMethod(q);
    NumberOfImages(q)=height(sub);
    MeanInfarctPercent(q)=mean(sub.InfarctPercent,'omitnan');
    StdInfarctPercent(q)=std(sub.InfarctPercent,'omitnan');
    MeanManualInfarctPercent(q)=mean(sub.ManualInfarctPercent,'omitnan');
    MeanInfarctPercentWithinManualHeart(q)= ...
        mean(sub.InfarctPercentWithinManualHeart,'omitnan');

    validPercent = isfinite(sub.ManualInfarctPercent);
    NumberWithManualReference(q)=nnz(validPercent);
    if any(validPercent)
        autoForValidation = sub.InfarctPercent;
        commonROI = isfinite(sub.InfarctPercentWithinManualHeart);
        autoForValidation(commonROI) = sub.InfarctPercentWithinManualHeart(commonROI);
        e = autoForValidation(validPercent)-sub.ManualInfarctPercent(validPercent);
        MAE_percent(q)=mean(abs(e));
        RMSE_percent(q)=sqrt(mean(e.^2));
    end

    validMasks = isfinite(sub.InfarctDice);
    NumberWithManualPixelMasks(q)=nnz(validMasks);
    MeanHeartDice(q)=mean(sub.HeartDice,'omitnan');
    MeanHeartIoU(q)=mean(sub.HeartIoU,'omitnan');
    MeanInfarctDice(q)=mean(sub.InfarctDice,'omitnan');
    MeanInfarctIoU(q)=mean(sub.InfarctIoU,'omitnan');
    MeanInfarctPrecision(q)=mean(sub.InfarctPrecision,'omitnan');
    MeanInfarctSensitivity(q)=mean(sub.InfarctSensitivity,'omitnan');
    MeanInfarctSpecificity(q)=mean(sub.InfarctSpecificity,'omitnan');
    MeanInfarctAccuracy(q)=mean(sub.InfarctAccuracy,'omitnan');
end
S = table(Dataset,ColorSpace,ThresholdMethod,NumberOfImages, ...
    MeanInfarctPercent,StdInfarctPercent,MeanManualInfarctPercent, ...
    MeanInfarctPercentWithinManualHeart,NumberWithManualReference, ...
    NumberWithManualPixelMasks,MAE_percent,RMSE_percent, ...
    MeanHeartDice,MeanHeartIoU,MeanInfarctDice,MeanInfarctIoU, ...
    MeanInfarctPrecision,MeanInfarctSensitivity, ...
    MeanInfarctSpecificity,MeanInfarctAccuracy);
end

function saveManualValidationGraphs(T,outRoot)
valid = isfinite(T.ManualInfarctPercent) & isfinite(T.InfarctPercent);
if ~any(valid)
    fprintf('No manual percentage references available; validation graphs skipped.\n');
    return;
end

V = T(valid,:);
methodLabels = unique(V.ColorSpace + "_" + V.ThresholdMethod,'stable');
autoForValidation = V.InfarctPercent;
commonROI = isfinite(V.InfarctPercentWithinManualHeart);
autoForValidation(commonROI) = V.InfarctPercentWithinManualHeart(commonROI);

% Scatter: automatic infarct percentage against manual reference percentage.
fig = figure('Visible','off','Color','w','Position',[100 100 950 750]);
hold on;
allValues = [V.ManualInfarctPercent; autoForValidation];
lo = min(allValues);
hi = max(allValues);
if lo == hi, hi = lo + 1; end
plot([lo hi],[lo hi],'k--','DisplayName','Ideal agreement (y = x)');
for q = 1:numel(methodLabels)
    ix = (V.ColorSpace + "_" + V.ThresholdMethod) == methodLabels(q);
    plot(V.ManualInfarctPercent(ix),autoForValidation(ix),'o', ...
        'LineWidth',1.2,'DisplayName',char(methodLabels(q)));
end
xlabel('Manual infarct area (% of manual heart)');
ylabel('Automatic infarct area (% of manual heart ROI when available)');
title('Automatic versus manual infarct percentage');
legend('Location','bestoutside'); grid on; axis square;
saveas(fig,fullfile(outRoot,'ManualValidation_AutomaticVsManual.png'));
close(fig);

% Mean Dice by color-space / threshold combination.
meanDice = NaN(numel(methodLabels),1);
meanAbsError = NaN(numel(methodLabels),1);
for q = 1:numel(methodLabels)
    ix = (V.ColorSpace + "_" + V.ThresholdMethod) == methodLabels(q);
    meanDice(q) = mean(V.InfarctDice(ix),'omitnan');
    meanAbsError(q) = mean(V.AbsoluteError_percent(ix),'omitnan');
end

fig2 = figure('Visible','off','Color','w','Position',[100 100 1200 650]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile;
if any(isfinite(meanDice))
    bar(meanDice);
    set(gca,'XTick',1:numel(methodLabels),'XTickLabel',cellstr(methodLabels));
    xtickangle(45); ylabel('Mean Dice coefficient');
    title('Manual-mask infarct overlap'); ylim([0 1]); grid on;
else
    axis off;
    text(0.1,0.5,'No pixel-level manual masks available for Dice scoring.');
end
nexttile;
bar(meanAbsError);
set(gca,'XTick',1:numel(methodLabels),'XTickLabel',cellstr(methodLabels));
xtickangle(45); ylabel('Mean absolute error (percentage points)');
title('Infarct percentage error'); grid on;
saveas(fig2,fullfile(outRoot,'ManualValidation_MethodMetrics.png'));
close(fig2);
end

function saveSummaryGraphs(T,H,outRoot)
fig = figure('Visible','off','Color','w','Position',[100 100 1300 700]);
groups = unique(T(:,{'Dataset','ThresholdMethod'}),'rows');
for q = 1:height(groups)
    subplot(height(groups),1,q);
    ix = T.Dataset==groups.Dataset(q) & T.ThresholdMethod==groups.ThresholdMethod(q);
    sub = T(ix,:);
    hold on;
    colors = unique(sub.ColorSpace,'stable');
    for c = 1:numel(colors)
        jj = sub.ColorSpace==colors(c);
        plot(find(jj),sub.InfarctPercent(jj),'-o','DisplayName',char(colors(c)));
    end
    ylabel('Infarct (%)');
    title(char(groups.Dataset(q)+' - '+groups.ThresholdMethod(q)));
    legend('Location','best'); grid on; hold off;
end
saveas(fig,fullfile(outRoot,'Summary_InfarctPercent_ByImage.png'));
close(fig);

fig2 = figure('Visible','off','Color','w','Position',[100 100 1200 600]);
valid = isfinite(H.WeightedInfarctPercent);
if any(valid)
    bar(H.WeightedInfarctPercent(valid));
    labels = H.Heart(valid)+"_"+H.ColorSpace(valid)+"_"+H.ThresholdMethod(valid);
    set(gca,'XTick',1:nnz(valid),'XTickLabel',cellstr(labels));
    xtickangle(45); ylabel('Weighted infarct size (%)');
    title('Heart-level weighted infarct percentage'); grid on;
else
    text(0.1,0.5,'No slice weights parsed. Check filenames for IS<number>mg.');
    axis off;
end
saveas(fig2,fullfile(outRoot,'Summary_HeartLevelInfarct.png'));
close(fig2);
end
