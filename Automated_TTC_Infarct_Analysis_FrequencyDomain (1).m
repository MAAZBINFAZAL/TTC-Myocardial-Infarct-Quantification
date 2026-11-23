%% ========================================================================
%  Project: Automated Quantification of Myocardial Infarct Size
%  Course:  EGR 532 - Biomedical Imaging and Image Processing (Class Project #1)
%  Author:  Maaz Bin Fazal
%  Instructor: Dr. Samhita Rhodes
%  Institution: Grand Valley State University
%  Date:    November 19, 2026
%  ========================================================================

clear; clc; close all;
fprintf('\n=== OPTIMIZED AUTOMATED TTC HEART / INFARCT ANALYSIS ===\n');

%% 0. CONFIGURATION - EDIT THESE PATHS IF NEEDED
cfg.projectRoot = pwd;
% If the script is launched while MATLAB is inside 03_Code, use the project
% root one level above it so 05_Manual_Tracing is found automatically.
[~,currentFolderName] = fileparts(cfg.projectRoot);
if strcmpi(currentFolderName,'03_Code')
    cfg.projectRoot = fileparts(cfg.projectRoot);
end
cfg.trainingFolder = fullfile(cfg.projectRoot,'Training');
cfg.testFolder = fullfile(cfg.projectRoot,'Test');

cfg.outputFolderName = ['Infarct_Optimized_' datestr(now,'yyyymmdd_HHMMSS')];

% Heart extraction candidates are compared on TRAINING references only.
cfg.defaultHeartMaskMethod = 'LabDistance'; % LabDistance | Saturation | Hybrid
cfg.backgroundDistanceMin = 0.06;
cfg.heartOpeningRadius = 2;       % suppress thin graph/grid lines before closing
cfg.heartClosingRadius = 3;       % close only small tissue-boundary gaps
cfg.maximumSmallHoleFraction = 0.001; % fill only tiny enclosed holes (0.1% of tissue)
cfg.minimumHeartComponentPixels = 100;
cfg.keepLargestHeartComponent = true;
cfg.minimumHeartAreaFraction = 0.03;
cfg.maximumHeartAreaFraction = 0.95;

% Adaptive-threshold sensitivity candidates. Selected using TRAINING labels.
cfg.tuneOnTraining = true;
cfg.adaptiveSensitivityCandidates = 0.35:0.05:0.75;
cfg.defaultAdaptiveSensitivity = 0.50;

% Score smoothing: keep off by default to preserve small infarct boundaries.
cfg.useMedianFilterForScores = false;
cfg.medianWindow = [3 3];
cfg.useGaussianSmoothing = false;
cfg.gaussianSigma = 0.8;

% Frequency-domain analysis (LE8-style zero padding, centered 2-D DFT).
% Cutoff fraction is relative to Nyquist; 0.06 gives D0 = 0.03 cycles/pixel.
% The selected Gaussian LPF branch is applied to the zero-padded 2-D DFT.
cfg.frequencyCutoffFractions = 0.06; % selected cutoff for final frequency branch
cfg.frequencyButterworthOrder = 2;
cfg.frequencyPaddingFactor = 2;
% Only Gaussian LPF 0.06 is used for infarct classification.
% Ideal/Butterworth responses remain diagnostic images only.
cfg.frequencyIncludeIdealLPF = false;
cfg.frequencyIncludeButterworthLPF = false;
cfg.frequencyIncludeGaussianLPF = true;
cfg.frequencyIncludeHighPassDiagnostics = true;
cfg.frequencyIncludeLaplacianDiagnostic = true;

% Infarct mask cleanup. These are starting settings, not validated values.
cfg.removeSmallInfarctObjects = true;
cfg.minimumInfarctObjectFraction = 0.0005;
cfg.infarctClosingRadius = 1;
cfg.infarctOpeningRadius = 0;

% Reference files:
% ManualReference.csv columns:
% Filename,ManualHeartPercent,ManualInfarctPercent
% ManualHeartPercent = heart area as % of the FULL image.
% ManualInfarctPercent = infarct area as % of the MANUALLY outlined heart.
cfg.manualReferenceCSV = fullfile(cfg.projectRoot,'ManualReference.csv');
% Existing manual masks are stored in:
% 05_Manual_Tracing/<imageBase>/<imageBase>_Manual_HeartMask.tif
% 05_Manual_Tracing/<imageBase>/<imageBase>_Manual_InfarctMask.tif
cfg.manualHeartMaskFolder = fullfile(cfg.projectRoot,'05_Manual_Tracing');
cfg.manualInfarctMaskFolder = fullfile(cfg.projectRoot,'05_Manual_Tracing');

cfg.saveMasks = true;
cfg.saveDiagnostics = true;
cfg.saveHistograms = true;
cfg.saveEdgeDiagnostics = true;

extensions = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'};
colorNames = {'RGB','HSV','CIELAB','Grayscale'};
frequencyScoreNames = buildFrequencyScoreNames(cfg);
scoreNames = [colorNames frequencyScoreNames];
thresholdNames = {'Otsu','Adaptive'};
heartMethods = {'LabDistance','Saturation','Hybrid'};

%% 1. LOCATE DATA
if ~isfolder(cfg.trainingFolder)
    fprintf('Training folder not found: %s\n',cfg.trainingFolder);
    p = uigetdir(cfg.projectRoot,'Select TRAINING folder (Cancel if unavailable)');
    if isequal(p,0), cfg.trainingFolder = ''; else, cfg.trainingFolder = p; end
end
if ~isempty(cfg.trainingFolder) && ~isfolder(cfg.trainingFolder)
    cfg.trainingFolder = '';
end

if ~isfolder(cfg.testFolder)
    fprintf('Test folder not found: %s\n',cfg.testFolder);
    p = uigetdir(cfg.projectRoot,'Select TEST folder');
    if isequal(p,0), error('Select a Test folder or set cfg.testFolder.'); end
    cfg.testFolder = p;
end

trainFiles = {};
if ~isempty(cfg.trainingFolder)
    trainFiles = listImages(cfg.trainingFolder,extensions);
end
testFiles = listImages(cfg.testFolder,extensions);

if isempty(testFiles)
    error('No test images found. Check cfg.testFolder.');
end
fprintf('\nTraining images: %d | Test images: %d\n',numel(trainFiles),numel(testFiles));
fprintf('Training folder: %s\nTest folder: %s\n',cfg.trainingFolder,cfg.testFolder);

%% 2. LOAD MANUAL REFERENCES
manualTable = table();
if isfile(cfg.manualReferenceCSV)
    try
        manualTable = readtable(cfg.manualReferenceCSV);
        fprintf('Manual reference CSV loaded: %s\n',cfg.manualReferenceCSV);
    catch ME
        warning('Could not read ManualReference.csv: %s',ME.message);
    end
else
    fprintf(['No ManualReference.csv found. Pixel-wise accuracy and ' ...
        'percentage errors will be NaN until references are supplied.\n']);
end

%% 3. TRAINING-ONLY CALIBRATION
% Select the heart-mask method and one adaptive sensitivity per color space.
% If reference masks/percentages are unavailable, defaults are retained.
bestHeartMethod = cfg.defaultHeartMaskMethod;
bestSensitivity = struct();
for c = 1:numel(scoreNames)
    bestSensitivity.(matlab.lang.makeValidName(scoreNames{c})) = ...
        cfg.defaultAdaptiveSensitivity;
end
calibrationRows = struct([]);

if cfg.tuneOnTraining && ~isempty(trainFiles)
    fprintf('\nCalibrating with TRAINING images only...\n');
    [bestHeartMethod,bestSensitivity,calibrationRows] = ...
        calibrateTraining(trainFiles,cfg,scoreNames,heartMethods,manualTable);
else
    fprintf('\nTraining calibration skipped; configured defaults will be used.\n');
end
fprintf('Selected heart method: %s\n',bestHeartMethod);
for c = 1:numel(scoreNames)
    key = matlab.lang.makeValidName(scoreNames{c});
    fprintf('Adaptive sensitivity %-32s = %.2f\n',scoreNames{c},bestSensitivity.(key));
end

%% 4. CREATE OUTPUT STRUCTURE
outRoot = fullfile(cfg.projectRoot,cfg.outputFolderName);
mkdirIfNeeded(outRoot);
mkdirIfNeeded(fullfile(outRoot,'HeartMasks'));
mkdirIfNeeded(fullfile(outRoot,'InfarctMasks'));
mkdirIfNeeded(fullfile(outRoot,'Diagnostics'));

%% 5. PROCESS TRAINING AND TEST SETS WITH FROZEN PARAMETERS
% Both sets are processed with the same selected parameters. Training metrics
% are for development; Test metrics are the held-out evaluation.
sliceRows = struct([]);
heartRows = struct([]);
infarctMetricRows = struct([]);

sets = {'Training',trainFiles; 'Test',testFiles};
for ds = 1:size(sets,1)
    datasetName = sets{ds,1};
    files = sets{ds,2};
    if isempty(files), continue; end
    fprintf('\n--- Processing %s (%d images) ---\n',datasetName,numel(files));

    for k = 1:numel(files)
        imagePath = files{k};
        [~,baseName,ext] = fileparts(imagePath);
        fileName = [baseName ext];
        fprintf('%s %d/%d: %s\n',datasetName,k,numel(files),fileName);

        try
            [raw,map] = imread(imagePath);
            if ~isempty(map)
                I = ind2rgb(raw,map);
            else
                I = toRGBDouble(raw);
            end
            I = toRGBDouble(I);
            [nr,nc,~] = size(I);
            if nr < 20 || nc < 20
                warning('Image too small; skipped: %s',fileName);
                continue;
            end

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

            %% 5A. HEART MASK
            [heartMask,heartInfo] = makeHeartMask(I,cfg,bestHeartMethod);
            heartPixels = nnz(heartMask);
            heartPercent = 100*heartPixels/numel(heartMask);

            qualityFlag = "OK";
            if heartPercent < 100*cfg.minimumHeartAreaFraction || ...
                    heartPercent > 100*cfg.maximumHeartAreaFraction
                qualityFlag = "CHECK_HEART_AREA";
                warning('Heart-mask area outside QC limits for %s (%.2f%%).', ...
                    fileName,heartPercent);
            end
            if any(heartMask(1,:)) || any(heartMask(end,:)) || ...
                    any(heartMask(:,1)) || any(heartMask(:,end))
                qualityFlag = "CHECK_BORDER_CONTACT";
            end

            manualHeartMask = readReferenceMask(cfg.manualHeartMaskFolder,baseName,nr,nc,'_Manual_HeartMask');
            manualInfarctMask = readReferenceMask(cfg.manualInfarctMaskFolder,baseName,nr,nc,'_Manual_InfarctMask');
            [manualHeartPercent,manualInfarctPercent] = ...
                getManualPercentages(manualTable,fileName,datasetName);

            % If no CSV percentages are supplied, calculate them directly
            % from the user's manually traced binary masks.
            if ~isfinite(manualHeartPercent) && ~isempty(manualHeartMask)
                manualHeartPercent = 100*nnz(manualHeartMask)/numel(manualHeartMask);
            end
            if ~isfinite(manualInfarctPercent) && ~isempty(manualInfarctMask)
                if ~isempty(manualHeartMask)
                    manualInfarctPercent = 100*nnz(manualInfarctMask & manualHeartMask) ...
                        /max(nnz(manualHeartMask),1);
                else
                    manualInfarctPercent = 100*nnz(manualInfarctMask) ...
                        /max(nnz(heartMask),1);
                end
            end

            %% 5B. HEART MASK METRICS
            hm = binaryMetrics(heartMask,manualHeartMask);
            heartAreaError = NaN;
            if isfinite(manualHeartPercent)
                heartAreaError = heartPercent-manualHeartPercent;
            end
            hr = struct('Dataset',datasetName,'Heart',heartID(fileName), ...
                'Filename',fileName,'HeartMaskMethod',bestHeartMethod, ...
                'HeartPixels',heartPixels,'ImagePixels',numel(heartMask), ...
                'HeartMaskPercentOfImage',heartPercent, ...
                'ManualHeartPercent',manualHeartPercent, ...
                'HeartAreaError_pp',heartAreaError, ...
                'HeartDice_F1',hm.Dice,'HeartIoU',hm.IoU, ...
                'HeartPrecision',hm.Precision,'HeartRecall',hm.Recall, ...
                'HeartSpecificity',hm.Specificity,'HeartAccuracy',hm.Accuracy, ...
                'HeartBalancedAccuracy',hm.BalancedAccuracy,'HeartMCC',hm.MCC, ...
                'QualityFlag',char(qualityFlag));
            heartRows = appendStruct(heartRows,hr);

            %% 5C. COLOR-SPACE SCORES
            [scores,labChannels,features,freqDiag] = makeColorScores(I,heartMask,cfg);

            %% 5D. THRESHOLD COMPARISON
            imageMasks = struct();
            imageThresholds = struct();

            for c = 1:numel(scoreNames)
                cname = scoreNames{c};
                score = scores.(cname);

                % Final frequency branch is Gaussian LPF 0.06 + Adaptive only.
                % Conventional color/grayscale branches retain Otsu + Adaptive.
                methodsForScore = thresholdNames;
                if startsWith(cname,'FD_')
                    methodsForScore = {'Adaptive'};
                end

                if cfg.useMedianFilterForScores
                    score = medfilt2(score,cfg.medianWindow);
                end
                if cfg.useGaussianSmoothing && exist('imgaussfilt','file') == 2
                    score = imgaussfilt(score,cfg.gaussianSigma);
                end
                score = min(max(score,0),1);

                for t = 1:numel(methodsForScore)
                    tname = methodsForScore{t};
                    thresholdValue = NaN;
                    localThreshold = [];

                    if strcmp(tname,'Otsu')
                        vals = score(heartMask);
                        vals = vals(isfinite(vals));
                        if isempty(vals)
                            thresholdValue = 1;
                        else
                            thresholdValue = graythresh(vals);
                        end
                        candidate = score > thresholdValue;
                    else
                        if exist('adaptthresh','file') ~= 2
                            warning('adaptthresh unavailable; Adaptive skipped.');
                            continue;
                        end
                        sensKey = matlab.lang.makeValidName(cname);
                        sens = bestSensitivity.(sensKey);
                        localThreshold = adaptthresh(score,sens, ...
                            'ForegroundPolarity','bright');
                        candidate = score > localThreshold;
                        thresholdValue = sens;
                    end

                    candidate = logical(candidate) & heartMask;
                    infarctMask = cleanInfarctMask(candidate,heartMask,cfg);
                    infarctPixels = nnz(infarctMask);
                    infarctPercent = 100*infarctPixels/max(heartPixels,1);

                    % If no manual percentage is supplied but a manual infarct
                    % mask exists, estimate the reference % within the manual
                    % heart mask when available, otherwise within this heart mask.
                    refInfarctPercent = manualInfarctPercent;
                    if ~isfinite(refInfarctPercent) && ~isempty(manualInfarctMask)
                        denominator = heartPixels;
                        if ~isempty(manualHeartMask), denominator = nnz(manualHeartMask); end
                        numerator = nnz(manualInfarctMask);
                        if ~isempty(manualHeartMask)
                            numerator = nnz(manualInfarctMask & manualHeartMask);
                        end
                        refInfarctPercent = 100*numerator/max(denominator,1);
                    end

                    signedError = NaN;
                    if isfinite(refInfarctPercent)
                        signedError = infarctPercent-refInfarctPercent;
                    end
                    absError = abs(signedError);

                    manualInfarctForMetrics = manualInfarctMask;
                    if ~isempty(manualInfarctForMetrics) && ~isempty(manualHeartMask)
                        manualInfarctForMetrics = manualInfarctForMetrics & manualHeartMask;
                    end
                    im = binaryMetrics(infarctMask,manualInfarctForMetrics);
                    sliceWeight = weightFromName(fileName);
                    estimatedWeight = NaN;
                    estimatedNonInfarctWeight = NaN;
                    manualEstimatedWeight = NaN;
                    manualNonInfarctWeight = NaN;
                    infarctWeightError = NaN;
                    if isfinite(sliceWeight)
                        estimatedWeight = infarctPercent/100*sliceWeight;
                        estimatedNonInfarctWeight = sliceWeight-estimatedWeight;
                        if isfinite(refInfarctPercent)
                            manualEstimatedWeight = refInfarctPercent/100*sliceWeight;
                            manualNonInfarctWeight = sliceWeight-manualEstimatedWeight;
                            infarctWeightError = estimatedWeight-manualEstimatedWeight;
                        end
                    end

                    row = struct('Dataset',datasetName,'Heart',heartID(fileName), ...
                        'Filename',fileName,'SliceWeight_mg',sliceWeight, ...
                        'AnalysisMethod',cname,'ThresholdMethod',tname, ...
                        'ThresholdValue',thresholdValue, ...
                        'AdaptiveSensitivity',bestSensitivity.(matlab.lang.makeValidName(cname)), ...
                        'ImageRows',nr,'ImageColumns',nc, ...
                        'HeartPixels',heartPixels,'InfarctPixels',infarctPixels, ...
                        'HeartMaskPercentOfImage',heartPercent, ...
                        'InfarctPercentOfHeart',infarctPercent, ...
                        'EstimatedInfarctWeight_mg',estimatedWeight, ...
                        'EstimatedNonInfarctWeight_mg',estimatedNonInfarctWeight, ...
                        'ManualInfarctWeight_mg',manualEstimatedWeight, ...
                        'ManualNonInfarctWeight_mg',manualNonInfarctWeight, ...
                        'InfarctWeightError_mg',infarctWeightError, ...
                        'AbsoluteInfarctWeightError_mg',abs(infarctWeightError), ...
                        'ManualInfarctPercent',refInfarctPercent, ...
                        'InfarctError_pp',signedError, ...
                        'InfarctAbsoluteError_pp',absError, ...
                        'InfarctDice_F1',im.Dice,'InfarctIoU',im.IoU, ...
                        'InfarctPrecision',im.Precision,'InfarctRecall',im.Recall, ...
                        'InfarctSpecificity',im.Specificity,'InfarctAccuracy',im.Accuracy, ...
                        'InfarctBalancedAccuracy',im.BalancedAccuracy,'InfarctMCC',im.MCC, ...
                        'QualityFlag',char(qualityFlag));

                    sliceRows = appendStruct(sliceRows,row);
                    infarctMetricRows = appendStruct(infarctMetricRows,row);

                    key = matlab.lang.makeValidName([cname '_' tname]);
                    imageMasks.(key) = infarctMask;
                    imageThresholds.(key) = thresholdValue;

                    if cfg.saveMasks
                        maskDir = fullfile(outRoot,'InfarctMasks',datasetName);
                        mkdirIfNeeded(maskDir);
                        imwrite(uint8(infarctMask)*255, ...
                            fullfile(maskDir,[baseName '_' key '.png']));
                    end
                end
            end

            %% 5E. SAVE HEART MASK AND DIAGNOSTICS
            if cfg.saveMasks
                imwrite(uint8(heartMask)*255, ...
                    fullfile(outRoot,'HeartMasks',[datasetName '_' baseName '_HeartMask.png']));
            end
            if cfg.saveDiagnostics
                saveDiagnostics(I,gray,medianGray,gaussianGray,sharpenedGray, ...
                    heartMask,heartInfo,scores,labChannels,freqDiag,imageMasks,imageThresholds, ...
                    datasetName,baseName,outRoot);
            end
            if cfg.saveHistograms
                saveHistograms(scores,heartMask,datasetName,baseName,outRoot);
            end
            if cfg.saveEdgeDiagnostics
                saveEdges(gray,datasetName,baseName,outRoot);
            end

            fprintf('  Heart coverage %.2f%% | %s\n',heartPercent,qualityFlag);

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

if isempty(sliceRows)
    error('No images processed successfully. Check folders and Processing_Errors.txt.');
end

%% 6. TABLES AND ACCURACY METRICS
sliceTable = struct2table(sliceRows);
heartMetricTable = struct2table(heartRows);
infarctMetricTable = struct2table(infarctMetricRows);

% Normalize grouping columns to string arrays before comparisons/grouping.
sliceTable.Dataset = string(sliceTable.Dataset);
sliceTable.Heart = string(sliceTable.Heart);
sliceTable.Filename = string(sliceTable.Filename);
sliceTable.AnalysisMethod = string(sliceTable.AnalysisMethod);
sliceTable.ThresholdMethod = string(sliceTable.ThresholdMethod);
heartMetricTable.Dataset = string(heartMetricTable.Dataset);
heartMetricTable.Heart = string(heartMetricTable.Heart);
heartMetricTable.Filename = string(heartMetricTable.Filename);
heartMetricTable.HeartMaskMethod = string(heartMetricTable.HeartMaskMethod);
infarctMetricTable.Dataset = string(infarctMetricTable.Dataset);
infarctMetricTable.Heart = string(infarctMetricTable.Heart);
infarctMetricTable.Filename = string(infarctMetricTable.Filename);
infarctMetricTable.AnalysisMethod = string(infarctMetricTable.AnalysisMethod);
infarctMetricTable.ThresholdMethod = string(infarctMetricTable.ThresholdMethod);

% Explicitly separate percentage error from pixel-mask overlap metrics.
methodSummary = comparisonSummary(sliceTable);
frequencyRows = startsWith(string(methodSummary.AnalysisMethod),"FD_");
frequencySummary = methodSummary(frequencyRows,:);
heartSummary = heartLevelSummary(sliceTable);
heartMethodSummary = heartComparisonSummary(heartMetricTable);
calibrationTable = struct2tableSafe(calibrationRows);

%% 7. SELECT BEST METHOD FROM TRAINING ONLY
bestMethodText = selectBestTrainingMethod(methodSummary);
fprintf('\nTRAINING-ONLY method selection: %s\n',bestMethodText);

%% 8. EXPORT RESULTS
try
    writetable(sliceTable,fullfile(outRoot,'SliceResults.csv'));
    writetable(heartMetricTable,fullfile(outRoot,'HeartMaskMetrics.csv'));
    writetable(heartMethodSummary,fullfile(outRoot,'HeartMethodComparison.csv'));
    writetable(infarctMetricTable,fullfile(outRoot,'InfarctMaskMetrics.csv'));
    writetable(methodSummary,fullfile(outRoot,'MethodComparison.csv'));
    writetable(frequencySummary,fullfile(outRoot,'FrequencyDomainComparison.csv'));
    writetable(heartSummary,fullfile(outRoot,'HeartLevelSummary.csv'));
    if ~isempty(calibrationTable)
        writetable(calibrationTable,fullfile(outRoot,'TrainingCalibration.csv'));
    end
catch ME
    warning('CSV export issue: %s',ME.message);
end

try
    xlsx = fullfile(outRoot,'Infarct_Analysis_Results.xlsx');
    writetable(sliceTable,xlsx,'Sheet','SliceResults');
    writetable(heartMetricTable,xlsx,'Sheet','HeartMaskMetrics');
    writetable(heartMethodSummary,xlsx,'Sheet','HeartMethodComparison');
    writetable(infarctMetricTable,xlsx,'Sheet','InfarctMaskMetrics');
    writetable(methodSummary,xlsx,'Sheet','MethodComparison');
    writetable(frequencySummary,xlsx,'Sheet','FrequencyDomain');
    writetable(heartSummary,xlsx,'Sheet','HeartSummary');
    if ~isempty(calibrationTable)
        writetable(calibrationTable,xlsx,'Sheet','TrainingCalibration');
    end
catch ME
    warning('Excel export issue: %s',ME.message);
end

%% 9. SUMMARY FIGURES AND RUN RECORD
saveSummaryGraphs(sliceTable,heartMetricTable,methodSummary,outRoot);

fid = fopen(fullfile(outRoot,'Run_Configuration.txt'),'w');
if fid >= 0
    fprintf(fid,'Optimized TTC analysis pipeline\n');
    fprintf(fid,'Training folder: %s\n',cfg.trainingFolder);
    fprintf(fid,'Test folder: %s\n',cfg.testFolder);
    fprintf(fid,'Selected heart method (training only): %s\n',bestHeartMethod);
    fprintf(fid,'Default adaptive sensitivity: %.3f\n',cfg.defaultAdaptiveSensitivity);
    fprintf(fid,'Tuning candidates: %s\n',mat2str(cfg.adaptiveSensitivityCandidates));
    fprintf(fid,'Frequency cutoff fraction: %s (Gaussian LPF classification branch)\n',mat2str(cfg.frequencyCutoffFractions));
    fprintf(fid,'Frequency classification: Gaussian LPF cutoff 0.06 + Adaptive only\n');
    fprintf(fid,'Ideal/Butterworth LPFs: diagnostic images only\n');
    fprintf(fid,'Heart opening radius: %d\n',cfg.heartOpeningRadius);
    fprintf(fid,'Heart closing radius: %d\n',cfg.heartClosingRadius);
    fprintf(fid,'Maximum small-hole fill fraction: %.4f\n',cfg.maximumSmallHoleFraction);
    fprintf(fid,'Frequency Butterworth order: %d\n',cfg.frequencyButterworthOrder);
    fprintf(fid,'Frequency score methods: %s\n',strjoin(frequencyScoreNames,', '));
    fprintf(fid,'Selected sensitivities:\n');
    for c = 1:numel(scoreNames)
        key = matlab.lang.makeValidName(scoreNames{c});
        fprintf(fid,'  %s: %.3f\n',scoreNames{c},bestSensitivity.(key));
    end
    fprintf(fid,'Manual reference CSV: %s\n',cfg.manualReferenceCSV);
    fprintf(fid,'Manual mask root: %s\n',cfg.manualHeartMaskFolder);
    fprintf(fid,'Manual mask layout: <root>/<imageBase>/<imageBase>_Manual_HeartMask.tif and _Manual_InfarctMask.tif\n');
    fprintf(fid,'Training-only best method: %s\n',bestMethodText);
    fprintf(fid,'\nInterpretation notes:\n');
    fprintf(fid,'HeartMaskPercentOfImage is area coverage, not segmentation accuracy.\n');
    fprintf(fid,'Dice/F1/IoU require manual pixel masks.\n');
    fprintf(fid,'Only Gaussian LPF cutoff 0.06 + Adaptive is used for frequency-domain infarct classification.\n');
    fprintf(fid,'Ideal/Butterworth LPF, HPF, and Laplacian outputs are diagnostics only.\n');
    fprintf(fid,'Heart mask removes thin grid lines with opening, keeps the largest component, and fills only tiny holes; large cavities remain excluded.\n');
    fprintf(fid,'MAE/RMSE require manual infarct percentages or reference masks.\n');
    fprintf(fid,'Test data must not be used to tune thresholds or select methods.\n');
    fprintf(fid,'Area-based slice weighting is an estimate and requires protocol validation.\n');
    fclose(fid);
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Output folder: %s\n',outRoot);
fprintf('Best method based on TRAINING references: %s\n',bestMethodText);
fprintf('Review HeartMasks and Diagnostics before interpreting infarct percentages.\n');

%% ============================= FUNCTIONS ==============================

function files = listImages(root,extensions)
files = {};
if isempty(root) || ~isfolder(root), return; end
for q = 1:numel(extensions)
    a = dir(fullfile(root,'**',['*' extensions{q}]));
    b = dir(fullfile(root,'**',['*' upper(extensions{q})]));
    both = [a; b];
    for j = 1:numel(both)
        if ~both(j).isdir
            files{end+1,1} = fullfile(both(j).folder,both(j).name); %#ok<AGROW>
        end
    end
end
if ~isempty(files), files = unique(files,'stable'); end
end

function mkdirIfNeeded(p)
if ~exist(p,'dir'), mkdir(p); end
end

function S = appendStruct(S,row)
if isempty(S), S = row; else, S(end+1) = row; end %#ok<AGROW>
end

function T = struct2tableSafe(S)
if isempty(S), T = table(); else, T = struct2table(S); end
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

function [mask,info] = makeHeartMask(I,cfg,method)
% Shared myocardial/tissue mask for conventional and frequency analyses.
% Excludes background/grid lines and preserves large anatomical cavities.
lab = rgb2lab(I);
[h,w,~] = size(lab);
band = max(2,round(0.04*min(h,w)));
border = false(h,w);
border(1:band,:) = true; border(end-band+1:end,:) = true;
border(:,1:band) = true; border(:,end-band+1:end) = true;

P = reshape(lab,[],3);
bg = median(P(border(:),:),1);
D = lab - reshape(bg,1,1,3);
distanceRaw = sqrt(sum(D.^2,3));
distance = mat2gray(distanceRaw);
distanceThreshold = max(cfg.backgroundDistanceMin,graythresh(distance));

hsv = rgb2hsv(I);
S = hsv(:,:,2);
satThreshold = max(0.08,graythresh(S));

switch lower(method)
    case 'saturation'
        candidate = S > satThreshold;
        methodUsed = 'Saturation';
    case 'hybrid'
        candidate = (distance > distanceThreshold) | (S > satThreshold);
        methodUsed = 'Hybrid';
    otherwise
        candidate = distance > distanceThreshold;
        methodUsed = 'LabDistance';
end

% Remove thin graph-paper/grid structures before they can connect to tissue.
if cfg.heartOpeningRadius > 0
    candidate = imopen(candidate,strel('disk',cfg.heartOpeningRadius,0));
end
candidate = bwareaopen(candidate,cfg.minimumHeartComponentPixels);

% Retain only the largest plausible tissue component.
cc = bwconncomp(candidate);
mask = false(size(candidate));
if cc.NumObjects > 0
    areas = cellfun(@numel,cc.PixelIdxList);
    [~,ix] = max(areas);
    mask(cc.PixelIdxList{ix}) = true;
end

% Close small contour gaps, then fill only very small enclosed holes.
if cfg.heartClosingRadius > 0
    mask = imclose(mask,strel('disk',cfg.heartClosingRadius,0));
end
mask = fillOnlySmallHoles(mask,cfg.maximumSmallHoleFraction);

info.method = methodUsed;
info.distance = distance;
info.distanceRaw = distanceRaw;
info.distanceThreshold = distanceThreshold;
info.saturation = S;
info.saturationThreshold = satThreshold;
info.initialCandidate = candidate;
info.backgroundLab = bg;
info.labL = lab(:,:,1);
info.laba = lab(:,:,2);
info.labb = lab(:,:,3);
end

function mask = fillOnlySmallHoles(mask,maxHoleFraction)
% Fill only tiny enclosed gaps. Large ventricular cavities remain excluded.
mask = logical(mask);
filled = imfill(mask,'holes');
holePixels = filled & ~mask;
cc = bwconncomp(holePixels);
referenceArea = max(nnz(filled),1);
maximumHolePixels = max(1,floor(maxHoleFraction*referenceArea));

for q = 1:cc.NumObjects
    idx = cc.PixelIdxList{q};
    if numel(idx) <= maximumHolePixels
        mask(idx) = true;
    end
end
end

function names = buildFrequencyScoreNames(cfg)
% Candidate score names are stable, valid MATLAB struct field names.
names = {};
for q = 1:numel(cfg.frequencyCutoffFractions)
    suffix = sprintf('Cutoff_%02d',round(100*cfg.frequencyCutoffFractions(q)));
    if cfg.frequencyIncludeIdealLPF
        names{end+1} = ['FD_IdealLPF_' suffix]; %#ok<AGROW>
    end
    if cfg.frequencyIncludeButterworthLPF
        names{end+1} = ['FD_ButterworthLPF2_' suffix]; %#ok<AGROW>
    end
    if cfg.frequencyIncludeGaussianLPF
        names{end+1} = ['FD_GaussianLPF_' suffix]; %#ok<AGROW>
    end
end
end

function [scores,labChannels,F,freqDiag] = makeColorScores(I,mask,cfg)
R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
redRatio = R./(R+G+B+eps);
rgbScore = 1-redRatio;

hsv = rgb2hsv(I);
H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);
hsvScore = V.*(1-S);

lab = rgb2lab(I);
L = lab(:,:,1); a = lab(:,:,2); b = lab(:,:,3);
Lnorm = min(max(L/100,0),1);
anorm = min(max((a+128)/255,0),1);
% TTC-pale tissue tends to be brighter and less red than viable red myocardium.
labComposite = 0.55*Lnorm + 0.45*(1-anorm);

gray = rgb2gray(I);
scores.RGB = min(max(rgbScore,0),1);
scores.HSV = min(max(hsvScore,0),1);
scores.CIELAB = min(max(labComposite,0),1);
scores.Grayscale = min(max(gray,0),1);

labChannels.Lstar = L/100;
labChannels.astar = (a+128)/255;
labChannels.bstar = (b+128)/255;

F.Mean_R = mean(R(mask),'omitnan');
F.Mean_G = mean(G(mask),'omitnan');
F.Mean_B = mean(B(mask),'omitnan');
F.Mean_H = mean(H(mask),'omitnan');
F.Mean_S = mean(S(mask),'omitnan');
F.Mean_V = mean(V(mask),'omitnan');
F.Mean_Lstar = mean(L(mask),'omitnan');
F.Mean_astar = mean(a(mask),'omitnan');
F.Mean_bstar = mean(b(mask),'omitnan');
F.Mean_Gray = mean(gray(mask),'omitnan');
F.Mean_RedRatio = mean(redRatio(mask),'omitnan');
F.Mean_CIELABComposite = mean(labComposite(mask),'omitnan');

[freqScores,freqDiag] = makeFrequencyDomainScores(gray,mask,cfg);
frequencyNames = fieldnames(freqScores);
for q = 1:numel(frequencyNames)
    scores.(frequencyNames{q}) = freqScores.(frequencyNames{q});
end
end

function [scores,diagOut] = makeFrequencyDomainScores(gray,heartMask,cfg)
% Zero-padded, centered 2-D DFT. The only frequency classifier is
% Gaussian low-pass filtering at cutoff fraction 0.06.
[M,N] = size(gray);
P = cfg.frequencyPaddingFactor*M;
Q = cfg.frequencyPaddingFactor*N;
P = max(P,M); Q = max(Q,N);

fp = zeros(P,Q);
fp(1:M,1:N) = gray;
Fcentered = fftshift(fft2(fp));

fx = (-floor(Q/2):ceil(Q/2)-1)/Q;
fy = (-floor(P/2):ceil(P/2)-1)/P;
[FX,FY] = meshgrid(fx,fy);
D = sqrt(FX.^2 + FY.^2);

cutoffFraction = cfg.frequencyCutoffFractions(1);
D0 = cutoffFraction/2; % fraction is relative to Nyquist (0.5 cycles/pixel)
suffix = sprintf('Cutoff_%02d',round(100*cutoffFraction));

% Selected Gaussian LPF score used for Adaptive infarct segmentation.
Hgauss = exp(-(D.^2)/(2*D0^2));
gaussianFiltered = real(ifft2(ifftshift(Hgauss.*Fcentered)));
gaussianCrop = gaussianFiltered(1:M,1:N);
scores = struct();
scores.(['FD_GaussianLPF_' suffix]) = min(max(gaussianCrop,0),1);

% Ideal and Butterworth low-pass images are retained for filter comparison
% diagnostics only; they are not used to generate infarct masks.
Hideal = double(D <= D0);
idealFiltered = real(ifft2(ifftshift(Hideal.*Fcentered)));
diagOut.idealLPF = idealFiltered(1:M,1:N);

Hbutter = 1 ./ (1 + (D./max(D0,eps)).^(2*cfg.frequencyButterworthOrder));
butterFiltered = real(ifft2(ifftshift(Hbutter.*Fcentered)));
diagOut.butterworthLPF = butterFiltered(1:M,1:N);
diagOut.gaussianLPF = gaussianCrop;

% High-pass outputs are diagnostics only (edges/texture, not tissue labels).
if cfg.frequencyIncludeHighPassDiagnostics
    hpg = real(ifft2(ifftshift((1-Hgauss).*Fcentered)));
    hpb = real(ifft2(ifftshift((1-Hbutter).*Fcentered)));
    hpi = real(ifft2(ifftshift((1-Hideal).*Fcentered)));
    diagOut.gaussianHPF = normalizeMagnitude(abs(hpg(1:M,1:N)),heartMask);
    diagOut.butterworthHPF = normalizeMagnitude(abs(hpb(1:M,1:N)),heartMask);
    diagOut.idealHPF = normalizeMagnitude(abs(hpi(1:M,1:N)),heartMask);
end

% Frequency-domain Laplacian is also diagnostic-only.
if cfg.frequencyIncludeLaplacianDiagnostic
    Hlap = -4*pi^2*(D.^2);
    lap = real(ifft2(ifftshift(Hlap.*Fcentered)));
    diagOut.frequencyLaplacian = normalizeMagnitude(abs(lap(1:M,1:N)),heartMask);
    try
        hSpatial = fspecial('laplacian',0.2);
        lapSpatial = imfilter(gray,hSpatial,'replicate');
        diagOut.spatialLaplacian = normalizeMagnitude(abs(lapSpatial),heartMask);
    catch
        diagOut.spatialLaplacian = zeros(M,N);
    end
end

diagOut.logSpectrum = log1p(abs(Fcentered));
diagOut.cutoffFractions = cutoffFraction;
diagOut.paddingSize = [P Q];
end

function out = normalizeMagnitude(A,mask)
vals = A(mask & isfinite(A));
if isempty(vals)
    scale = max(A(:));
else
    scale = prctile(vals,99.5);
end
if ~isfinite(scale) || scale <= 0, scale = 1; end
out = min(max(A/scale,0),1);
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

function [bestHeart,bestSensitivity,rows] = calibrateTraining(files,cfg,scoreNames,heartMethods,manualTable)
bestHeart = cfg.defaultHeartMaskMethod;
bestSensitivity = struct();
for c = 1:numel(scoreNames)
    bestSensitivity.(matlab.lang.makeValidName(scoreNames{c})) = cfg.defaultAdaptiveSensitivity;
end
rows = struct([]);

heartDice = nan(numel(heartMethods),1);
heartAreaMAE = nan(numel(heartMethods),1);
for h = 1:numel(heartMethods)
    d = []; e = [];
    for k = 1:numel(files)
        [~,base,ext] = fileparts(files{k}); name = [base ext];
        try
            [raw,map] = imread(files{k});
            if ~isempty(map), I = ind2rgb(raw,map); else, I = toRGBDouble(raw); end
            I = toRGBDouble(I);
            [nr,nc,~] = size(I);
            M = readReferenceMask(cfg.manualHeartMaskFolder,base,nr,nc,'_Manual_HeartMask');
            [manualPct,~] = getManualPercentages(manualTable,name,'Training');
            [pred,~] = makeHeartMask(I,cfg,heartMethods{h});
            if ~isempty(M)
                z = binaryMetrics(pred,M);
                if isfinite(z.Dice), d(end+1) = z.Dice; end %#ok<AGROW>
            end
            if isfinite(manualPct)
                e(end+1) = abs(100*nnz(pred)/numel(pred)-manualPct); %#ok<AGROW>
            end
        catch ME
            warning('Heart calibration skipped for %s: %s',name,ME.message);
        end
    end
    if ~isempty(d), heartDice(h) = mean(d);
    elseif ~isempty(e), heartAreaMAE(h) = mean(e);
    end
end

if any(isfinite(heartDice))
    [~,ix] = max(heartDice); bestHeart = heartMethods{ix};
    criterion = 'Mean training Dice';
elseif any(isfinite(heartAreaMAE))
    [~,ix] = min(heartAreaMAE); bestHeart = heartMethods{ix};
    criterion = 'Training heart-area MAE (percentage points)';
else
    criterion = 'No heart reference; configured default retained';
end
for h = 1:numel(heartMethods)
    rows = appendStruct(rows,struct('Stage','HeartMask','Method',heartMethods{h}, ...
        'Parameter',NaN,'MetricName',criterion, ...
        'MetricValue',chooseFinite(heartDice(h),heartAreaMAE(h)), ...
        'Selected',strcmp(bestHeart,heartMethods{h})));
end

% Cache each training image's color-space and frequency-domain scores once.
% This avoids recomputing FFTs for every adaptive-sensitivity candidate.
cache = cell(numel(files),1);
for k = 1:numel(files)
    [~,base,ext] = fileparts(files{k}); name = [base ext];
    try
        [raw,map] = imread(files{k});
        if ~isempty(map), I = ind2rgb(raw,map); else, I = toRGBDouble(raw); end
        I = toRGBDouble(I);
        [nr,nc,~] = size(I);
        [heart,~] = makeHeartMask(I,cfg,bestHeart);
        [scores,~,~,~] = makeColorScores(I,heart,cfg);
        manualMask = readReferenceMask(cfg.manualInfarctMaskFolder,base,nr,nc,'_Manual_InfarctMask');
        manualHeart = readReferenceMask(cfg.manualHeartMaskFolder,base,nr,nc,'_Manual_HeartMask');
        if ~isempty(manualMask) && ~isempty(manualHeart)
            manualMask = manualMask & manualHeart;
        end
        [manualHeartPct,manualPct] = getManualPercentages(manualTable,name,'Training');
        if ~isfinite(manualPct) && ~isempty(manualMask)
            denominator = nnz(heart);
            if ~isempty(manualHeart), denominator = nnz(manualHeart); end
            numerator = nnz(manualMask);
            if ~isempty(manualHeart), numerator = nnz(manualMask & manualHeart); end
            manualPct = 100*numerator/max(denominator,1);
        end
        cache{k} = struct('Scores',scores,'Heart',heart,'ManualMask',manualMask, ...
            'ManualPercent',manualPct,'ManualHeartPercent',manualHeartPct,'Name',name);
    catch ME
        warning('Infarct calibration cache skipped for %s: %s',name,ME.message);
        cache{k} = [];
    end
end

% Tune one adaptive sensitivity for each color/frequency-domain score.
for c = 1:numel(scoreNames)
    scoreName = scoreNames{c};
    key = matlab.lang.makeValidName(scoreName);
    diceBySensitivity = nan(size(cfg.adaptiveSensitivityCandidates));
    maeBySensitivity = nan(size(cfg.adaptiveSensitivityCandidates));

    for si = 1:numel(cfg.adaptiveSensitivityCandidates)
        sens = cfg.adaptiveSensitivityCandidates(si);
        d = []; e = [];
        for k = 1:numel(cache)
            item = cache{k};
            if isempty(item), continue; end
            try
                score = item.Scores.(scoreName);
                if exist('adaptthresh','file') ~= 2, continue; end
                localT = adaptthresh(score,sens,'ForegroundPolarity','bright');
                pred = cleanInfarctMask(score>localT,item.Heart,cfg);
                if ~isempty(item.ManualMask)
                    z = binaryMetrics(pred,item.ManualMask);
                    if isfinite(z.Dice), d(end+1) = z.Dice; end %#ok<AGROW>
                end
                if isfinite(item.ManualPercent)
                    predPct = 100*nnz(pred)/max(nnz(item.Heart),1);
                    e(end+1) = abs(predPct-item.ManualPercent); %#ok<AGROW>
                end
            catch
            end
        end
        if ~isempty(d), diceBySensitivity(si) = mean(d); end
        if ~isempty(e), maeBySensitivity(si) = mean(e); end
    end

    if any(isfinite(diceBySensitivity))
        [bestValue,ix] = max(diceBySensitivity);
        bestSensitivity.(key) = cfg.adaptiveSensitivityCandidates(ix);
        metricName = 'Mean training Dice';
    elseif any(isfinite(maeBySensitivity))
        [bestValue,ix] = min(maeBySensitivity);
        bestSensitivity.(key) = cfg.adaptiveSensitivityCandidates(ix);
        metricName = 'Training infarct-percent MAE (percentage points)';
    else
        bestValue = NaN; metricName = 'No infarct reference; default retained';
    end
    rows = appendStruct(rows,struct('Stage','AdaptiveThreshold', ...
        'Method',scoreName,'Parameter',bestSensitivity.(key), ...
        'MetricName',metricName,'MetricValue',bestValue,'Selected',true));
end
end

function v = chooseFinite(a,b)
if isfinite(a), v = a; else, v = b; end
end

function M = readReferenceMask(folder,baseName,nr,nc,suffix)
% Read the user's existing masks from:
% 05_Manual_Tracing/<baseName>/<baseName>_Manual_HeartMask.tif
% 05_Manual_Tracing/<baseName>/<baseName>_Manual_InfarctMask.tif
M = [];
if nargin < 5, suffix = ''; end
if isempty(folder) || ~isfolder(folder), return; end

extensions = {'.tif','.tiff','.png','.jpg','.bmp'};
searchFolders = {fullfile(folder,baseName),folder};

for f = 1:numel(searchFolders)
    if ~isfolder(searchFolders{f}), continue; end
    for e = 1:numel(extensions)
        p = fullfile(searchFolders{f},[baseName suffix extensions{e}]);
        if isfile(p)
            A = imread(p);
            if ndims(A) == 3, A = rgb2gray(A); end
            if size(A,1) ~= nr || size(A,2) ~= nc
                warning('Reference mask size mismatch; ignored: %s',p);
                return;
            end
            M = A > 0;
            return;
        end
    end
end
end

function [heartPct,infarctPct] = getManualPercentages(T,fileName,datasetName)
heartPct = NaN; infarctPct = NaN;
if isempty(T) || height(T)==0, return; end
if ~ismember('Filename',T.Properties.VariableNames), return; end
ix = strcmpi(string(T.Filename),string(fileName));
if ismember('Dataset',T.Properties.VariableNames)
    ix = ix & strcmpi(string(T.Dataset),string(datasetName));
end
if ~any(ix), return; end
r = find(ix,1,'first');
if ismember('ManualHeartPercent',T.Properties.VariableNames)
    heartPct = double(T.ManualHeartPercent(r));
end
if ismember('ManualInfarctPercent',T.Properties.VariableNames)
    infarctPct = double(T.ManualInfarctPercent(r));
end
end

function M = binaryMetrics(pred,ref)
M = struct('Dice',NaN,'IoU',NaN,'Precision',NaN,'Recall',NaN, ...
    'Specificity',NaN,'Accuracy',NaN,'BalancedAccuracy',NaN,'MCC',NaN);
if isempty(ref), return; end
pred = logical(pred); ref = logical(ref);
if ~isequal(size(pred),size(ref)), return; end
TP = nnz(pred & ref);
FP = nnz(pred & ~ref);
FN = nnz(~pred & ref);
TN = nnz(~pred & ~ref);
denDice = 2*TP+FP+FN;
denIoU = TP+FP+FN;
M.Dice = safeDivide(2*TP,denDice);
M.IoU = safeDivide(TP,denIoU);
M.Precision = safeDivide(TP,TP+FP);
M.Recall = safeDivide(TP,TP+FN);
M.Specificity = safeDivide(TN,TN+FP);
M.Accuracy = safeDivide(TP+TN,TP+TN+FP+FN);
if isfinite(M.Recall) && isfinite(M.Specificity)
    M.BalancedAccuracy = mean([M.Recall M.Specificity]);
end
M.MCC = safeDivide(TP*TN-FP*FN, ...
    sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
end

function x = safeDivide(a,b)
if b == 0, x = NaN; else, x = a/b; end
end

function w = weightFromName(name)
tok = regexp(name,'IS([0-9]+(?:\.[0-9]+)?)mg','tokens','once','ignorecase');
if isempty(tok), w = NaN; else, w = str2double(tok{1}); end
end

function h = heartID(name)
% Correctly reads subject/heart prefix such as 108IS132mg or 210IS138mg.
tok = regexp(name,'^([0-9]+)IS','tokens','once','ignorecase');
if isempty(tok), h = 'HeartUnspecified'; else, h = ['Heart' tok{1}]; end
end

function S = comparisonSummary(T)
keys = unique(T(:,{'Dataset','AnalysisMethod','ThresholdMethod'}),'rows');
n = height(keys);
Dataset = strings(n,1); AnalysisMethod = strings(n,1); ThresholdMethod = strings(n,1);
NumberOfImages = zeros(n,1); MeanInfarctPercent = nan(n,1);
StdInfarctPercent = nan(n,1); NumberWithManualPercent = zeros(n,1);
MAE_percentPoints = nan(n,1); RMSE_percentPoints = nan(n,1);
Bias_percentPoints = nan(n,1); MeanDice_F1 = nan(n,1); MeanIoU = nan(n,1);
MeanPrecision = nan(n,1); MeanRecall = nan(n,1);
MeanSpecificity = nan(n,1); MeanAccuracy = nan(n,1);
MeanBalancedAccuracy = nan(n,1); MeanMCC = nan(n,1);

for q = 1:n
    ix = T.Dataset==keys.Dataset(q) & T.AnalysisMethod==keys.AnalysisMethod(q) & ...
        T.ThresholdMethod==keys.ThresholdMethod(q);
    sub = T(ix,:);
    Dataset(q)=keys.Dataset(q); AnalysisMethod(q)=keys.AnalysisMethod(q);
    ThresholdMethod(q)=keys.ThresholdMethod(q);
    NumberOfImages(q)=height(sub);
    MeanInfarctPercent(q)=mean(sub.InfarctPercentOfHeart,'omitnan');
    StdInfarctPercent(q)=std(sub.InfarctPercentOfHeart,'omitnan');

    valid = isfinite(sub.ManualInfarctPercent) & isfinite(sub.InfarctPercentOfHeart);
    NumberWithManualPercent(q)=nnz(valid);
    if any(valid)
        e = sub.InfarctPercentOfHeart(valid)-sub.ManualInfarctPercent(valid);
        MAE_percentPoints(q)=mean(abs(e));
        RMSE_percentPoints(q)=sqrt(mean(e.^2));
        Bias_percentPoints(q)=mean(e);
    end
    MeanDice_F1(q)=mean(sub.InfarctDice_F1,'omitnan');
    MeanIoU(q)=mean(sub.InfarctIoU,'omitnan');
    MeanPrecision(q)=mean(sub.InfarctPrecision,'omitnan');
    MeanRecall(q)=mean(sub.InfarctRecall,'omitnan');
    MeanSpecificity(q)=mean(sub.InfarctSpecificity,'omitnan');
    MeanAccuracy(q)=mean(sub.InfarctAccuracy,'omitnan');
    MeanBalancedAccuracy(q)=mean(sub.InfarctBalancedAccuracy,'omitnan');
    MeanMCC(q)=mean(sub.InfarctMCC,'omitnan');
end
S = table(Dataset,AnalysisMethod,ThresholdMethod,NumberOfImages, ...
    MeanInfarctPercent,StdInfarctPercent,NumberWithManualPercent, ...
    MAE_percentPoints,RMSE_percentPoints,Bias_percentPoints, ...
    MeanDice_F1,MeanIoU,MeanPrecision,MeanRecall,MeanSpecificity, ...
    MeanAccuracy,MeanBalancedAccuracy,MeanMCC);
end

function S = heartComparisonSummary(T)
keys = unique(T(:,{'Dataset','HeartMaskMethod'}),'rows');
n = height(keys);
Dataset = strings(n,1); HeartMaskMethod = strings(n,1);
NumberOfImages = zeros(n,1); MeanHeartDice_F1 = nan(n,1);
MeanHeartIoU = nan(n,1); MeanHeartPrecision = nan(n,1);
MeanHeartRecall = nan(n,1); MeanHeartSpecificity = nan(n,1);
MeanHeartAccuracy = nan(n,1); MeanHeartBalancedAccuracy = nan(n,1);
MeanHeartMCC = nan(n,1); NumberWithManualHeartPercent = zeros(n,1);
HeartAreaMAE_pp = nan(n,1); HeartAreaRMSE_pp = nan(n,1);
for q = 1:n
    ix = T.Dataset==keys.Dataset(q) & T.HeartMaskMethod==keys.HeartMaskMethod(q);
    sub = T(ix,:);
    Dataset(q)=keys.Dataset(q); HeartMaskMethod(q)=keys.HeartMaskMethod(q);
    NumberOfImages(q)=height(sub);
    MeanHeartDice_F1(q)=mean(sub.HeartDice_F1,'omitnan');
    MeanHeartIoU(q)=mean(sub.HeartIoU,'omitnan');
    MeanHeartPrecision(q)=mean(sub.HeartPrecision,'omitnan');
    MeanHeartRecall(q)=mean(sub.HeartRecall,'omitnan');
    MeanHeartSpecificity(q)=mean(sub.HeartSpecificity,'omitnan');
    MeanHeartAccuracy(q)=mean(sub.HeartAccuracy,'omitnan');
    MeanHeartBalancedAccuracy(q)=mean(sub.HeartBalancedAccuracy,'omitnan');
    MeanHeartMCC(q)=mean(sub.HeartMCC,'omitnan');
    valid = isfinite(sub.HeartAreaError_pp);
    NumberWithManualHeartPercent(q)=nnz(valid);
    if any(valid)
        e = sub.HeartAreaError_pp(valid);
        HeartAreaMAE_pp(q)=mean(abs(e));
        HeartAreaRMSE_pp(q)=sqrt(mean(e.^2));
    end
end
S = table(Dataset,HeartMaskMethod,NumberOfImages,MeanHeartDice_F1, ...
    MeanHeartIoU,MeanHeartPrecision,MeanHeartRecall,MeanHeartSpecificity, ...
    MeanHeartAccuracy,MeanHeartBalancedAccuracy,MeanHeartMCC, ...
    NumberWithManualHeartPercent,HeartAreaMAE_pp,HeartAreaRMSE_pp);
end

function S = heartLevelSummary(T)
keys = unique(T(:,{'Dataset','Heart','AnalysisMethod','ThresholdMethod'}),'rows');
n = height(keys);
Dataset = strings(n,1); Heart = strings(n,1); AnalysisMethod = strings(n,1);
ThresholdMethod = strings(n,1); NumberOfSlices = zeros(n,1);
TotalSliceWeight_mg = nan(n,1);
EstimatedInfarctWeight_mg = nan(n,1);
EstimatedNonInfarctWeight_mg = nan(n,1);
WeightedInfarctPercent = nan(n,1);
ManualInfarctWeight_mg = nan(n,1);
ManualNonInfarctWeight_mg = nan(n,1);
ManualWeightedInfarctPercent = nan(n,1);
MeanSliceInfarctPercent = nan(n,1);
InfarctWeightMAE_mg = nan(n,1);

for q = 1:n
    ix = T.Dataset==keys.Dataset(q) & T.Heart==keys.Heart(q) & ...
        T.AnalysisMethod==keys.AnalysisMethod(q) & T.ThresholdMethod==keys.ThresholdMethod(q);
    sub = T(ix,:);
    Dataset(q)=keys.Dataset(q); Heart(q)=keys.Heart(q);
    AnalysisMethod(q)=keys.AnalysisMethod(q); ThresholdMethod(q)=keys.ThresholdMethod(q);
    NumberOfSlices(q)=height(sub);
    MeanSliceInfarctPercent(q)=mean(sub.InfarctPercentOfHeart,'omitnan');

    validAuto = isfinite(sub.SliceWeight_mg) & isfinite(sub.EstimatedInfarctWeight_mg);
    if any(validAuto)
        TotalSliceWeight_mg(q)=sum(sub.SliceWeight_mg(validAuto));
        EstimatedInfarctWeight_mg(q)=sum(sub.EstimatedInfarctWeight_mg(validAuto));
        EstimatedNonInfarctWeight_mg(q)=sum(sub.EstimatedNonInfarctWeight_mg(validAuto));
        WeightedInfarctPercent(q)=100*EstimatedInfarctWeight_mg(q)/TotalSliceWeight_mg(q);
    end

    validManual = isfinite(sub.SliceWeight_mg) & isfinite(sub.ManualInfarctWeight_mg);
    if any(validManual)
        ManualInfarctWeight_mg(q)=sum(sub.ManualInfarctWeight_mg(validManual));
        ManualNonInfarctWeight_mg(q)=sum(sub.ManualNonInfarctWeight_mg(validManual));
        manualTotalWeight = sum(sub.SliceWeight_mg(validManual));
        ManualWeightedInfarctPercent(q)=100*ManualInfarctWeight_mg(q)/manualTotalWeight;
    end

    validWeightError = isfinite(sub.InfarctWeightError_mg);
    if any(validWeightError)
        InfarctWeightMAE_mg(q)=mean(abs(sub.InfarctWeightError_mg(validWeightError)));
    end
end

S = table(Dataset,Heart,AnalysisMethod,ThresholdMethod,NumberOfSlices, ...
    TotalSliceWeight_mg,EstimatedInfarctWeight_mg,EstimatedNonInfarctWeight_mg, ...
    WeightedInfarctPercent,ManualInfarctWeight_mg,ManualNonInfarctWeight_mg, ...
    ManualWeightedInfarctPercent,InfarctWeightMAE_mg,MeanSliceInfarctPercent);
end

function bestText = selectBestTrainingMethod(S)
bestText = 'NOT VALIDATED - no manual training references';
if isempty(S) || ~ismember('Dataset',S.Properties.VariableNames), return; end
ix = S.Dataset=="Training";
T = S(ix,:);
if isempty(T), return; end

if any(isfinite(T.MeanDice_F1))
    T = T(isfinite(T.MeanDice_F1),:);
    [~,ix] = max(T.MeanDice_F1);
    bestText = sprintf('%s + %s (mean training Dice/F1 = %.4f)', ...
        T.AnalysisMethod(ix),T.ThresholdMethod(ix),T.MeanDice_F1(ix));
elseif any(isfinite(T.RMSE_percentPoints))
    T = T(isfinite(T.RMSE_percentPoints),:);
    [~,ix] = min(T.RMSE_percentPoints);
    bestText = sprintf('%s + %s (training infarct RMSE = %.3f percentage points)', ...
        T.AnalysisMethod(ix),T.ThresholdMethod(ix),T.RMSE_percentPoints(ix));
end
end

function saveDiagnostics(I,gray,med,gauss,sharp,heart,info,scores,labChannels,freqDiag, ...
    masks,thresholds,ds,name,outRoot)
outDir = fullfile(outRoot,'Diagnostics',ds);
mkdirIfNeeded(outDir);

fig = figure('Visible','off','Color','w','Position',[50 50 1500 950]);
subplot(3,3,1); imshow(I); title('Original RGB');
subplot(3,3,2); imshow(heart); title('Automatic heart mask');
subplot(3,3,3);
imshow(I); hold on;
try, visboundaries(heart,'Color','g'); catch, end
title('Heart boundary overlay');
subplot(3,3,4); imshow(gray,[]); title('Grayscale');
subplot(3,3,5); imshow(info.distance,[]); title('CIELAB distance from background');
subplot(3,3,6); imshow(labChannels.Lstar,[]); title('CIELAB L* lightness');
subplot(3,3,7); imshow(labChannels.astar,[]); title('CIELAB a* red-green channel');
subplot(3,3,8); imshow(med,[]); title('Median-filter diagnostic');
subplot(3,3,9); imshow(gauss,[]); title('Gaussian-filter diagnostic');
sgtitle([ds ' - ' name ' | ' info.method],'Interpreter','none');
saveas(fig,fullfile(outDir,[name '_MainDiagnostics.png']));
close(fig);

fn = fieldnames(masks);
if ~isempty(fn)
    fig2 = figure('Visible','off','Color','w','Position',[50 50 1400 850]);
    for q = 1:numel(fn)
        subplot(ceil(numel(fn)/4),4,q);
        imshow(masks.(fn{q}));
        tv = thresholds.(fn{q});
        title(sprintf('%s | %.3f',strrep(fn{q},'_',' '),tv),'Interpreter','none');
    end
    sgtitle([ds ' - ' name ' | Infarct segmentation comparison'],'Interpreter','none');
    saveas(fig2,fullfile(outDir,[name '_SegmentationComparison.png']));
    close(fig2);
end

% Dedicated LE8-style frequency-domain diagnostic panel.
fig3 = figure('Visible','off','Color','w','Position',[50 50 1500 900]);
subplot(2,4,1); imshow(gray,[]); title('Original grayscale');
subplot(2,4,2); imshow(freqDiag.logSpectrum,[]); title('Centered log magnitude spectrum');
subplot(2,4,3); imshow(freqDiag.idealLPF,[]); title('Ideal low-pass');
subplot(2,4,4); imshow(freqDiag.butterworthLPF,[]); title('Butterworth low-pass (n=2)');
subplot(2,4,5); imshow(freqDiag.gaussianLPF,[]); title('Gaussian low-pass');
if isfield(freqDiag,'idealHPF')
    subplot(2,4,6); imshow(freqDiag.idealHPF,[]); title('Ideal high-pass magnitude');
    subplot(2,4,7); imshow(freqDiag.butterworthHPF,[]); title('Butterworth high-pass magnitude');
    subplot(2,4,8); imshow(freqDiag.gaussianHPF,[]); title('Gaussian high-pass magnitude');
else
    subplot(2,4,6); axis off;
    subplot(2,4,7); axis off;
    subplot(2,4,8); axis off;
end
sgtitle(sprintf('%s - %s | Frequency-domain analysis | pad %dx%d', ...
    ds,name,freqDiag.paddingSize(1),freqDiag.paddingSize(2)),'Interpreter','none');
saveas(fig3,fullfile(outDir,[name '_FrequencyDomain_Analysis.png']));
close(fig3);

if isfield(freqDiag,'frequencyLaplacian')
    fig4 = figure('Visible','off','Color','w','Position',[100 100 1000 450]);
    subplot(1,2,1); imshow(freqDiag.spatialLaplacian,[]); title('Spatial Laplacian magnitude');
    subplot(1,2,2); imshow(freqDiag.frequencyLaplacian,[]); title('Frequency-domain Laplacian magnitude');
    sgtitle('Spatial versus frequency-domain Laplacian');
    saveas(fig4,fullfile(outDir,[name '_Laplacian_Comparison.png']));
    close(fig4);
end
end

function saveHistograms(scores,mask,ds,name,outRoot)
names = fieldnames(scores);
nCols = ceil(sqrt(numel(names)));
nRows = ceil(numel(names)/nCols);
fig = figure('Visible','off','Color','w','Position',[100 100 1500 1000]);
for q = 1:numel(names)
    subplot(nRows,nCols,q);
    currentScore = scores.(names{q});
    histogram(currentScore(mask),50);
    xlabel('Normalized infarct score'); ylabel('Pixel count');
    title([names{q} ' score'],'Interpreter','none'); grid on;
end
sgtitle('Color-space and frequency-domain score distributions inside heart mask');
outDir = fullfile(outRoot,'Diagnostics',ds);
mkdirIfNeeded(outDir);
saveas(fig,fullfile(outDir,[name '_ScoreHistograms.png']));
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
catch ME
    warning('Edge diagnostics skipped for %s: %s',name,ME.message);
end
end

function saveSummaryGraphs(T,heartT,methodT,outRoot)
% Plot estimates separately from validation scores; never label estimates as accuracy.
testIx = T.Dataset=="Test";
testT = T(testIx,:);
if ~isempty(testT)
    groups = unique(testT(:,{'AnalysisMethod','ThresholdMethod'}),'rows');
    means = nan(height(groups),1);
    labels = strings(height(groups),1);
    for q = 1:height(groups)
        ix = testT.AnalysisMethod==groups.AnalysisMethod(q) & ...
            testT.ThresholdMethod==groups.ThresholdMethod(q);
        means(q)=mean(testT.InfarctPercentOfHeart(ix),'omitnan');
        labels(q)=groups.AnalysisMethod(q)+"-"+groups.ThresholdMethod(q);
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1200 600]);
    bar(means); set(gca,'XTick',1:numel(labels),'XTickLabel',cellstr(labels));
    xtickangle(35); ylabel('Estimated infarct area (% of heart)');
    title('Test-set infarct estimates - NOT an accuracy ranking'); grid on;
    saveas(fig,fullfile(outRoot,'Test_InfarctEstimate_Comparison.png'));
    close(fig);
end

if any(isfinite(methodT.MeanDice_F1))
    q = methodT.Dataset=="Test" & isfinite(methodT.MeanDice_F1);
    fig = figure('Visible','off','Color','w','Position',[100 100 1200 600]);
    labels = methodT.AnalysisMethod(q)+"-"+methodT.ThresholdMethod(q);
    bar(methodT.MeanDice_F1(q));
    set(gca,'XTick',1:numel(labels),'XTickLabel',cellstr(labels));
    xtickangle(35); ylabel('Mean Dice / F1');
    title('Test-set pixel-mask agreement with manual references'); ylim([0 1]); grid on;
    saveas(fig,fullfile(outRoot,'Test_Dice_Comparison.png'));
    close(fig);
end

if ~isempty(heartT)
    q = heartT.Dataset=="Test";
    ht = heartT(q,:);
    if ~isempty(ht)
        fig = figure('Visible','off','Color','w','Position',[100 100 1000 500]);
        bar(ht.HeartMaskPercentOfImage);
        set(gca,'XTick',1:height(ht),'XTickLabel',cellstr(ht.Filename));
        xtickangle(30); ylabel('Heart mask coverage (% of image)');
        title('Heart mask area coverage - not segmentation accuracy'); grid on;
        saveas(fig,fullfile(outRoot,'Test_HeartMaskCoverage.png'));
        close(fig);
    end
end

% Frequency-domain LPF candidates are ranked separately for quick review.
fdIx = startsWith(string(methodT.AnalysisMethod),"FD_") & methodT.Dataset=="Test";
fdT = methodT(fdIx,:);
if ~isempty(fdT)
    labels = fdT.AnalysisMethod + "-" + fdT.ThresholdMethod;
    if any(isfinite(fdT.MeanDice_F1))
        vals = fdT.MeanDice_F1;
        metricLabel = 'Mean Dice / F1';
    elseif any(isfinite(fdT.RMSE_percentPoints))
        vals = fdT.RMSE_percentPoints;
        metricLabel = 'RMSE (percentage points)';
    else
        vals = fdT.MeanInfarctPercent;
        metricLabel = 'Estimated infarct area (% of heart) - not accuracy';
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1400 650]);
    bar(vals); set(gca,'XTick',1:numel(labels),'XTickLabel',cellstr(labels));
    xtickangle(45); ylabel(metricLabel);
    title('Frequency-domain method comparison'); grid on;
    saveas(fig,fullfile(outRoot,'Test_FrequencyDomain_Comparison.png'));
    close(fig);
end
end
