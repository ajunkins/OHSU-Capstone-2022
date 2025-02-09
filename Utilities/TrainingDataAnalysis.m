classdef TrainingDataAnalysis < PatternRecognition.TrainingData
    % Tools for loading, viewing and manipulating training data
    % Extends TrainingData to add analysis
    %
    % Usage:
    %   s = dir('WR*.trainingData');
    %   obj = TrainingDataAnalysis;
    %   for i = 1:length(s)
    %       obj.loadTrainingData(s(i).name);
    %       obj.plot_emg_per_class()
    %   end
    properties
        fullFileName = '';
    end
    methods
        function obj = TrainingDataAnalysis(fileName)
            % Creator
            if nargin < 1
                fprintf('[%s] Creating Training Data Analysis Object\n',mfilename);
            else
                fprintf('[%s] Creating Training Data Analysis Object from file: "%s"\n',mfilename,fileName);
                obj.loadTrainingData(fileName);
            end
        end
        function [success, fullFile] = loadTrainingData(obj,fname)
            % Overload load method to store filename for use in plotting /
            % analysis
            
            if nargin < 2
                [success, fullFile] = loadTrainingData@PatternRecognition.TrainingData(obj);
            else
                [success, fullFile] = loadTrainingData@PatternRecognition.TrainingData(obj,fname);
            end
            
            obj.fullFileName = fullFile;
        end
        function plot_emg_unfiltered(obj,channels,prefix)
            % Plot all the emg data in the training file - unfiltered
            %
            % Usage:
            % plot_emg_unfiltered(obj,channels);
            
            if nargin < 2
                channels = obj.ActiveChannels;
            end
            if nargin < 3
                % set output file prefix
                prefix = '';
            end
            
            chEmg = obj.getStichedData(channels)';
            %chEmg = obj.getContinuousData(channels);
            
            % Plot result
            clf
            t = (1:size(chEmg,1)) ./ obj.SampleRate;
            h = plot(t,chEmg);
            
            %numChannels = length(channels);
            %c = distinguishable_colors(numChannels);
            c = distinguishable_colors(32);
            for i = 1:length(h)
                set(h(i),'Color',c(channels(i),:));
            end
            
            C = num2cell(channels);
            legend(cellfun(@(x)sprintf('%2d',x),C,'UniformOutput',false));
            
            [p, f, e] = fileparts(obj.fullFileName);
            dataLabel = [f '_unfiltered'];
            title(dataLabel, 'Interpreter','None');
            xlabel('Time (s)')
            ylabel('EMG Amplitude')
            
            % Save output
            outFile = [prefix dataLabel '.tif'];
            if ~exist(outFile,'file')
                saveas(gcf,outFile);
            else
                fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
            end
            
        end
        function plot_emg_filtered(obj,channels,prefix)
            % Plot all the emg data in the training file - filtered
            %
            
            if nargin < 2
                channels = obj.ActiveChannels;
            end
            if nargin < 3
                % set output file prefix
                prefix = '';
            end
            
            
            %chEmg = obj.getContinuousData(channels);
            chEmg = obj.getStichedData(channels)';
            chEmg = TrainingDataAnalysis.filter_data(chEmg);
            
            % Plot result
            clf
            t = (1:size(chEmg,1)) ./ obj.SampleRate;
            h = plot(t,chEmg);
            
            numChannels = length(channels);
            %c = distinguishable_colors(numChannels);
            c = distinguishable_colors(16);
            for i = 1:length(h)
                set(h(i),'Color',c(channels(i),:));
            end
            
            C = num2cell(channels);
            legend(cellfun(@(x)sprintf('%2d',x),C,'UniformOutput',false));
            
            [p, f, e] = fileparts(obj.fullFileName);
            dataLabel = [f '_filtered'];
            title(dataLabel, 'Interpreter','None');
            xlabel('Time (s)')
            ylabel('EMG Amplitude')
            
            % Save output
            outFile = [prefix dataLabel '.tif'];
            if ~exist(outFile,'file')
                saveas(gcf,outFile);
            else
                fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
            end
            
        end
        function plot_emg_per_class(obj,channels,activeClasses)
            % Plot numClasses x 1 subplots, with filtered EMG for each
            % class
            %
            % Usage:
            % TrainingDataAnalysis.plot_emg_per_class('WR_TR01_*.dat');
            
            if nargin < 2
                channels = obj.ActiveChannels;
            end
            
            % get data and labels
            [emgDataRaw, emgLabels] = obj.getStichedData(channels);
            emgDataRaw = emgDataRaw';

            if nargin < 3
                activeClasses = unique(emgLabels);
            end

            % Setup Labels
            [~, thisFile, ~] = fileparts(obj.fullFileName);
            dataLabel = [thisFile '_classEmg'];
            
            % filter the data
            chEmg = TrainingDataAnalysis.filter_data(emgDataRaw,0);
                        
            classNames = obj.ClassNames;
            
            clf
            ch = channels;
            c = distinguishable_colors(16);
            
            for i = 1:length(activeClasses)
                iClass = activeClasses(i);
                thisClass = iClass == emgLabels;
                h = subplot(length(activeClasses),1,i);
                hold on
                
                strClass = classNames{iClass};
                acronymClass = upper(strClass(regexp(strClass, '\<.')));
                ylabel(acronymClass)
                set(h,'YTick',[]);
                if i < length(activeClasses)
                    %set(h,'XTick',[]);
                else
                    xlabel('Time t, sec');
                end
                
                classEmg = chEmg(thisClass,:);
                t = (1:size(classEmg,1)) / obj.SampleRate;
                hLines = plot(t,classEmg);
                
                c = distinguishable_colors(16);
                for iLine = 1:length(hLines)
                    set(hLines(iLine),'Color',c(channels(iLine),:));
                end
                
                ylim([-1.2 1.2]);
                if i == 1
                    title(dataLabel,'Interpreter','None');
                end
            end
            drawnow
            
            
            % Save output
            outFile = [dataLabel '.tif'];
            if ~exist(outFile,'file')
                saveas(gcf,outFile);
            else
                fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
            end
            
        end
        
        function dataLabel = plot_emg_with_breaks(obj,doFilter,channels)
            
            [p, f, e] = fileparts(obj.fullFileName);
            dataLabel = [f '_emgChannels'];

            % The training data object contains partially overlapping
            % frames of emg time data.  Each frame is also labeled with a
            % class label.  The class label number should correspond to the
            % ClassNames property.
            %MPL_0320140612_133016.trainingData
            
            % Class labels here will be 'filtered' by those that are
            % enabled
            l = obj.getClassLabels;
            classNames = obj.ClassNames;
            
            uniqueLabels = unique(l);
            nClassIds = length(uniqueLabels);            
            nClassNames = length(classNames);
            
            % it's common to have more named classes than label ids (some
            % classes might not have been trained).  It's uncommon to have
            % more ids than labels
            % this is an error state in which the numeric labels don't
            % match the named classes.  we can try to recover by
            % disabling any unmatched labels
            if nClassIds > nClassNames
                warning('TrainingData:MismatchedLabels','More numeric data labels found than named data labels')
                unmatched = setdiff(uniqueLabels,1:nClassNames);
                for i = unmatched
                    obj.disableLabeledData(i);
                end
                
                % % recompute now that unmatched data is disabled
                % l = obj.getClassLabels;
                % uniqueLabels = unique(l);
                % nClassIds = length(uniqueLabels);
                % nClassNames = length(classNames);
                
            end
            
            if nargin < 3
                channels = obj.ActiveChannels;
            end

            % Return empty if no active channels
            if isempty(channels)
                dataLabel = [];
                return    
            end
            
            % get data and labels
            [emgDataRaw, emgLabels] = obj.getStichedData(channels,1);
            
            if isempty(emgDataRaw)
                dataLabel = [];
                return
            end
            emgDataRaw = emgDataRaw';
            
            chEmg = emgDataRaw;
            l = emgLabels;
            if nargin < 2
                doFilter = false;
            end
            
            if doFilter
                chEmg = TrainingDataAnalysis.filter_data(chEmg,[]);
                dataLabel = strcat(dataLabel,'_filtered');
            end
            
            c = distinguishable_colors(max(channels));
            
            w = obj.WindowSize;
            classChange = [find(diff(l) ~= 0) length(l)];
            
            for i = 1:obj.NumClasses
                strClass = obj.ClassNames{i};
                acronymClass = upper(strClass(regexp(strClass, '\<.')));
                acronymClassname{i} = acronymClass;
            end
            
            
            xTickLabels = acronymClassname(l(classChange));
            xTick = mean( [[0 classChange(1:end-1)]; classChange] );
            
            clf
            h = gca;
            hold on
            
            for i = 1:length(channels)
                plot(chEmg(:,i),'Color',c(i,:))
            end
            title(sprintf('%s Total=%d Active=%d',dataLabel,...
                length(obj.getClassLabels),length(obj.getAllClassLabels)) , 'Interpreter','None');
            set(h,'XTick',xTick)
            set(h,'XTickLabel',xTickLabels);
            
            if doFilter
                rng = max(std(chEmg)) * 10;
                ylim([-rng rng])
            end
            
            yLimits = ylim;
            
            xBreaks = classChange;
            xBreaks = [xBreaks; xBreaks; nan(size(xBreaks))];
            yBreaks = repmat([yLimits(1); yLimits(2); NaN],1,size(xBreaks,2));
            
            plot(xBreaks,yBreaks,'k')
            
            C = num2cell(channels);
            legend(cellfun(@(x)sprintf('%2d',x),C,'UniformOutput',false));
            
            
            if nargout < 1
                % Save output
                outFile = [dataLabel '.png'];
                if ~exist(outFile,'file')
                    win = get(gcf,'Position');
                    pad = 50;
                    screencapture(gcf,[pad pad win(3)-(1*pad) win(4)-(1*pad)],outFile);
                else
                    fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
                end
            end
        end
        function dataLabel = plot_features_sorted_class(obj)
            
            %%
            
            % create title for plot
            [p, f, e] = fileparts(obj.fullFileName);
            dataLabel = [f '_features'];
            titleTxt = sprintf('%s Total=%d Active=%d',dataLabel,...
                length(obj.getClassLabels),length(obj.getAllClassLabels));
            
            % get the long list of numerical ids for classes trained
            class_id = obj.getClassLabels;

            % convert to text strings
            class_list = obj.ClassNames(class_id);
            
            % sort text strings (which also groups)
            [sorted_list, old_order] = sort(class_list);
            
            % convert the numerical ids as well
            sorted_id = class_id(old_order);
            
            % find which classes have training data
            used_classes = unique(sorted_list);
            
            % Convert long format class names to acronyms
            for i = 1:length(used_classes)
                strClass = used_classes{i};
                acronymClass = upper(strClass(regexp(strClass, '\<.')));
                acronymClassname{i} = acronymClass;
            end
            
            % find where the class names change, so we can plot by group
            class_transition_id = find(diff(sorted_id) ~= 0);
            
            % define the x labels as well as x location for each
            xTickLabels = acronymClassname;
            xTick = mean( [[0 class_transition_id]; [class_transition_id length(sorted_id)]] );
            
            clf
            
            for iFeature = 1:obj.NumFeatures
                
                h = subplot(obj.NumFeatures,1,iFeature);
                hold on
                
                if iFeature == 1 % first row
                    title(titleTxt , 'Interpreter','None')
%                     ylim([0 1])
                end
                
                f = obj.getFeatureData;
                
                c = distinguishable_colors(obj.NumActiveChannels);
                
                for i = 1:obj.NumActiveChannels
                    iCh = obj.ActiveChannels(i);
                    lineData = squeeze(f(iCh,iFeature,old_order));
                    plot(lineData,'Color',c(i,:))
                end
                
                if iFeature == obj.NumFeatures %last row
                    set(h,'XTick',xTick)
                    set(h,'XTickLabel',xTickLabels);
                else
                    set(h,'XTick',[])
                end
                
                ylabel(obj.FeatureNames{iFeature},'Interpreter','None')
                
                yLimits = ylim;
                
                xBreaks = [0 class_transition_id length(sorted_id)];
                xBreaks = [xBreaks; xBreaks; nan(size(xBreaks))];
                yBreaks = repmat([yLimits(1); yLimits(2); NaN],1,size(xBreaks,2));
                xlim([0 length(sorted_id)])
                plot(xBreaks,yBreaks,'k')
            end
            
            if nargin < 1
                save_output(dataLabel);
            end
            
            
        end
        function plot_pca(obj)
            % Pass thru for plotting principal components
            GUIs.guiPlotPca(obj)
        end
    end
    methods (Static = true)
        function plot_mav_per_class(filterSpec,channels)
            % Plot all the emg data in the training file - unfiltered
            %
            % Usage:
            % TrainingDataAnalysis.plot_emg_per_class('WR_TR01_*.dat');
            set(0,'DefaultLineLineWidth',3)
            if nargin < 1
                filterSpec = '*.dat';
            end
            
            % Load Data
            [d fileName] = TrainingDataAnalysis.load_data(filterSpec);
            
            if nargin < 2
                channels = d.activeChannels;
            end
            
            filteredData = TrainingDataAnalysis.filter_data(d.emgData);
            %%
            % channels = 16
            
            % plot Data
            clf
            ch = channels;
            c = get(gca, 'ColorOrder');
            for iClass = 1:length(d.classNames)
                thisClass = iClass == d.classLabelId;
                
                h = subplot(length(d.classNames),1,iClass);
                hold on
                ylabel(d.classNames{iClass})
                set(h,'YTick',[]);
                set(h,'XTick',[]);
                ylim([-2 2]);
                
                if ~any(thisClass)
                    continue
                end
                classEmgFrames = filteredData(1:16,:,thisClass);
                mav = squeeze(mean(abs(classEmgFrames),2));
                
                ylabel(d.classNames{iClass})
                
                hLines = plot(mav');
                
                c = distinguishable_colors(16);
                for i = 1:length(hLines)
                    set(hLines(i),'Color',c(i,:));
                end
                
                %                 xBreaks = size(d.emgData,2):size(d.emgData,2):sum(thisClass)*size(d.emgData,2);
                %                 xBreaks = [xBreaks; xBreaks; nan(size(xBreaks))];
                %                 yBreaks = repmat([-10; 10; NaN],1,size(xBreaks,2));
                %plot(xBreaks(:),yBreaks(:),'k-');
                
                %%
                set(hLines,'Visible','off');
                set(hLines(ch),'Visible','on');
                ylabel(d.classNames{iClass})
                %     xlim([0 size(emgData,2)]);
                ylim([-.1 1])
                if iClass == 1
                    title(fileName,'Interpreter','None');
                end
            end
            
            
            [p f e] = fileparts(fileName);
            dataLabel = [f '_classMav'];
            
            % Save output
            outFile = [dataLabel '.tif'];
            if ~exist(outFile,'file')
                saveas(gcf,outFile);
            else
                fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
            end
            
        end
        function features3D = plot_one_class_emg(filterSpec,channels)
            
            if nargin < 1, filterSpec = '*.dat'; end
            if nargin < 2, channels = 1:16; end
            
            % Load Data
            [d fileName] = TrainingDataAnalysis.load_data(filterSpec);
            classLabelId = d.classLabelId;
            classNames = d.classNames;
            % filter
            filteredData = TrainingDataAnalysis.filter_data(d.emgData);
            %channels = d.activeChannels;
            
            % regenerate features (overwrite those loaded from the data file)
            zc = UserConfig.getUserConfigVar('FeatureExtract.zcThreshold',0.15);
            ssc = UserConfig.getUserConfigVar('FeatureExtract.sscThreshold',0.15);

            [numChannelsAll, numSamplesPerWindow, numSamples] = size(d.emgData);
            numFeatures = 4;
            features3D = zeros(numChannelsAll,numFeatures,numSamples);
            for i = 1:size(filteredData,3)
                features3D(:,:,i) = feature_extract(...
                    filteredData(:,:,i),size(filteredData,2),...
                    zc,ssc);
            end
            return
            %%
            clf
            ch = channels;
            
            c = [get(gca, 'ColorOrder');get(gca, 'ColorOrder')];
            for iClass = 1:length(classNames);
                for iFeature = 3;
                    for iChannel = ch
                        thisClass = iClass == classLabelId;
                        classEmg = filteredData(iChannel,:,thisClass);
                        
                        h = subplot(length(ch),2,2*(iChannel-1)+1);
                        hLines = plot(reshape(classEmg,1,[])','Color',c(iChannel,:));
                        set(h,'YTick',[]);
                        set(h,'XTick',[]);
                        %     xlim([0 size(emgData,2)]);
                        ylim([-2.5 2.5])
                        if iChannel == 1
                            title({fileName classNames{iClass}},'Interpreter','None');
                        end
                        
                        
                        featureData = squeeze(features3D(iChannel,iFeature,thisClass));
                        %             if any(iFeature == [2 3 4])
                        %                 %featureData = log(featureData+1);
                        %                 featureData = exp(featureData)-1;
                        %             else
                        %                 %featureData = featureData*3;
                        %                 featureData = featureData*1;
                        %             end
                        
                        
                        h = subplot(length(ch),2,2*(iChannel-1)+2);
                        hLines = plot(featureData,'Color',c(iChannel,:));
                        set(h,'YTick',[]);
                        set(h,'XTick',[]);
                        %     xlim([0 size(emgData,2)]);
                        ylim([-1 500])
                        if iChannel == 1
                            title({fileName [classNames{iClass} ...
                                [' - Feature #' num2str(iFeature) ' Thresh=' num2str(thresh)]]},...
                                'Interpreter','None');
                        end
                        
                        
                    end
                    
                    drawnow
                    saveas(gcf,fullfile('img\',[fileName '-' classNames{iClass} '-Feature' num2str(iFeature) '.tif']),'tif');
                    
                end
            end
            
            
            
            %%
            return
            
            for iFrame = 1:length(classLabelId)
                
                plot(emgData(1:8,:,i)')
                ylim([-2.5 2.5])
                drawnow
            end
            
        end
        function [d, fileName, pathName] = load_data(filterSpec)
            % load data from filename of filter specification e.g. *.dat
            if exist(filterSpec,'file')
                fileName = filterSpec;
                [pathName] = fileparts(which(fileName));
            else
                [fileName, pathName] = uigetfile(filterSpec);
            end
            d = load(fullfile(pathName,fileName),'-mat');
            assert(~isempty(d.signalData),'Signal Data Not Found');
        end
        function filteredData = filter_data(dataIn,reflectValue)
            
            if nargin < 2
                reflectValue = 1.2;
            end
            
            % filter Data
            % note that if filtfilt is used, the filter order is doubled
            %HPF = Inputs.HighPass(10,2,1000);
            Fs = 1000;
            HPF = Inputs.HighPass(20,3,Fs);
            %NF = Inputs.Notch([120 240 360],64,1,Fs);
            
            HPF.ReflectOnApply = ~isempty(reflectValue);
            HPF.ReflectValue = reflectValue;
            
            filteredData = HPF.apply(double(dataIn));
            %filteredData = NF.apply(filteredData);
        end
        
        function hData = batchLoadTrainingData(dataPath,filter)
            % Load all training data in directory and return an array of
            % TrainingData Objects
            %
            % Inputs: 
            %   dataPath - Full path to directory containing *.trainingData files
            %
            % Usage:
            %   TrainingDataAnalysis.batchLoadTrainingData('c:\data\Myo_01\')
            
            if nargin < 2
                filter = '*.trainingData';
            end
                
            s = rdir(fullfile(dataPath,filter));

            % sort by date
            [~,idx] = sort([s.datenum]);
            s = s(idx);

            if isempty(s)
                warning('No files found in: %s',fullfile(dataPath,'*.trainingData'));
                hData = [];
                return
            end
            
            for iFile = 1:length(s)
                try 
                    hData(iFile) = TrainingDataAnalysis(s(iFile).name); %#ok<AGROW>
                catch ME
                    fprintf('[%s.m] Error Loading %s. Aborting.\n',mfilename,s(iFile).name);
                    warning(ME.message)
                    return
                end
            end %iFiles
            
        end
        function batchRunQuickLook(pathName)
            %%
            % get files
            
            if nargin < 1
                pathName = pwd;%'C:\tmp\MPL_01_WD_R_MiniVIE\MiniVIE\';
            end
            
            s = dir(fullfile(pathName,'*.trainingData'));
            
            obj = TrainingDataAnalysis;
            
            set(0,'defaultfigurecolor',[1 1 1]);
            figure(99)
            drawnow
            jFrame = get(handle(gcf),'JavaFrame');
            jFrame.setMaximized(true);
            drawnow
            
            for i = 1:length(s)
                obj.loadTrainingData(fullfile(pathName,s(i).name));
                obj.setActiveChannels(1:16);
                drawnow
                dataLabel = plot_emg_with_breaks(obj,0);
                save_output(dataLabel,pathName);
                drawnow
                dataLabel = plot_emg_with_breaks(obj,1);
                save_output(dataLabel,pathName);
                drawnow
                dataLabel = obj.plot_features_sorted_class();
                save_output(dataLabel,pathName);
                drawnow
                
                % Save output
                [p, f, e] = fileparts(obj.fullFileName);
                dataLabel = [f '_PCA'];
                
                hAxes = GUIs.guiPlotPca(obj);
                titleTxt = sprintf('%s Total=%d Active=%d',dataLabel,...
                    length(obj.getClassLabels),length(obj.getAllClassLabels));
                title(hAxes(1),titleTxt,'Interpreter','None');
                drawnow
                save_output(dataLabel,pathName);
            end
        end
    end
end

function plot_each_emg_channel

%%

channels = obj.ActiveChannels;
chEmg = obj.getContinuousData(channels);
chEmg = TrainingDataAnalysis.filter_data(chEmg);

numChannels = length(channels);
c = distinguishable_colors(numChannels);
clf
for i = 1:numChannels
    h = subplot(numChannels,1,i)
    plot(chEmg(:,i),'Color',c(i,:))
    
    set(h,'YTick',[]);
    set(h,'XTick',[]);
    ylabel(h,num2str(channels(i)))
    rng = 0.7;
    ylim([-rng rng]);
    
end



end
function plot_each_emg_channel_with_breaks

%%

channels = obj.ActiveChannels;
chEmg = obj.getContinuousData(channels);
chEmg = TrainingDataAnalysis.filter_data(chEmg);

c = distinguishable_colors(obj.NumActiveChannels);
l = obj.getAllClassLabels;
w = obj.WindowSize;
classChange = [find(diff(l) ~= 0) length(l)];

for i = 1:obj.NumClasses
    strClass = obj.ClassNames{i};
    acronymClass = upper(strClass(regexp(strClass, '\<.')));
    acronymClassname{i} = acronymClass;
end


xTickLabels = acronymClassname(l(classChange));
xTick = mean( [[0 classChange(1:end-1)*w]; classChange*w] );

clf
[p, f, e] = fileparts(obj.fullFileName);
dataLabel = [f '_emgChannels'];

for i = 1:obj.NumActiveChannels
    h = subplot(obj.NumActiveChannels,1,i);
    plot(chEmg(:,i),'Color',c(i,:))
    
    hold on
    xBreaks = classChange * w;
    xBreaks = [xBreaks; xBreaks; nan(size(xBreaks))];
    yBreaks = repmat([-10; 10; NaN],1,size(xBreaks,2));
    
    plot(xBreaks,yBreaks,'k')
    
    set(h,'YTick',[]);
    if i == 1
        title(dataLabel, 'Interpreter','None');
    end
    if i == obj.NumActiveChannels
        set(h,'XTick',xTick)
        set(h,'XTickLabel',xTickLabels)
    else
        set(h,'XTick',[]);
    end
    ylabel(h,num2str(channels(i)))
    rng = 4;
    ylim([-rng rng]);
    
end

% Save output
outFile = [dataLabel '.png'];
if ~exist(outFile,'file')
    win = get(gcf,'Position');
    pad = 50;
    screencapture(gcf,[pad pad win(3)-(1*pad) win(4)-(1*pad)],outFile);
else
    fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
end

end

function save_output(dataLabel,pathName)
            % Save output
            outFile = fullfile(pathName,[dataLabel '.png']);
            if ~exist(outFile,'file')
                win = get(gcf,'Position');
                pad = 50;
                screencapture(gcf,[pad pad win(3)-(1*pad) win(4)-(1*pad)],outFile);
            else
                fprintf('[%s] File "%s" already exists\n',mfilename,outFile);
            end
end

