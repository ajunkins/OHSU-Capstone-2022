windowSize = 15;


t = TrainingDataAnalysis;% PatternRecognition.TrainingData

t.setFeatureNames({'MAV'})
t.initialize(1,windowSize)
t.setActiveChannels(1)
% t.addTrainingData


close all

%%
for i = 1:50
windowData = 0.05*(rand(1,windowSize)-0.5);
%plot(windowData)
zc = UserConfig.getUserConfigVar('FeatureExtract.zcThreshold',0.15);
ssc = UserConfig.getUserConfigVar('FeatureExtract.sscThreshold',0.15);
features = feature_extract(windowData,windowSize,zc,ssc);
t.addTrainingData(1, features(:,1), windowData)
end

for i = 1:50
windowData = 0.9*(rand(1,windowSize)-0.5);
%plot(windowData)
zc = UserConfig.getUserConfigVar('FeatureExtract.zcThreshold',0.15);
ssc = UserConfig.getUserConfigVar('FeatureExtract.sscThreshold',0.15);
features = feature_extract(windowData,windowSize,zc,ssc);
t.addTrainingData(2, features(:,1), windowData)
end

%%

t.plot_emg_unfiltered
t.plot_emg_per_class
t.plot_features_sorted_class

%%
c = SignalAnalysis.Lda
c.initialize(t)
c.train
c.computeError



