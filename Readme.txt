MiniVIE

MATLAB Users:

The MiniVIE is a package library based on the Johns Hopkins University 
Applied Physics Laboratory Virtual Integration Environment (JHU/APL VIE).
This is a completely separate and independent code base that is designed to 
take the modularity and themes of the VIE and make them readily accessible.

The most interesting things you can do with this code base requires data acquisition
hardware to get EMG or similar signals into the environment.  However, a basic signal
simulator exists and can be used to test the functionality.

Use the tutorial.m script as a guide to see how the modules are created and 
interact with one another.

Python Users:

We have begun the process of replicating / augmenting the MiniVIE paradigm and functionality
in the python environment.  The intent is to facilitate code reuse across operating systems as 
well as enabling use on embedded hardware (e.g. Raspberry Pi).  The code based is and should maintain
python 2.x and 3.x compatibility.

At present (Aug 2016) the code based supports signal inputs from the Myo Armband, signal processing and 
classification using numpy and sklearn, and command outputs to both the physical and virtual versions 
of the JHU/APL Modular Prosthetic Limb System.




Initial Creation: 11/9/2010

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MiniVIE tutorial
% 1/25/2011  - At present the MiniVIE is more of a software API, rather
% than a turn-key application.  If you are not already familiar with object
% oriented programming in Matlab, please review the articles
% "Object-Oriented Programming" in the Matlab User's Guide under Help

%% Step 0: Setup VIE software
% Adding paths is like a #include in compiled languages.  add the Utilitiles
% path to gain access.  Note that the Class Packages (indicated by '+' in
% the directory name are part of the MiniVIE directory, so you have
% immediate access when you are in that directory, or if it is on the path
MiniVIE.configurePath();
% MiniVIE.createShortcuts(); % optional

%% Step 1: Setup Input Device
%  Create a handle to a Signal Input device and initialize it.
%  Signal inputs are designed to be modular and must inherit from the
%  Inputs.SignalInput base class.

% The particular type of input is dependant on your application and
% hardware configuration
% Uncomment one of the following to select a signals source
% SignalSource = Inputs.DaqHwDevice('nidaq','Dev1');
% SignalSource = Inputs.DaqHwDevice('mcc','0');
SignalSource = Inputs.SignalSimulator();

% If using the simulator, the box that opens up allows you to press keys to
% change the output pattern.  E.g. 'asdf' selects patterns '1234'

% The device must be initialized prior to use
SignalSource.initialize();

%% Step 2: Add input filters
% Associate filters with the input source
% TODO: This really should be on a per-channel basis
SignalSource.addfilter(Inputs.HighPass());  % set filter with default parameters
% SignalSource.addfilter(Inputs.LowPass());
SignalSource.addfilter(Inputs.Notch([180 290],30,64,1000));  % set filter with custom parameters
% SignalSource.addfilter(Inputs.MAV);

%Set the number of samples to return when getting data:
SignalSource.NumSamples = 2000;

%Get raw samples from the DAQ and access from the workspace and plot:

% myData = SignalSource.getData();
% plot(myData);

% Get filtered samples from the DAQ and access from the workspace:

% myFilteredData = SignalSource.getFilteredData();
% plot(myFilteredData);

%% Optional Step, preview signals. Close window when finished viewing
GUIs.guiSignalViewer(SignalSource); % <-- Use this to visualize signals

%% Step 3: Setup Classifier, Select Channels in use
SignalClassifier = SignalAnalysis.Lda();
TrainingData = PatternRecognition.TrainingData();
SignalClassifier.initialize(TrainingData);

SignalClassifier.setActiveChannels(1:4);  % <-- Update active channels
SignalClassifier.uiEnterClassNames

% Scenario will each have their own inputs which need to be mapped to
% virtual channels.

% E.g. Jackson Pollock app has Up Down Left Right, No Movement
% Air Guitar Hero has Index, Middle, Ring
% Breakout has Left Right, No Movement

%% Step 4: Setup TrainingInterface and data store
TrainingInterface = PatternRecognition.SimpleTrainer();

%% Step 4a: Collect New Data
TrainingInterface.NumRepetitions = 2;  % <-- Adjust (2 to 3 typical)
TrainingInterface.ContractionLengthSeconds = 2; % <-- Time to hold contraction (avoid muscle fatigue)
TrainingInterface.DelayLengthSeconds = 3; % <-- Recovery Time in seconds between contractions
TrainingInterface.EnablePictures = 0;

TrainingInterface.initialize(SignalSource,SignalClassifier,TrainingData);
TrainingInterface.collectdata(); % save prompt at end

%% Step 4b: Load Saved Data
TrainingData.loadTrainingData();

%% Step 5: Train the classifier
SignalClassifier.train();
SignalClassifier.computeError();

%% Step 6: Send data to MiniV for visualization
h = Scenarios.MiniVDisplayScenario;
h.initialize(SignalSource,SignalClassifier,TrainingData);
h.AutoOpenSpeed = 10;
h.update();
h.Verbose = 0;
start(h.Timer);

%%
return

%% Optional: Adjust the size of output filter (can be done during animation)
SignalClassifier.NumMajorityVotes = 5; % <-- Adjust majority votes [0 15]
%% Optional: Play Breakout (uses wrist flex and extend)
Presentation.MiniBreakout(SignalSource,SignalClassifier)
%% Optional: Guitar Hero Simulator

%% Optional:
% Adjust output gain for classifier
GUIs.guiGainAdjust(SignalClassifier)