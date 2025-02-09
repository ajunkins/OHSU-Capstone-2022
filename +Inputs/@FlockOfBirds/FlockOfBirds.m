classdef FlockOfBirds < handle
    %FLOCKOFBIRDS Class based on bird_io_in_matlab interface for the
    %Ascension Flock Of Birds System
    %   Class for interfacing Flock of Birds System.  This uses the
    %   bird_io_in_matlab functions for basic communications.  A single
    %   serial port interface to the bird master is used.
    %
    %   See: obj = Inputs.FlockOfBirds.Demo;
    %
    %   Usage:
    %     obj = Inputs.FlockOfBirds;
    %     obj.initialize('COM1');
    %     birdData = obj.getBirdGroup;
    %
    %   Where:
    %     x = birdData(1);
    %     y = birdData(2);
    %     z = birdData(3);
    %     Rz = birdData(4);
    %     Ry = birdData(5);
    %     Rx = birdData(6);
    %     idBird = birdData(7);
    %     F = makehgtform('translate',[x y z],...
    %         'zrotate',Rz,...
    %         'yrotate',Ry,...
    %         'xrotate',Rx);
    %
    %   Note: For USB to serial devices on windows: Check the latency
    %   setting under COM port settings.  A value of 2ms works well
    %       Port Settings -> Advanced -> Latency Timer = 2ms 
    %
    %   Revisions:
    %   Armiger 3/5/2012: Created
    %   Armiger 2/19/2018: Updated and tested with Extended Range
    %   Transmitter
    % 
    
    properties
        Bird % handle to flock of birds object
        NumSensors = 4; % Excluding Extended Range Transmitter
        F % 4x4xnumBirds.  Store the latest pose for each bird
        ERT = 1 % ERT = 0/1 Enable Extended range transmitter
    end
    properties (SetAccess = private)
        isInitialized = false;
    end
    
    methods
        function obj = FlockOfBirds
            % Creator
        end
        function initialize(obj,strComPort)
            %obj.initialize(strComPort)
            % Create serial port
            % set FoB specific params
            % begin streaming
            
            if obj.isInitialized
                fprintf('Device already initialized\n');
                return
            end
            
            if nargin < 2
                strComPort = 'COM33';
            end
            
            % Setup port
            s = serial(strComPort);          % default = 'COM1'
            set(s,'BaudRate',115200);        % default = 9600
            set(s,'RequestToSend','off');    % default = on
            set(s,'DataTerminalReady','on'); % default = on
            set(s,'InputBufferSize',512*10); % default = 512
            set(s,'Timeout',0.5);            % default = 10
            
            % Pass back handle to serial port
            obj.Bird = s;
            
            % Open port
            fopen(s);

            if obj.ERT
                numBirds = obj.NumSensors + 1;
            else
                numBirds = obj.NumSensors;
            end

            % Set mode
            for i = 1:numBirds
                fwrite(s,[240+i 89]); % set to gather position and angles
            end
            
            % no light response
            % if error, blinking lights -- ensure another serial port not
            % holding RequestToSend high
            
            % Autoconfig
            pause(0.3);   % 300 misec delay required before AutoConfig commands (p. 83)
            fwrite(s,[240+1 80 50 numBirds]);  % autoconfig for Master => bird 1
            pause(0.3);   % 300 misec delay required after AutoConfig commands (p. 83)
            
            % lights should go on
            
            % GROUP MODE
            % PARAMETERnumber = 35
            % The GROUP MODE command is only used if you have multiple Birds working together
            % in a Master/Slave configuration and you want to get data from all the Birds by talking to
            % only the Master Bird.
            
            % turn group mode on if necessary
            fwrite(s,[240+1 80 35 1])
            
            % Start Stream
            fprintf(s,'@'); % Stream Start
            
            % Set init flag
            obj.isInitialized = true;
        end
                
        function birdData = getBirdGroup(obj)
            %birdData = getBirdGroup(obj)
            % get the position and angle of the flock of birds sensor(s).
            % This assumes that stream mode is active and that there is new
            % data available.
            %
            %     x = birdData(1);
            %     y = birdData(2);
            %     z = birdData(3);
            %     Rz = birdData(4);
            %     Ry = birdData(5);
            %     Rx = birdData(6);
            %     idBird = birdData(7);
            %     F = makehgtform('translate',[x y z],...
            %         'zrotate',Rz,...
            %         'yrotate',Ry,...
            %         'xrotate',Rx);


            birdData = [];

            if ~obj.isInitialized
                error('Device not initialized\n');
            end
                        
            s = obj.Bird;
            numBytes = 13;
            
            % Read available bytes
            numAvailable = s.BytesAvailable;
            if numAvailable > 0
                [streamBytes, numRead] = fread(s,numAvailable);
            else
                % disp('No BytesAvailable on serial port')
                return
            end
            
            % fread returns bytes as double precision, convert to uint8 for
            % bitwise operations
            streamBytes = uint8(streamBytes);
            
            % find start bits
            startBits = bitget(streamBytes,8);
            
            numMsgs = sum(startBits);
            
            % get the messages with start bits
            %idxRecent = find(startBits,5,'last');
            idxStart = find(startBits);
            
            % look for messages that have the right number of bytes
            %idxValid = find(diff(idxRecent) == numBytes,2,'last');
            idxValid = find(diff(idxStart) == numBytes);
            numValid = length(idxValid);
            
            if numValid > 0
                birdData = zeros(7,numValid);
            else
                disp('No valid messages found')
                return
            end
            
            for i = 1:numValid
                msgStart = idxStart(idxValid(i));
                thisMessage = streamBytes(msgStart:msgStart+numBytes-1);
                
                % get messages
                [pos, ang, group] = Inputs.FlockOfBirds.parseBytesPositionAngles(thisMessage, 1, obj.ERT);
                
                %fprintf('Msg #%3d Bird %i\tX:%+6.3f\tY:%+6.3f\tZ:%+6.3f\t',i,group,pos);
                %fprintf('Rz:%+6.1f\tRy:%+6.1f\tRx:%+6.1f\n',ang*180/pi);
                %posAll(:,group) = pos;
                %angAll(:,group) = ang;
                birdData(1:3,i) = pos;
                birdData(4:6,i) = ang;
                birdData(7,i) = group;
                
            end
            
        end
        
        function [F, idBird] = getframes(obj)
            %[F, idBird] = getframes(obj)
            % returns 4x4xNumSamples frame transforms for Flock of
            % Birds
            %
            % idBird provides the bird id for each frame
            %
            % This function calls the getBirdGroup method, but then
            % performs an additional step of converting the output to a 4x4
            % frame
            
            % call the actual data handling method: 
            % (Note, init status is checked within getBirdGroup
            birdData = getBirdGroup(obj);

            if isempty(birdData)
                F = [];
                idBird = [];
                return
            end
            
            NumSamples = size(birdData,2);
            idBird = zeros(1,NumSamples);
            
            F = repmat(eye(4),[1 1 NumSamples]);
            for i = 1:NumSamples
                idBird = birdData(7,:);
                x = birdData(1,i);
                y = birdData(2,i);
                z = birdData(3,i);
                Rz = birdData(4,i);
                Ry = birdData(5,i);
                Rx = birdData(6,i);
                F(:,:,i) = makehgtform('translate',[x y z],...
                    'zrotate',Rz,...
                    'yrotate',Ry,...
                    'xrotate',Rx);
            end
        end
        
        function preview(obj)
            % Create a strip chart for each device showing XYZ positions
            % Setup plots
            h = cell(1,obj.NumSensors);
            for i = 1:obj.NumSensors
                h{i} = LivePlot(3,100,{'X' 'Y' 'Z'},i);
            end
            
            StartStopForm([])
            while StartStopForm
                drawnow
                birdData = obj.getBirdGroup();
                for i = 1:size(birdData,2)
                    thisSensor = birdData(7,i) - (1*obj.ERT);                  
                    h{thisSensor}.putdata(birdData(1:3,i))
                end
            end
            
        end %% preview
        function previewAngles(obj)
            % Create a strip chart for each device showing XYZ positions
            % Setup plots
            h = cell(1,obj.NumSensors);
            for i = 1:obj.NumSensors
                h{i} = LivePlot(3,100,{'X' 'Y' 'Z'},i);
            end
            
            StartStopForm([])
            while StartStopForm
                drawnow
                birdData = obj.getBirdGroup();
                for i = 1:size(birdData,2)
                    thisSensor = birdData(7,i) - (1*obj.ERT);                  
                    h{thisSensor}.putdata(birdData(4:6,i))
                end
            end
            
        end %% preview
        
        function preview3d(obj)
            
            % Provide a frame offset between Transmitter and Global
            % Coordinate Systems

            % Frame offset from Global Coordinate System to Transmitter
            T_GCS_TRNS = makehgtform('xrotate',pi);
            
            % Setup plots
            hTriad = setup_plot(obj.NumSensors,T_GCS_TRNS);
            fprintf('[%s] Starting Preview...\n',mfilename);

            StartStopForm([])
            while StartStopForm
                drawnow
                
                [T_TRNS_RCV, id] = obj.getframes();
                
                if isempty(id)
                    continue
                end
                
                if all(ishandle(hTriad))
                    for i = 1:length(id)
                        % We will plot the frame from GCS to the RCV which
                        % is T_GCS to TRNS * T_TRNS to RCV 
                        T_GCS_RCV = T_GCS_TRNS*T_TRNS_RCV(:,:,i);
                        set(hTriad(id(i)-obj.ERT),'Matrix',T_GCS_RCV);
                    end
                else
                    fprintf('[%s] Preview Stopped.\n',mfilename);
                    break
                end
            end %% plot_data
            
        end %% preview3d
        
        function close(obj)
            % cancel streaming, close and cleanup serial port
            
            % stop streaming
            try 
                % Stream stop
                fprintf(obj.Bird,'?'); % Stream Stop

                % Autoconfig
                pause(0.3);   % 300 misec delay required before AutoConfig commands (p. 83)
                fwrite(obj.Bird,[240+1 80 50 numBirds]);  % autoconfig for Master => bird 1
                pause(0.3);   % 300 misec delay required after AutoConfig commands (p. 83)
            end
            
            % close port
            try
                fclose(obj.Bird);
            end
            
            % remove port
            try 
                delete(obj.Bird);
            end
            
            % Set init flag
            obj.isInitialized = false;

        end
    end
    methods (Static = true)
        function obj = Demo
            % Requires MiniVIE Utilities
            obj = Inputs.FlockOfBirds;
            obj.NumSensors = 1;
            obj.initialize;
            obj.preview;
        end
        function s = TestSession
            %% Example command-line session
            % This provides an example command-line session, independant of
            % the class object, using native matlab commands
            
            %% Setup Port
            delete(instrfindall)
            s = serial('COM33');              % default = 'COM1'
            set(s,'BaudRate',115200);        % default = 9600
            set(s,'RequestToSend','off');    % default = on
            set(s,'DataTerminalReady','on'); % default = on
            set(s,'InputBufferSize',512*10); % default = 512
            set(s,'Timeout',0.5);            % default = 10
            
            fopen(s);
            
            %% Set initial mode
            numBirds = 4;
            
            ERT = 1;  % ERT = 0 No Extended range transmitter
            
            % Set mode
            for i = 1:numBirds+ERT
                fwrite(s,[240+i 89]); % set to gather position and angles
            end
            
            % no light response
            % if error, blinking lights -- ensure another serial port not
            % holding RequestToSend high
            
            %% Autoconfig
            pause(0.3);   % 300 misec delay required before AutoConfig commands (p. 83)
            fwrite(s,[240+1 80 50 numBirds+ERT]);  % autoconfig for Master => bird 1
            pause(0.3);   % 300 misec delay required after AutoConfig commands (p. 83)
            
            % lights should go on
            
            %% GROUP MODE
            % PARAMETERnumber = 35
            % The GROUP MODE command is only used if you have multiple Birds working together
            % in a Master/Slave configuration and you want to get data from all the Birds by talking to
            % only the Master Bird.
            
            % turn group mode on if necessary
            fwrite(s,[240+1 80 35 1])
            
            %% Loop
            StartStopForm([]);
            while StartStopForm
                drawnow;
                %% Send Request
                % request data to be sent Point or Stream
                fprintf(s,'B'); % Point command
                
                % should receive 13*numBirds bytes
                
                %% Get response
                % In the POSITION/ANGLES mode, the outputs from the POSITION and ANGLES modes
                % are combined into one record containing the following twelve bytes:
                % MSB LSB
                % 7   6   5   4   3   2   1   0   BYTE #
                % 1   X8  X7  X6  X5  X4  X3  X2  #1  LSbyte X
                % 0   X15 X14 X13 X12 X11 X10 X9  #2  MSbyte X
                % 0   Y8  Y7  Y6  Y5  Y4  Y3  Y2  #3  LSbyte Y
                % 0   Y15 Y14 Y13 Y12 Y11 Y10 Y9  #4  MSbyte Y
                % 0   Z8  Z7  Z6  Z5  Z4  Z3  Z2  #5  LSbyte Z
                % 0   Z15 Z14 Z13 Z12 Z11 Z10 Z9  #6  MSbyte Z
                % 0   Z8  Z7  Z6  Z5  Z4  Z3  Z2  #7  LSbyte Zang
                % 0   Z15 Z14 Z13 Z12 Z11 Z10 Z9  #8  MSbyte Zang
                % 0   Y8  Y7  Y6  Y5  Y4  Y3  Y2  #9  LSbyte Yang
                % 0   Y15 Y14 Y13 Y12 Y11 Y10 Y9  #10 MSbyte Yang
                % 0   X8  X7  X6  X5  X4  X3  X2  #11 LSbyte Xang
                % 0   X15 X14 X13 X12 X11 X10 X9  #12 MSbyte Xang
                
                %  The GROUP MODE address byte is only present if
                % GROUP MODE is enabled (see change value GROUP MODE).
                numBytes = 13;
                % read binary data
                [birdBytes, numRead] = fread(s,numBytes*numBirds,'uint8');
                
                %% Parse response
                if numRead < numBytes*numBirds
                    msg = sprintf('The number of bytes read [%d] was fewer than required [%d] \n',...
                        numRead,numBytes*numBirds);
                    disp(msg)
                    continue
                end
                
                [pos, ang] = Inputs.FlockOfBirds.parseBytesPositionAngles(birdBytes, 1, obj.ERT);
                
                for i = 1:numBirds
                    fprintf('Bird %i\tX:%+6.3f\tY:%+6.3f\tZ:%+6.3f\t',i,pos(:,i));
                    fprintf('Rz:%+6.1f\tRy:%+6.1f\tRx:%+6.1f\n',ang(:,i)*180/pi);
                end
                
            end
            
            %% Test Stream mode
            fprintf(s,'@'); % Stream Start
            
            %% Stream stop
            fprintf(s,'?'); % Stream Stop
            
            %% Read available bytes
            numAvailable = s.BytesAvailable;
            if numAvailable > 0
                [streamBytes, numRead] = fread(s,numAvailable);
            else
                disp('No Data')
                streamBytes = [];
            end
            
            streamBytes = uint8(streamBytes);
            
            startBits = bitget(streamBytes,8);
            
            numMsgs = sum(startBits);
            
            % get the messages with start bits
            %idxRecent = find(startBits,5,'last');
            idxRecent = find(startBits);
            
            % look for messages that have the right number of bytes
            %idxValid = find(diff(idxRecent) == numBytes,2,'last');
            idxValid = find(diff(idxRecent) == numBytes);
            
            for i = 1:length(idxValid)
                msgStart = idxRecent(idxValid(i));
                thisMessage = streamBytes(msgStart:msgStart+numBytes-1);
                
                % get messages
                
                [pos, ang, group] = Inputs.FlockOfBirds.parseBytesPositionAngles(thisMessage, 1, obj.ERT);
                
                fprintf('Msg #%3d Bird %i\tX:%+6.3f\tY:%+6.3f\tZ:%+6.3f\t',i,group,pos);
                fprintf('Rz:%+6.1f\tRy:%+6.1f\tRx:%+6.1f\n',ang*180/pi);
            end
            
        end
        
        function [pos, ang, group, msg] = parseBytesPositionAngles(birdBytes,isGroupMode, isERT)
            
            % default outputs
            [pos, ang, group] = deal([]);
            
            msg = '';
            
            if nargin < 2
                isGroupMode = true;
            end
            
            % In the POSITION/ANGLES mode, the outputs from the POSITION and ANGLES modes
            % are combined into one record containing the following twelve bytes:
            % MSB LSB
            % 7 6 5 4 3 2 1 0 BYTE #
            % 1 X8 X7 X6 X5 X4 X3 X2 #1 LSbyte X
            % 0 X15 X14 X13 X12 X11 X10 X9 #2 MSbyte X
            % 0 Y8 Y7 Y6 Y5 Y4 Y3 Y2 #3 LSbyte Y
            % 0 Y15 Y14 Y13 Y12 Y11 Y10 Y9 #4 MSbyte Y
            % 0 Z8 Z7 Z6 Z5 Z4 Z3 Z2 #5 LSbyte Z
            % 0 Z15 Z14 Z13 Z12 Z11 Z10 Z9 #6 MSbyte Z
            % 0 Z8 Z7 Z6 Z5 Z4 Z3 Z2 #7 LSbyte Zang
            % 0 Z15 Z14 Z13 Z12 Z11 Z10 Z9 #8 MSbyte Zang
            % 0 Y8 Y7 Y6 Y5 Y4 Y3 Y2 #9 LSbyte Yang
            % 0 Y15 Y14 Y13 Y12 Y11 Y10 Y9 #10 MSbyte Yang
            % 0 X8 X7 X6 X5 X4 X3 X2 #11 LSbyte Xang
            % 0 X15 X14 X13 X12 X11 X10 X9 #12 MSbyte Xang
            
            %  The GROUP MODE address byte is only present if
            % GROUP MODE is enabled (see change value GROUP MODE).
            if isGroupMode
                numBytes = 13;
            else
                numBytes = 12;
                group = 1;
            end
            
            % Parse response
            
            birdBytes = uint8(birdBytes);
            birdBytes = reshape(birdBytes,numBytes,[]);
            
            % get high bit
            startBits = bitget(birdBytes,8);
            
            if any(startBits(1,:) ~= 1) || ~all(all(startBits(2:end,:) == 0))
                msg = sprintf('The message start bits are out of order\n');
                disp(msg)
                return
            end
            
            
            LSB = uint16(birdBytes(1:2:numBytes-1,:));
            MSB = uint16(birdBytes(2:2:end,:));
            
            % shift up and typecast to int16
            MSB = bitshift(MSB,9);
            LSB = bitshift(LSB,2);
            
            unsignedWords = bitor(MSB,LSB);
            
            V = double(typecast(unsignedWords(:),'int16'));
            V = reshape(V,6,[]);
            
            % To convert the position received into inches, first convert them into a signed integer. This will give
            % you a number between -32768 and + 32767. Second, multiply by 36 if using the default range for a
            % standard transmitter or 72 if you have used the change value #3 command. If using an extended
            % range transmitter, use 144. Finally, divide the number by 32768 to get the position in inches. The
            % equation should look like this:
            % Standard Range Transmitter: (signed int * 36) / 32768
            % Standard Range Transmitter: (signed int * 72) / 32768
            % Extended Range Transmitter: (signed int * 144) / 32768
            
            if isERT
                scale = 144;
            else
                scale = 32;
            end
            
            pos = V(1:3,:) * (scale/32768.0)*2.54/100; % bird conversion factor for meters
            ang = V(4:6,:) * (180.0/32768.0)*pi/180; % bird conversion factor for radians
            
            group = double(birdBytes(end,:));
        end
    end
end

function hTriad = setup_plot(nSensors,F_ERT)
f = figure(9);
%f_setWindowState(f,'maximize');
clf(f)
hold on
hAxes = gca;
hTriad = zeros(nSensors,1);
for i = 1:nSensors
    scale = 0.1;
    % color = {'c-','m-','y-','k'}
    % hTriad(i) = f_plot_triad(eye(4),scale,color{i});
    %hTriad(i) = f_plot_triad(eye(4),scale);
    hTriad(i) = PlotUtils.triad(eye(4),scale,hAxes);
end

PlotUtils.triad(eye(4),1.0);
PlotUtils.triad(F_ERT,0.1);

axis equal
view(-75,20)
set(gca,'Projection','perspective');

xlabel('')
ylabel('')
zlabel('')

drawnow
end %% setup_plot
