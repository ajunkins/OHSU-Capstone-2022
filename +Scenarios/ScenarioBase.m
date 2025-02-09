classdef ScenarioBase < Common.MiniVieObj
    %SCENARIOBASE Base class for scenario objects
    %   This class is basically a plant model that has a built in timer.
    %   When executed, it will get data from the signal source, classify it
    %   to extract intent, map the intent to the appropriate control
    %   variables (and apply rate limiting and range limiting) and then
    %   send the data to the appropriate sink
    %
    % Note that the joint angles will be saved on closed and loaded on init
    % in the user's tempdir directory
    %
    % Note for finger only classification set the AutoOpenSpeed parameter
    % to > 0 to release automatically
    %
    % 17Jan2012 Armiger: Created
    % 15Jul2012 Armiger: Added load/save from tempdir
    
    properties
        SignalSource;
        SignalClassifier;
        ArmStateModel;
        Timer;
        
        % For Grasp Based control
        GraspId;
        GraspValue = 0;     % normalized position
        GraspVelocity = 0;
        GraspLocked = 0;
                
        % percentage that allows changing of grasps.  Beyond this, the
        % grasp will be 'locked' in
        GraspChangeThreshold = 0.2;

        % percentage that allows changing of rocs.  Beyond this, the
        % roc will be 'locked' in
        RocChangeThreshold = 0.05;
        
        % Counter for opening hand the remaining to full rest position
        GraspChangeCounter = 0;
        GraspChangeCount = 40;
        
        % For opening hand without Hand Open class:
        AutoOpenSpeed = 0;

        % Stores a structure format of roc table data loaded from xml.  See
        % MPL.RocTable
        RocTable;
        RocTableXmlFilename;
        
        Verbose = 1;

        
        % DEPRECATED
        % Store joint state parameters
        JointAnglesDegrees;
        JointVelocity;
        TempFileName = 'jointAngles';
        
        Intent = [];
        
    end
    
    properties (Constant = true)
        constrain = @(X,minX,maxX) min(max(X,minX),maxX);
    end
    
    methods
        function initialize(obj,SignalSource,SignalClassifier)
            % initialize(obj,SignalSource,SignalClassifier)
            obj.SignalSource = SignalSource;
            obj.SignalClassifier = SignalClassifier;
            
            if isempty(obj.SignalClassifier.getClassNames)
                % Must have some classes defined to do anyting
                msg = 'No Output Classnames Specified';
                error(msg);
            end
            
            obj.ArmStateModel = Controls.ArmStateModel();
            
            obj.Timer = UiTools.create_timer(mfilename,@(src,evt)update(obj));
            period = 0.05;
            fprintf('[%s] Setting timer refresh rate to %4.2f s\n',mfilename,period);
            obj.Timer.Period = period;
            
            % % Load previous angles.  Does this work with the joint state
            % % machine?
            % jointAngles = UiTools.load_temp_file(obj.TempFileName);
            % if isempty(jointAngles)
            %     obj.JointAnglesDegrees = zeros(size(action_bus_definition));
            % else
            %     obj.JointAnglesDegrees = jointAngles;
            % end
            % obj.JointVelocity = zeros(size(action_bus_definition));
            obj.getRocConfig();

        end
        function start(obj)
            % Start the main timer function for the scenario.  the timer
            % function calls the update() method at the specified frequency
            
            if ~isempty(obj.Timer) && ~isvalid(obj.Timer)
                % Use the isvalid method to determine if a timer object exists in memory, but is not cleared from the workspace.
                fprintf('[%s] Timer object exists in memory, but is invalid.  Re-initialize module.\n',mfilename);
                return
            end
            
            % && ishandle(obj.Timer) <-- this is always false
            if ~isempty(obj.Timer) && strcmpi(obj.Timer.Running,'off')
                % call the update funciton once manually.
                % if there is an error this will help debug
                obj.update();
                start(obj.Timer);
            end
        end
        function stop(obj)
            % Stop the main timer function for the scenario.  the timer
            % function calls the update() method at the specified frequency
            
            
            % && ishandle(obj.Timer) <-- this is always false
            
            if ~isempty(obj.Timer) && ~isvalid(obj.Timer)
                % Use the isvalid method to determine if a timer object exists in memory, but is not cleared from the workspace.
                obj.Timer = [];
            end
            
            if ~isempty(obj.Timer) && strcmpi(obj.Timer.Running,'on')
                stop(obj.Timer);
            end
        end
        function update(obj)
            %update(obj)
            % Called by timer function, Get intent and update arm
            
            % Use a try block to display more info if an error occurs
            try
                % Step 1: Get Intent
                [className,prSpeed] = getIntentSignals(obj);
                
                % Step 2: Convert Intent to limb commands
                obj.generateUpperArmCommand(className,prSpeed);
                obj.generateGraspCommand(className,prSpeed);
                
            catch ME
                UiTools.display_error_stack(ME);
            end
            
        end % update
        function [className,prSpeed,rawEmg,windowData,features2D,voteDecision] = getIntentSignals(obj)
            % Perform classification with error checking
            
            % Init output variables
            [className,prSpeed,rawEmg,windowData,features2D,voteDecision] = deal([]);
            
            % Verify inputs
            if isempty(obj.SignalSource)
                if obj.Verbose > 0
                    disp('No Signal Source');
                end
                return
            end
            if isempty(obj.SignalClassifier)
                if obj.Verbose > 0
                    disp('No Signal Classifier');
                end
                return
            end
            
            % Get intent from data stream
            [classOut,voteDecision,className,prSpeed,rawEmg,windowData,features2D] ...
                = getIntent(obj.SignalSource,obj.SignalClassifier);
            
            obj.Intent.classOut = classOut;
            obj.Intent.voteDecision = voteDecision;
            obj.Intent.className = className;
            obj.Intent.prSpeed = prSpeed;
            obj.Intent.rawEmg = rawEmg;
            obj.Intent.windowData = windowData;
            obj.Intent.features2D = features2D;
            
            if obj.Verbose > 0
                % Display command line output
                
                % could be number or enum type
                RocVal = obj.ArmStateModel.getRocVal;
                RocId = obj.ArmStateModel.getRocId;
                if isnumeric(RocId)
                    RocId = num2str(RocId);
                else
                    RocId = char(RocId);
                end
                
                fprintf('Class=%2d; Vote=%2d; Class = %24s; S=%6.4f \t | Roc=%s; P=%6.4f',...
                    classOut,voteDecision,className,prSpeed,RocId,RocVal);
            end
            
        end
        function generateUpperArmCommand(obj,className,prSpeed)
            % assign velocities to the joint state model based on the
            % classified signal name
            if isempty(className)
                return
            end
            
            s = obj.ArmStateModel;
            globalGain = 3;
            prSpeed = prSpeed * globalGain;
            
            % ensure velocities are stopped
            s.velocity(1:7) = 0;
            
            % Note gains can/should be adjusted using guiAdjustGains
            rocId = [];
            rocV = [];
            switch className
                case {'No Movement' 'Rest'}
                    % debug for endpoint
                    if length(s.structState(8).State) >= 6;
                        s.structState(8).State = [0 0 0 0 0 0];
                    end
                case {'Shoulder Flexion'}
                    s.setVelocity(MPL.EnumArm.SHOULDER_FE,+prSpeed);
                case {'Shoulder Extension'}
                    s.setVelocity(MPL.EnumArm.SHOULDER_FE,-prSpeed);
                case {'Shoulder Adduction'}
                    s.setVelocity(MPL.EnumArm.SHOULDER_AB_AD,+prSpeed);
                case {'Shoulder Abduction'}
                    s.setVelocity(MPL.EnumArm.SHOULDER_AB_AD,-prSpeed);
                case {'Humeral Internal Rotation'}
                    s.setVelocity(MPL.EnumArm.HUMERAL_ROT,+prSpeed);
                case {'Humeral External Rotation'}
                    s.setVelocity(MPL.EnumArm.HUMERAL_ROT,-prSpeed);
                case {'Elbow Flexion' 'Elbow Up' 'Elbow_Flex'}
                    s.setVelocity(MPL.EnumArm.ELBOW,+prSpeed);
                case {'Elbow Extension' 'Elbow Down' 'Elbow_Extend'}
                    s.setVelocity(MPL.EnumArm.ELBOW,-prSpeed);
                case {'Pronate' 'Wrist Rotate In'}
                    s.setVelocity(MPL.EnumArm.WRIST_ROT,+prSpeed);
                case {'Supinate' 'Wrist Rotate Out'}
                    s.setVelocity(MPL.EnumArm.WRIST_ROT,-prSpeed);
                case {'Down','Hand Down', 'Ulnar Deviation','Wrist Adduction','Ulnar_Dev'}
                    s.setVelocity(MPL.EnumArm.WRIST_AB_AD,+prSpeed);
                case {'Up', 'Hand Up', 'Radial Deviation','Wrist Abduction','Radial_Dev'}
                    s.setVelocity(MPL.EnumArm.WRIST_AB_AD,-prSpeed);
                case {'Left' 'Wrist Flex' 'Wrist Flex In' 'Wrist_Flex'}
                    s.setVelocity(MPL.EnumArm.WRIST_FE,+prSpeed);
                case {'Right' 'Wrist Extend' 'Wrist Extend Out' 'Wrist_Extend'}
                    s.setVelocity(MPL.EnumArm.WRIST_FE,-prSpeed);
                case {'Whole Arm Roc 1 FWD' 'Whole Arm Roc FWD'}
                    rocId = 16;
                    rocV = 0.1;
                case {'Whole Arm Roc 1 REV' 'Whole Arm Roc REV'}
                    rocId = 16;
                    rocV = -0.3;
                case 'Whole Arm Roc 2 FWD'
                    rocId = 17;
                    rocV = 0.2;
                case 'Whole Arm Roc 2 REV'
                    rocId = 17;
                    rocV = -0.2;
                case 'Whole Arm Roc 3 FWD'
                    rocId = 18;
                    rocV = 0.4;
                case 'Whole Arm Roc 3 REV'
                    rocId = 18;
                    rocV = -0.4;
            end
            
            % If classified a roc id, ensure it's in the table and then
            % assign it to the appropriate joints:
            
            % Note it's possible to not have a roc table since this is
            % really only valid for the MPL and may not apply to all
            % scenarios
            
            if ~isempty(obj.RocTable) && ~isempty(rocId)

                availableIds = [obj.RocTable(:).id];
                if ~ismember(rocId,availableIds)
                    warning('Requested ROC #%d not in table %s\n',rocId,obj.RocTableXmlFilename);
                end
                
                % TODO add in ROC position limiting
                % s.structState(s.RocStateId).Value < obj.RocChangeThreshold && (rocV > 0)
                
                
                % assign id to s.JointState(rocJoints).RocId
                rocJoints = obj.RocTable(rocId).joints;
                for i = rocJoints
                    s.JointState(i).RocId = rocId;
                    s.JointState(i).RocVelocity = rocV;
                end
                
                s.JointState(16)
                
            end     
            
            if strncmp(className,'Endpoint',8)
                % Handle Endpoint Classes under a special case
                
                Vx = 0.1*prSpeed;
                Vy = 0.1*prSpeed;
                Vz = 0.1*prSpeed;
                roll = 0.1*prSpeed;
                pitch = 0.1*prSpeed;
                yaw = 0.1*prSpeed;
                
                switch className
                    case 'Endpoint Out'
                        s.structState(8).State = [Vx 0 0 0 0 0 0 0];
                    case 'Endpoint In'
                        s.structState(8).State = [-Vx 0 0 0 0 0 0 0];
                    case 'Endpoint Left'
                        s.structState(8).State = [0 Vy 0 0 0 0 0 0];
                    case 'Endpoint Right'
                        s.structState(8).State = [0 -Vy 0 0 0 0 0 0];
                    case 'Endpoint Up'
                        s.structState(8).State = [0 0 Vz 0 0 0 0 0];
                    case 'Endpoint Down'
                        s.structState(8).State = [0 0 -Vz 0 0 0 0 0];
                    case 'Endpoint Roll In'
                        s.structState(8).State = [0 0 0 roll 0 0 0 0 0];
                    case 'Endpoint Roll Out'
                        s.structState(8).State = [0 0 0 -roll 0 0 0 0];
                    case 'Endpoint Pitch Up'
                        s.structState(8).State = [0 0 0 0 pitch 0 0 0];
                    case 'Endpoint Pitch Down'
                        s.structState(8).State = [0 0 0 0 -pitch 0 0 0];
                    case 'Endpoint Yaw In'
                        s.structState(8).State = [0 0 0 0 0 yaw 0 0];
                    case 'Endpoint Yaw Out'
                        s.structState(8).State = [0 0 0 0 0 -yaw 0 0];
                    otherwise
                        warning('Unmatched Endpoint Class');
                end
            end
            
            % Parse partial classname
            strMatch = 'Whole Arm Roc';
            if strncmp(className,strMatch,length(strMatch))

                % only change roc state if at beginning of motion
                if s.structState(s.RocStateId).Value < obj.RocChangeThreshold && (rocV > 0)
                    s.setRocId(rocId);
                end
                 
                s.setVelocity(s.RocStateId,rocV);
                
            end
            
        end
        function isGraspClass = generateGraspCommand(obj,className,prSpeed)
            
            if isempty(className)
                isGraspClass = false;
                return
            end
            
            % Get the decoded grasp name.  This is equivelant to the class
            % name, but if it is a 'Grasp' then a flag will be set that
            % this class can be used for 'hand close'
            isGraspClass = strfind(lower(className),'grasp');
            if isGraspClass
                % Strip off the 'Grasp' string and leave only the type
                graspName = strtrim(className(1:end-5));
                % MANUAL OVERRIDE: Setting a fixed grasp speed
                %prSpeed = 2.5;
            else
                graspName = className;
            end
            
            % Get a list of valid grasp types
            [enumGrasp, cellGrasps] = enumeration('Controls.GraspTypes');
            
            % Handle special case for grasp auto open.  In this paradigm,
            % only hand close patterns are trained.  The hand opens only
            % during no movement classes
            
            s = obj.ArmStateModel;

            % Set a switch for sending hand open / close with endpoint
            isEndpointMode = length(s.structState(8).State) >= 6;
            
            switch graspName
                case 'Hand Open'

                    if isEndpointMode
                        rocId = 1;
                        rocValue = 1;
                        s.structState(8).State = [0 0 0 0 0 0 rocId rocValue];
                    else
                        % Joint Mode
                        s.setVelocity(s.RocStateId,-prSpeed);
                    end
                    
                case cellGrasps
                    % Any valid grasp == Hand Close
                    graspId = enumGrasp( strcmp(graspName,cellGrasps) );
                    if obj.GraspValue < obj.GraspChangeThreshold
                        s.setRocId(graspId);
                    end
                    
                    if isEndpointMode
                        rocId = 1;
                        rocValue = -1;
                        s.structState(8).State = [0 0 0 0 0 0 rocId rocValue];
                    else
                        % Joint Mode
                        s.setVelocity(s.RocStateId,+prSpeed);
                    end
                
                case {'No Movement','Rest'}
                    if isEndpointMode
                        rocId = 1;
                        rocValue = 0;
                        s.structState(8).State = [0 0 0 0 0 0 rocId rocValue];
                    else
                        % Joint Mode
                        s.setVelocity(s.RocStateId,0);
                    end
                    
                    % Auto-open
                    if obj.AutoOpenSpeed > 0
                        %desiredGraspVelocity = -obj.AutoOpenSpeed;
                        s.setVelocity(s.RocStateId,-obj.AutoOpenSpeed);
                    end
                    
                otherwise
                    %s.setVelocity(s.RocStateId,0);
                    if isGraspClass
                        fprintf('[%s.m] Unmatched grasp: "%s"\n',mfilename,graspName);
                    end
            end
            
            
            % advance the state model
            obj.ArmStateModel.update();
            
            % joint angles and grasp values fields are updated here for
            % backward comparatbility
            % update the state
            state = obj.ArmStateModel.getValues();
            
            obj.JointAnglesDegrees(1:7) = state(1:7) * 180/pi;
            if isGraspClass
                obj.GraspValue = state(8);
                obj.GraspId = obj.ArmStateModel.structState(8).State;
            end
            

            %generateGraspCommandTwoState(obj,className,prSpeed);
        end
        function generateGraspCommandTwoState(obj,className,prSpeed)
            
            if isempty(className)
                return
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%
            % Process grasps
            %%%%%%%%%%%%%%%%%%%%%%%%
            
            % Implement a new grasp control paradign in which a barrier is
            % created between the grasp shaping portion (rest to
            % prehension) and then prehension to fully closed.  Nominally
            % the threshold will be 20%
            %
            %
            % Starting from rest, decode grasps in real-time allowing
            % switching from the grasp type.  The max movement will be up
            % to 20% until a rest is received.  The grasp will then be
            % 'locked' in the current grasp state.  The next grasp decoded
            % after no movement will close the hand in the locked grasp
            % conformation.  Any hand close command will result in the hand
            % closing in the locked grasp type, allowing for
            % misclassificaiton.  Hand open and close commands will then
            % only move the hand along the prescribed grasp trajectory
            % (from 20% to 100%).
            %
            % To return to the rest position, the hand must start at or
            % close to 20% (or 25%), and then a sustained hand open command
            % must be issued.
            
            
            
            % Get the decoded grasp name.  This is equivelant to the class
            % name, but if it is a 'Grasp' then a flag will be set that
            % this class can be used for 'hand close'
            isGraspClass = strfind(lower(className),'grasp');
            if isGraspClass
                % Strip off the 'Grasp' string and leave only the type
                graspName = strtrim(className(1:end-5));
            else
                graspName = className;
            end
            
            % Get a list of valid grasp types
            [enumGrasp, cellGrasps] = enumeration('Controls.GraspTypes');
            
            
            % Handle special case for grasp auto open.  In this paradigm,
            % only hand close patterns are trained.  The hand opens only
            % during no movement classes
            % TODO: Implement
            %                         % Auto-open
            %                         if obj.AutoOpenSpeed > 0
            %                             desiredGraspVelocity = -obj.AutoOpenSpeed;
            %                         end
            
            
            
            switch graspName
                case 'Hand Open'
                    desiredGraspVelocity = -prSpeed*0.5;
                case cellGrasps
                    % Any valid grasp == Hand Close
                    desiredGraspVelocity = +prSpeed*0.5;
                    
                    if obj.GraspValue < 0.2
                        % only change grasp Ids if the hand is mostly open
                        obj.GraspId = enumGrasp( strcmp(graspName,cellGrasps) );
                    end
                    
                case {'No Movement','Rest'}
                    desiredGraspVelocity = 0;
                otherwise
                    desiredGraspVelocity = 0;
                    if isGraspClass
                        fprintf('[%s.m] Unmatched grasp: "%s"\n',mfilename,graspName);
                    end
            end
            
            % override
            
            % Limit the max velocity
            graspGain = 0.05;
            obj.GraspVelocity = obj.constrain(desiredGraspVelocity*graspGain,-0.1,0.1);
            
            % Limit the grasp range
            obj.GraspValue = obj.constrain(obj.GraspValue + obj.GraspVelocity, 0.0, 1.0);
            
            
            % Traditional grasp movement
            obj.GraspVelocity = obj.constrain(desiredGraspVelocity,-0.1,0.1);
            % Limit the grasp range
            obj.GraspValue = obj.constrain(obj.GraspValue + obj.GraspVelocity, 0.0, 1.0);
            return
            
            
            return
            
            %obj.GraspValue = max(obj.GraspValue,.21);
            obj.GraspLocked = 1;
            
            %             fprintf('[%s] Grasp Locked==%d; Counter==%2d; Value==%4.1f\n',...
            %                 mfilename,obj.GraspLocked,obj.GraspChangeCounter,obj.GraspValue);
            
            if obj.GraspLocked
                % Range is 20% to 100%, no grasp changes allowed
                
                if strcmpi(graspName,{'Hand Open'})
                    obj.GraspChangeCounter = obj.GraspChangeCounter + 1;
                else
                    obj.GraspChangeCounter = 0;
                end
                
                if obj.GraspChangeCounter > obj.GraspChangeCount
                    % unlock hand
                    obj.GraspLocked = 0;
                    obj.GraspValue = 0.2;
                    obj.GraspVelocity = 0;
                    obj.GraspChangeCounter = 0;
                    return
                else
                    % keep hand locked and move within confined trajectory
                    obj.GraspLocked = 1;
                    
                    % Limit the max velocity
                    obj.GraspVelocity = obj.constrain(desiredGraspVelocity,-0.1,0.1);
                    v = obj.GraspVelocity;
                    % Limit the grasp range
                    obj.GraspValue = obj.constrain(obj.GraspValue + obj.GraspVelocity, 0.2, 1.0);
                end
                
            else
                % Grasp is unlocked, range limited to 0% to 20%, grasp
                % changes allowed
                
                switch graspName
                    case cellGrasps
                        % Increment position along grasp trajectory
                        obj.GraspId = enumGrasp( strcmp(graspName,cellGrasps) );
                        % Limit the max velocity
                        obj.GraspVelocity = obj.constrain(desiredGraspVelocity,-0.1,0.1);
                        % Limit the grasp range
                        obj.GraspValue = obj.constrain(obj.GraspValue + obj.GraspVelocity, 0.0, 0.2);
                end
                
                % Count how long a grasp close is given
                if isGraspClass
                    obj.GraspChangeCounter = obj.GraspChangeCounter + 1;
                else
                    obj.GraspChangeCounter = 0;
                end
                
                if obj.GraspChangeCounter > obj.GraspChangeCount
                    % lock hand
                    obj.GraspLocked = 1;
                    obj.GraspValue = 0.2;
                    obj.GraspVelocity = 0;
                    obj.GraspChangeCounter = 0;
                    return
                else
                    % keep hand unlocked and move within prehension trajectory
                    obj.GraspLocked = 0;
                    
                    % Limit the max velocity
                    obj.GraspVelocity = obj.constrain(desiredGraspVelocity,-0.1,0.1);
                    % Limit the grasp range
                    obj.GraspValue = obj.constrain(obj.GraspValue + obj.GraspVelocity, 0.0, 0.2);
                end
            end
            
        end
        function close(obj)
            try
                stop(obj.Timer);
                delete(obj.Timer);
                obj.ArmStateModel.saveTempState();
            end
        end
        function getRocConfig(obj)
            % Function load roc table into memory and stores in the
            % RocTable property
            %
            % Note this will first try to load info about Roc table xml
            % file from the user config .xml file
            
            % create local ROC tables (even though roc tables in vulcan x
            % can also be specified)
            xmlFileName = UserConfig.getUserConfigVar('rocTable','NONE');
            if strcmp(xmlFileName,'NONE')
                % No file, create from code
                obj.RocTable = MPL.RocTable.createRocTables;
                obj.RocTableXmlFilename = 'NONE';
            else
                obj.RocTable = MPL.RocTable.readRocTable(xmlFileName);
                obj.RocTableXmlFilename = xmlFileName;
            end
        end
        function mplAngles = getArmAngles(obj)
            % Get the current arm angles from the state controller and roc
            % interpolation
            %
            
            m = obj.ArmStateModel;
            rocValue = m.structState(m.RocStateId).Value;
            rocId = m.structState(m.RocStateId).State;
            if isa(rocId,'Controls.GraspTypes')
                % convert char grasp class name (e.g. 'Spherical') to numerical mpl
                % grasp value (e.g. 7)
                rocId = MPL.GraspConverter.graspLookup(rocId);
            end
            
            mplAngles = zeros(1,27);
            mplAngles(1:7) = [m.structState(1:7).Value];
            
            % Generate vulcanX message.  If local roc table exists, use it
            
            % check bounds
            rocValue = max(min(rocValue,1),0);
            % lookup the Roc id and find the right table
            iEntry = (rocId == [obj.RocTable(:).id]);
            if sum(iEntry) < 1
                error('Roc Id %d not found',rocId);
            elseif sum(iEntry) > 1
                warning('More than 1 Roc Tables share the id # %d',rocId);
                roc = obj.RocTable(find(iEntry,1,'first'));
            else
                roc = obj.RocTable(iEntry);
            end
            
            % perform local interpolation
            mplAngles(roc.joints) = interp1(roc.waypoint,roc.angles,rocValue);
        end
    end
end
