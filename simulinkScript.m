%% Ensure the right model is loaded (no shadowing) and open it
bdclose all

addpath('/Users/axel/Documents/HandOver/Matlab/models','-begin');
rehash path

open_system('/Users/axel/Documents/HandOver/Matlab/models/StepA_model.slx');
mdl = 'StepA_model';

% sanity: verify file path of the loaded model is yours
assert(contains(get_param(mdl,'FileName'), '/models/'), ...
    'Shadowed model still active on path');



%% Setting up data for Simulink simulation

allTrain = cat(3, Xtr{:});               % [F x W x N]
mu = mean(allTrain, [2 3], 'omitnan');   % [F x 1 x 1]
sg = std(allTrain, 0, [2 3], 'omitnan'); sg(sg==0) = 1;

normCell = @(C) cellfun(@(m) (m - mu)./sg, C, 'uni', 0);
Xtr = normCell(Xtr);  Xva = normCell(Xva);  Xte = normCell(Xte);

%%
steps = length(Yte);

Ts = 1;
timeVector = (0:Ts:(steps-1))';
Xin = timeseries(cat(3, Xte{:}));
trueSoH = timeseries(Yte');
%% 

% Create input data for the model.
% NOTE: this is how set things up when working with Inport blocks. There
% are alternative ways with other blocks.
ds = Simulink.SimulationData.Dataset;
ds = setElement(ds,1,Xin);
ds = setElement(ds,2,trueSoH);
%%
% Note there is some sort of scaling or offset error in this current
% version and there should be and easy fix. 
%Simulate the model and export the simulation output to the workspace
mdlPath = "/Users/axel/Documents/HandOver/Matlab/models/StepA_model.slx";
in = Simulink.SimulationInput(mdlPath);
in = in.setModelParameter("StopTime",'steps');
in = in.setModelParameter("FixedStep",'Ts');
in = in.setExternalInput(ds);
simOut = sim(in);