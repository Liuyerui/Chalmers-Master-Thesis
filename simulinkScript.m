%% 修正维度后的 Simulink 运行脚本

%% 1. 环境准备
bdclose all;
modelPath = 'D:/postgraduate/master_thesis/Chalmers-Master-Thesis/models/StepA_model.slx';
open_system(modelPath);
mdl = 'StepA_model';

%% 2. 数据归一化 (保持你原始的逻辑)
allTrain = cat(3, Xtr{:});               
mu = mean(allTrain, [2 3], 'omitnan');   
sg = std(allTrain, 0, [2 3], 'omitnan'); 
sg(sg==0) = 1;

normCell = @(C) cellfun(@(m) (m - mu)./sg, C, 'uni', 0);
Xte_norm = normCell(Xte); 

%% 3. 构造数据 (核心修正区)
steps = numel(Xte_norm);
Ts = 1;
timeVector = (0:Ts:(steps-1))';

% --- 修正 Xin 的构造 ---
% 之前你的 cell2mat(X4cell.') 可能会导致维度变成 [ (steps*4) x 1 ]
% 我们需要确保它是 [ steps x 4 ]
X4cell = cellfun(@(m) m(:, end), Xte_norm, 'uni', 0); % 提取每个 cell 的 [4x1]
Xmat = squeeze(cell2mat(reshape(X4cell, 1, 1, [])));  % 得到 [4 x steps]
Xmat = Xmat';                                        % 转置为 [steps x 4]

% --- 修正 trueSoH 的维度 ---
% 报错提示模型需要维度 [4]，所以我们要把 1 维的 SOH 复制成 4 列
Ycol = Yte(:);            % [steps x 1]
Ymat = repmat(Ycol, 1, 4); % [steps x 4]  <-- 匹配模型要求的维度 [4]

% 构造 timeseries
% 注意：timeseries 在处理多维输入时，第一维必须是时间点
Xin     = timeseries(Xmat, timeVector);     
trueSoH = timeseries(Ymat, timeVector);     

% 构造 Dataset
ds = Simulink.SimulationData.Dataset;
ds = ds.addElement(Xin, 'sequenceinput');   
ds = ds.addElement(trueSoH, 'trueSOH');     

%% 4. 运行仿真
in = Simulink.SimulationInput(mdl);
in = in.setModelParameter('StopTime', num2str((steps-1)*Ts));
in = in.setModelParameter('FixedStep', num2str(Ts));
in = in.setExternalInput(ds);

simOut = sim(in);

%% 5. 关于"预测值偏低"的后续处理
% 如果运行完后黄线依然在 0.6 附近（而紫线在 1.0 附近）：
% 请检查你的神经网络训练时，是否对 Y 也做了归一化？
% 如果训练时 Y_norm = (Y - mean(Y)) / std(Y)
% 那么你需要对预测结果进行反处理：
% 
% predData = simOut.yout.getElement('out1').Values.Data;
% mu_y = 0.85; % 举例：你训练集中 SOH 的均值
% sg_y = 0.1;  % 举例：你训练集中 SOH 的标准差
% adjusted_pred = predData * sg_y + mu_y;