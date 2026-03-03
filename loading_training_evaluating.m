%% LSTM SOH  Step A (baseline)
% 0) Load + filter
S = load(fullfile(pwd,'processed','XJTU_all_cycles.mat'));
T = S.allCycles;

% Exclude Sim_satellite cells for now
T = T(~startsWith(string(T.File), "Sim_satellite"), :);
T = sortrows(T, ["File","Cycle"]);

% 1) Config  (NO capacity as input  avoid leakage)
feat = ["V_mean","V_std","t_sec","Cycle"];  % add cycle index
W = 5;
rng(42);

% 2) Split by File (no leakage)
files = unique(T.File,'stable');
idx   = randperm(numel(files));
n     = numel(files);
fTrain = files(idx(1:round(0.7*n)));
fVal   = files(idx(round(0.7*n)+1:round(0.85*n)));
fTest  = files(idx(round(0.85*n)+1:end));

% 3) Build sequences per split (assumes makeSequences returns X{N}[F◊W], Y[N◊1], info struct with File,CycleTarget)
[Xtr,Ytr,infoTr,Ytr_last] = makeSequences(T, W, feat, 'IncludeFiles', fTrain);
[Xva,Yva,infoVa,Yva_last] = makeSequences(T, W, feat, 'IncludeFiles', fVal);
[Xte,Yte,infoTe,Yte_last] = makeSequences(T, W, feat, 'IncludeFiles', fTest);

% 4) Normalize using TRAIN stats only
allTrain = cat(3, Xtr{:});               % [F x W x N]
mu = mean(allTrain, [2 3], 'omitnan');   % [F x 1 x 1]
sg = std(allTrain, 0, [2 3], 'omitnan'); sg(sg==0) = 1;

normCell = @(C) cellfun(@(m) (m - mu)./sg, C, 'uni', 0);
Xtr = normCell(Xtr);  Xva = normCell(Xva);  Xte = normCell(Xte);

% 5) Model (simple LSTM)  STEP A
% numFeat = numel(feat);
% layers = [
%     sequenceInputLayer(numFeat)
%     lstmLayer(64,'OutputMode','last')
%     fullyConnectedLayer(32)
%     reluLayer
%     fullyConnectedLayer(1)
%     regressionLayer];
% 
% opts = trainingOptions('adam', ...
%     'MaxEpochs', 40, ...
%     'MiniBatchSize', 64, ...
%     'InitialLearnRate', 1e-3, ...
%     'GradientThreshold', 1, ...
%     'Shuffle', 'every-epoch', ...
%     'ValidationData', {Xva, Yva}, ...
%     'ValidationFrequency', 50, ...
%     'Verbose', false);

% 导入 PyTorch 模型代替手动定义 layers
% -----------------------------------------------------------

modelfile = "E:\Master thesis\lstm_model.pt";

% 1. 导入模型 (得到 dlnetwork 对象)
net = importNetworkFromPyTorch(modelfile);

% 2. 定义输入层
inputSize = numel(feat); 

inputLayer = sequenceInputLayer(inputSize, ...
    'Name', 'seq_input', ...
    'Normalization', 'none', ... 
    'MinLength', 1);

% 3. 添加输入层并初始化
net = addInputLayer(net, inputLayer, Initialize=true);

% 4. 转换为 LayerGraph 以适配 trainNetwork
lgraph = layerGraph(net);

% 检查并添加回归层 (如果模型末尾不是 regressionLayer)
if ~isa(lgraph.Layers(end), 'nnet.layer.RegressionOutputLayer')
    % 创建回归层
    regLayer = regressionLayer('Name', 'out_regression');
    
    % 获取当前网络最后一层的名字
    lastLayerName = lgraph.Layers(end).Name;
    
    % 将回归层加入图
    lgraph = addLayers(lgraph, regLayer);
    
    % 连接： 最后一层 -> 回归层
    lgraph = connectLayers(lgraph, lastLayerName, 'out_regression');
end

% 训练选项 (保持你原有的配置不变)
opts = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xva, Yva}, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'TargetDataFormats', "BC", ...
    'InputDataFormats', "CTB");

%%
% net = trainNetwork(Xtr, Ytr, layers, opts);

% 5. 开始训练
net = trainnet(Xtr, Ytr, net, @robustMSE, opts);

%% Step A  Evaluate (with stable MAPE)
% 1. 手动将测试集转换为 dlarray (CTB 格式)
% Xte 是 Cell Array，我们需要把它堆叠成一个 3D 数组 [Channel, Time, Batch]
try
    % 尝试直接堆叠 (假设所有序列长度一致 W=5)
    Xte_mat = cat(3, Xte{:}); 
    dlXte = dlarray(Xte_mat, 'CTB');
catch
    error('Xte 中的序列长度不一致，无法直接堆叠进行批量预测。');
end

% 2. 直接使用 predict (避开 minibatchpredict 的标签检查)
% predict 会根据输入的一次性算出结果。由于测试集通常不大，内存应该够用。
% 如果测试集非常大导致内存溢出，需要手写 for 循环分批 predict。
dlYPred = predict(net, dlXte);

% 3. 极其暴力的后处理 (忽略所有维度标签)
% 无论输出是 "CT"、"CB" 还是 "BC"，我们只关心里面的数值
YPred_raw = extractdata(stripdims(dlYPred));

% 4. 强制展平为列向量
pred = double(YPred_raw(:)); 
Yte_vec = double(Yte(:));

% 5. 维度完整性检查 (防止顺序乱了)
% 如果 pred 的元素数量和 Yte 不一致，说明模型输出维度不仅是标签错了，形状也错了
if numel(pred) ~= numel(Yte_vec)
    warning('预测结果数量 (%d) 与 测试集标签数量 (%d) 不匹配！尝试转置...', numel(pred), numel(Yte_vec));
    % 这种情况下通常不会发生，除非模型输出是序列对序列 (Seq2Seq) 而不是序列对一
    % 如果发生了，通常取最后一个时间步
    if size(YPred_raw, 2) == numel(Yte_vec)
         pred = double(YPred_raw');
    elseif size(YPred_raw, 1) == numel(Yte_vec)
         pred = double(YPred_raw);
    else
         error('无法匹配预测结果与标签的维度。模型输出形状: %s', mat2str(size(YPred_raw)));
    end
end

% 6. 计算指标
mae  = mean(abs(pred - Yte));
rmse = sqrt(mean((pred - Yte).^2));
epsMape = 1e-8;
mape_A  = mean(abs((pred - Yte) ./ max(abs(Yte), epsMape)));
fprintf('STEP A  LSTM (features+cycle)  MAE=%.4f, RMSE=%.4f, MAPE=%.4f\n', mae, rmse, mape_A);

% Baseline: last-value
mae_last  = mean(abs(Yte_last - Yte));
rmse_last = sqrt(mean((Yte_last - Yte).^2));
mape_last = mean(abs((Yte_last - Yte) ./ max(abs(Yte), epsMape)));
fprintf('Baseline Last-Value           MAE=%.4f, RMSE=%.4f, MAPE=%.4f\n', mae_last, rmse_last, mape_last);

%% (Optional) quick per-file plot (first 3 test files)
filesPerSeq = getFieldAsStringCol(infoTe, 'File');
testFiles   = unique(filesPerSeq,'stable');
testFiles   = testFiles(2:1+min(3,numel(testFiles)));
figure('Name','STEP A: Pred vs True (first 3 test files)','NumberTitle','off');
tiledlayout(numel(testFiles),1,'Padding','compact','TileSpacing','compact')
for k = 1:numel(testFiles)
    f = testFiles(k);
    idx = find(filesPerSeq == f);
    [~, ord] = sort([infoTe(idx).CycleTarget]); idx = idx(ord);
    nexttile; plot(pred(idx),'o-'); hold on; plot(Yte(idx),'x-');
    ylabel('SoH'); title(string(f)); legend('Pred','True','Location','best'); grid on;
end
xlabel('Sequence # (ordered by target cycle)');

%% Save the STEP A model + normalization
if ~exist('models','dir'), mkdir models; end
lstm_Network_basic = net;
options = opts;
preproc.mu   = mu;
preproc.sg   = sg;
preproc.feat = feat;
preproc.W    = W;
splits.train = fTrain; splits.val = fVal; splits.test = fTest;

save(fullfile('models','DLmodel_LSTM_basic.mat'), ...
     'lstm_Network_basic','options','preproc','splits','-v7.3');

fprintf('Saved STEP A to models/DLmodel_LSTM_basic.mat\n');





%% ---------- Local helpers ----------
function M = indexMapSeq(info, Nkeep)
    % Map keys to SEQUENCE indices 1..Nkeep (not raw info indices)
    files  = getFieldAsStringCol(info, 'File');
    cycles = getFieldAsDoubleCol(info, 'CycleTarget');

    N = min([Nkeep, numel(files), numel(cycles)]);
    K = cell(0,1); V = [];

    for i = 1:N
        f = files(i); c = cycles(i);
        if ~(ismissing(f) || f=="" || isnan(c))
            kc = keyChar(f, c);           % char key "File#Cycle"
            if ~isempty(kc)
                K{end+1,1} = kc;          %#ok<AGROW>
                V(end+1,1) = i;           %#ok<AGROW>  % sequence index
            end
        end
    end

    if isempty(K)
        M = containers.Map('KeyType','char','ValueType','double');
    else
        M = containers.Map(K, num2cell(V));
    end
end

function v = getFieldRaw(info, field)
    if isstruct(info)
        v = {info.(field)}.';         % cell column
    elseif istable(info)
        v = info.(field);             % table var (any type)
    else
        error('indexMap: unsupported info type');
    end
end

function s = getFieldAsStringCol(info, field)
    raw = getFieldRaw(info, field);
    try
        s = string(raw);
    catch
        n = numel(raw);
        s = strings(n,1);
        for i = 1:n
            try
                si = string(raw{i});
            catch
                si = string(raw(i));
            end
            if isempty(si)
                s(i) = "";
            else
                s(i) = si(1);
            end
        end
    end
    s = s(:);
    if ~isempty(s) && size(s,2) > 1, s = s(:,1); end
    s(ismissing(s)) = "";
end

function d = getFieldAsDoubleCol(info, field)
    raw = getFieldRaw(info, field);

    if isnumeric(raw)
        d = double(raw(:)); return;
    end

    if iscell(raw)
        n = numel(raw);
        d = nan(n,1);
        for i = 1:n
            xi = raw{i};
            if isempty(xi)
                d(i) = NaN;
            elseif isnumeric(xi) && isscalar(xi)
                d(i) = double(xi);
            elseif isstring(xi) || ischar(xi) || iscategorical(xi)
                s = string(xi);
                d(i) = str2double(s(1));
            else
                try
                    d(i) = double(xi);
                catch
                    d(i) = str2double(string(xi));
                end
            end
        end
        return;
    end

    if isstring(raw) || ischar(raw) || iscategorical(raw)
        d = str2double(string(raw(:))); return;
    end

    d = str2double(string(raw(:)));
end

function k = keyChar(f, c)
    % char key "File#Cycle" or '' if invalid
    if ismissing(f) || strlength(f)==0 || isnan(c)
        k = '';
    else
        k = [char(f) '#' num2str(c)];
    end
end

%%
% 定义自适应损失函数
function loss = robustMSE(Y, T)
    % Y: 网络预测输出 (dlarray)
    % T: 真实标签 (dlarray)
    
    % 1. 去除维度标签 (Strip Dimension Labels)
    % stripdims 不会切断梯度，只是移除 'C', 'B', 'T' 这些元数据
    Y_raw = stripdims(Y);
    T_raw = stripdims(T);
    
    % 2. 维度对齐
    % PyTorch 导入的模型通常输出 [1, Batch] 或 [Batch, 1]
    % 我们将它们统一 reshape 成列向量 [N, 1] 以便相减
    Y_vec = reshape(Y_raw, [], 1);
    T_vec = reshape(T_raw, [], 1);
    
    % 3. 计算 MSE
    % 这里的运算都是在 dlarray 上进行的，梯度链条完好无损
    loss = mean((Y_vec - T_vec).^2);
end
% 定义自定义预测函数
function Ypred = customPredict(net, X, mbSize)
    numSamples = length(X);
    numBatches = ceil(numSamples / mbSize);
    Ypred = zeros(numSamples, 1);
    
    for i = 1:numBatches
        batchIdx = (i-1)*mbSize + 1 : min(i*mbSize, numSamples);
        
        % 准备批量数据
        Xbatch_cell = X(batchIdx);
        
        % 检查每个样本的格式，确保是 [4×W]
        for j = 1:length(Xbatch_cell)
            if size(Xbatch_cell{j}, 1) == 5  % 如果特征数不对
                Xbatch_cell{j} = Xbatch_cell{j}';
            end
        end
        
        % 堆叠成 [4×W×B]
        try
            Xbatch = cat(3, Xbatch_cell{:});
        catch ME
            % 如果维度不一致，可能需要调整
            fprintf('批次 %d 维度不一致，尝试调整...\n', i);
            % 统一为 [4×W] 格式
            for j = 1:length(Xbatch_cell)
                if size(Xbatch_cell{j}, 1) ~= 4
                    Xbatch_cell{j} = Xbatch_cell{j}';
                end
            end
            Xbatch = cat(3, Xbatch_cell{:});
        end
        
        % 转换为 dlarray
        XbatchDL = dlarray(Xbatch, 'CTB');
        
        % 预测
        Ybatch = predict(net, XbatchDL);
        
        % 处理输出格式 - 使用你调试成功的方法
        Ybatch_data = extractdata(Ybatch);
        
        % 根据维度处理
        if ismatrix(Ybatch_data) && size(Ybatch_data, 1) == 1
            % [1×B] -> [B×1]
            Ybatch_data = Ybatch_data';
        elseif ndims(Ybatch_data) == 3
            % [1×1×B] -> [B×1]
            Ybatch_data = reshape(Ybatch_data, [], 1);
        end
        
        Ypred(batchIdx) = Ybatch_data;
    end
end