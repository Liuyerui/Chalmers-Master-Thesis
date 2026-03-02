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
net = trainnet(Xtr, Ytr, net, @adaptiveMSE, opts);

%% Step A  Evaluate (with stable MAPE)
pred = customPredict(net, Xte, 64);

% 转换格式：minibatchpredict 可能返回 dlarray，需要转回普通数组
if isdlarray(pred)
    pred = extractdata(pred);
end

% 确保维度匹配 (防止是 [1, N] 而不是 [N, 1])
if size(pred, 1) < size(pred, 2)
    pred = pred';
end

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
function loss = adaptiveMSE(Y, T)
    % Y: 网络输出，可能是 [1×B] 格式 CT
    % T: 目标值 [B×1] 格式 BC
    
    % 步骤1: 提取数据（暂时切断自动微分）
    Ydata = extractdata(Y);
    Tdata = extractdata(T);
    
    % 步骤2: 确保 Y 是列向量
    Ydata = Ydata(:);
    Tdata = Tdata(:);
    
    % 步骤3: 计算 MSE（数值计算）
    mse_value = mean((Ydata - Tdata).^2);
    
    % 步骤4: 重新包装为 dlarray，重新连接计算图
    % 注意：这里使用 dlarray(mse_value) 会创建一个新的计算图起点
    % 但由于 mse_value 是从 Ydata 和 Tdata 计算得来的数值，
    % 而 Ydata 是从 Y 提取的，梯度信息已经丢失
    
    % 正确方法：使用原始 Y 和 T 重新计算 MSE，但确保格式正确
    % 重新创建格式正确的 dlarray 用于梯度计算
    if strcmp(dims(Y), 'CT') && size(Y, 1) == 1
        % 手动创建正确的格式
        B = size(Y, 2);  % 批量大小
        Y_correct = dlarray(zeros(B, 1), 'BC');
        % 将 Y 的值赋给 Y_correct
        Y_correct = dlarray(extractdata(Y)', 'BC');
    else
        Y_correct = Y;
    end
    
    % 现在格式正确，重新计算 MSE 以保持梯度
    loss = mse(Y_correct, T);
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