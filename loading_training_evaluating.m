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
numFeat = numel(feat);
layers = [
    sequenceInputLayer(numFeat)
    lstmLayer(64,'OutputMode','last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

opts = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xva, Yva}, ...
    'ValidationFrequency', 50, ...
    'Verbose', false);
%%
net = trainNetwork(Xtr, Ytr, layers, opts);

%% Step A  Evaluate (with stable MAPE)
pred = predict(net, Xte, 'MiniBatchSize', 64);
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
