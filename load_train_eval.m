%% LSTM SOH — Step A (baseline) + Step B (AR with GT-TF train, roll-forward test)

% 0) Load + filter
S = load(fullfile(pwd,'processed','XJTU_all_cycles.mat'));
T = S.allCycles;

% Exclude Sim_satellite cells for now
T = T(~startsWith(string(T.File), "Sim_satellite"), :);
T = sortrows(T, ["File","Cycle"]);

% 1) Config  (NO capacity as input → avoid leakage)
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

% 3) Build sequences per split (assumes makeSequences returns X{N}[F×W], Y[N×1], info struct with File,CycleTarget)
[Xtr,Ytr,infoTr,Ytr_last] = makeSequences(T, W, feat, 'IncludeFiles', fTrain);
[Xva,Yva,infoVa,Yva_last] = makeSequences(T, W, feat, 'IncludeFiles', fVal);
[Xte,Yte,infoTe,Yte_last] = makeSequences(T, W, feat, 'IncludeFiles', fTest);

% 4) Normalize using TRAIN stats only
allTrain = cat(3, Xtr{:});               % [F x W x N]
mu = mean(allTrain, [2 3], 'omitnan');   % [F x 1 x 1]
sg = std(allTrain, 0, [2 3], 'omitnan'); sg(sg==0) = 1;

normCell = @(C) cellfun(@(m) (m - mu)./sg, C, 'uni', 0);
Xtr = normCell(Xtr);  Xva = normCell(Xva);  Xte = normCell(Xte);

% 5) Model (simple LSTM) — STEP A
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
deepNetworkDesigner(layers)
%%
net = trainNetwork(Xtr, Ytr, layers, opts);

%% Step A — Evaluate (with stable MAPE)
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
testFiles   = testFiles(1:min(3,numel(testFiles)));
figure('Name','STEP A: Pred vs True (first 3 test files)','NumberTitle','off');
tiledlayout(numel(testFiles),1,'Padding','compact','TileSpacing','compact')
for k = 1+1:1+numel(testFiles)
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

%% STEP B — AR LSTM with GT-TF train, roll-forward test (no leakage)
% Build prev_true_SOH for TRAIN/VAL strictly sized to Xtr/Xva, with debug

% Source-of-truth sizes (sequence counts)
Ntr = numel(Xtr);
Nva = numel(Xva);
Nte = numel(Xte);
fprintf('\n[DEBUG] Sizes  | Ntr=%d, Nva=%d, Nte=%d | numel(infoTr)=%d, numel(infoVa)=%d, numel(infoTe)=%d\n', ...
    Ntr, Nva, Nte, numel(infoTr), numel(infoVa), numel(infoTe));

% Pull metadata arrays
filesTr  = getFieldAsStringCol(infoTr,'File');
cyclesTr = getFieldAsDoubleCol(infoTr,'CycleTarget');
filesVa  = getFieldAsStringCol(infoVa,'File');
cyclesVa = getFieldAsDoubleCol(infoVa,'CycleTarget');
filesTe  = getFieldAsStringCol(infoTe,'File');
cyclesTe = getFieldAsDoubleCol(infoTe,'CycleTarget');

fprintf('[DEBUG] info   | len(filesTr)=%d, len(cyclesTr)=%d | len(filesVa)=%d, len(cyclesVa)=%d | len(filesTe)=%d, len(cyclesTe)=%d\n', ...
    numel(filesTr), numel(cyclesTr), numel(filesVa), numel(cyclesVa), numel(filesTe), numel(cyclesTe));

if numel(filesTr)~=Ntr || numel(cyclesTr)~=Ntr
    warning('TRAIN: infoTr length (%d/%d) != Xtr length (%d). Will truncate/pad to Xtr length.', ...
        numel(filesTr), numel(cyclesTr), Ntr);
end
if numel(filesVa)~=Nva || numel(cyclesVa)~=Nva
    warning('VAL  : infoVa length (%d/%d) != Xva length (%d). Will truncate/pad to Xva length.', ...
        numel(filesVa), numel(cyclesVa), Nva);
end
if numel(filesTe)~=Nte || numel(cyclesTe)~=Nte
    warning('TEST : infoTe length (%d/%d) != Xte length (%d). Will clamp loop to Xte length.', ...
        numel(filesTe), numel(cyclesTe), Nte);
end

% Build key->sequenceIndex maps ALIGNED to sequence arrays (indices 1..N)
trainMap = indexMapSeq(infoTr, Ntr);   % keys map to 1..Ntr
valMap   = indexMapSeq(infoVa, Nva);   % keys map to 1..Nva
fprintf('[DEBUG] MAPS  | trainMap.Count=%d, valMap.Count=%d\n', trainMap.Count, valMap.Count);


% ---- TRAIN prev_true_SOH (GT teacher forcing inputs) ----
prev_true_tr = ones(Ntr,1);  % prior
badTr = 0; missPrevTr = 0;
for i = 1:Ntr
    % safe pick of File/Cycle even if info arrays are longer/shorter
    if i <= numel(filesTr),  f = filesTr(i);  else, f = "";  end
    if i <= numel(cyclesTr), c = cyclesTr(i); else, c = NaN; end

    if ismissing(f) || f=="" || isnan(c)
        badTr = badTr + 1;
        prev_true_tr(i) = 1.0;
        continue;
    end

    kPrev = keyChar(f, c-1);    % char key "File#(c-1)"
    if ~isempty(kPrev) && isKey(trainMap, kPrev)
        j = trainMap(kPrev);    % 1..Ntr by construction
        % guard index just in case
        if j >= 1 && j <= numel(Ytr)
            prev_true_tr(i) = Ytr(j);
        else
            prev_true_tr(i) = 1.0;
            missPrevTr = missPrevTr + 1;
        end
    else
        prev_true_tr(i) = 1.0;
        missPrevTr = missPrevTr + 1;
    end
end
fprintf('[DEBUG] TRAIN  | bad rows=%d (empty File/NaN cycle), missing prev matches=%d\n', badTr, missPrevTr);

% ---- VAL prev_true_SOH (same logic) ----
prev_true_va = ones(Nva,1);
badVa = 0; missPrevVa = 0;
for i = 1:Nva
    if i <= numel(filesVa),  f = filesVa(i);  else, f = "";  end
    if i <= numel(cyclesVa), c = cyclesVa(i); else, c = NaN; end

    if ismissing(f) || f=="" || isnan(c)
        badVa = badVa + 1;
        prev_true_va(i) = 1.0;
        continue;
    end

    kPrev = keyChar(f, c-1);
    if ~isempty(kPrev) && isKey(valMap, kPrev)
        j = valMap(kPrev);      % 1..Nva
        if j >= 1 && j <= numel(Yva)
            prev_true_va(i) = Yva(j);
        else
            prev_true_va(i) = 1.0;
            missPrevVa = missPrevVa + 1;
        end
    else
        prev_true_va(i) = 1.0;
        missPrevVa = missPrevVa + 1;
    end
end
fprintf('[DEBUG] VAL    | bad rows=%d (empty File/NaN cycle), missing prev matches=%d\n', badVa, missPrevVa);

% Normalize AR channel using TRAIN stats
mu_prev = mean(prev_true_tr, 'omitnan');
sg_prev = std(prev_true_tr, 0, 'omitnan'); if sg_prev==0, sg_prev = 1; end
normPrev = @(v) (v - mu_prev) ./ sg_prev;
fprintf('[DEBUG] AR norm| mu_prev=%.6f, sg_prev=%.6f\n', mu_prev, sg_prev);

% Augment sequences with normalized prev_true channel (repeat across W)
Xtr_TF = cellfun(@(X,pt) [X; repmat(normPrev(pt),1,size(X,2))], Xtr, num2cell(prev_true_tr), 'uni', 0);
Xva_TF = cellfun(@(X,pt) [X; repmat(normPrev(pt),1,size(X,2))], Xva, num2cell(prev_true_va), 'uni', 0);

% Train AR (GT-TF) model
numFeatTF = numFeat + 1;  % + prev_true_SOH channel
layersTF = [
    sequenceInputLayer(numFeatTF)
    lstmLayer(64,'OutputMode','last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

optsTF = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xva_TF, Yva}, ...
    'ValidationFrequency', 50, ...
    'Verbose', false);

net_TF = trainNetwork(Xtr_TF, Ytr, layersTF, optsTF);

% ---- Roll-forward TEST (no GT at inference) ----
% Sort by (File, Cycle). Clamp to available Xte length.
% Order test by (File, Cycle). Name the table columns explicitly.
Mte = min([numel(filesTe), numel(cyclesTe), Nte]);  % also clamp to Nte
if Mte == 0
    error('TEST: no metadata available to order test sequences.');
end

Tord = table(filesTe(1:Mte), cyclesTe(1:Mte), ...
             'VariableNames', {'filesTe','cyclesTe'});

% Debug a few rows to see what we're sorting
fprintf('[DEBUG] TEST ordering | Mte=%d | first rows preview:\n', Mte);
disp(Tord(1:min(5,height(Tord)), :));

[~, ordIdx] = sortrows(Tord, {'filesTe','cyclesTe'});
% ordIdx is already 1..Mte <= Nte, so no extra clamping needed

pred_TF  = zeros(Nte,1);
lastPred = containers.Map('KeyType','char','ValueType','double');

badTe = 0;
for k = 1:numel(ordIdx)
    ii = ordIdx(k);         % guaranteed 1..Nte
    if ii <= numel(filesTe),  f = filesTe(ii);  else, f = "";  end
    if ii <= numel(cyclesTe), c = cyclesTe(ii); else, c = NaN; end

    if ismissing(f) || f==""
        badTe = badTe + 1;
    end

    % previous prediction (or prior)
    kPrev = keyChar(f, c-1);    % char
    pp = 1.0;
    if ~isempty(kPrev) && isKey(lastPred, kPrev)
        pp = lastPred(kPrev);
    end

    % predict
    Xi = Xte{ii};
    X_aug = [Xi; repmat(normPrev(pp), 1, size(Xi,2))];
    yhat  = predict(net_TF, {X_aug}, 'MiniBatchSize', 1);
    pred_TF(ii) = yhat;

    % carry forward to c
    lastPred(keyChar(f, c)) = yhat;
end
fprintf('[DEBUG] TEST   | rows with empty File encountered: %d\n', badTe);

mae_TF  = mean(abs(pred_TF - Yte));
rmse_TF = sqrt(mean((pred_TF - Yte).^2));
epsMape = 1e-8;
mape_TF = mean(abs((pred_TF - Yte) ./ max(abs(Yte), epsMape)));
fprintf('STEP B  AR-LSTM (GT-TF train, roll-forward test)  MAE=%.4f, RMSE=%.4f, MAPE=%.4f\n', ...
    mae_TF, rmse_TF, mape_TF);

% Save STEP B model + AR scaling
save(fullfile('models','DLmodel_LSTM_AR_TF.mat'), ...
     'net_TF','options','preproc','splits','mu_prev','sg_prev','-v7.3');
fprintf('Saved STEP B to models/DLmodel_LSTM_AR_TF.mat\n');
%% inspecting the model
deepNetworkDesigner(net_TF)


%% STEP B — Pred vs True plots (overlay Step A vs Step B) for first 3 test files

filesPerSeqB = getFieldAsStringCol(infoTe, 'File');
testFilesB   = unique(filesPerSeqB,'stable');
testFilesB   = testFilesB(1+1:1+min(3,numel(testFilesB)));

figure('Name','STEP B: Pred vs True (first 3 test files)','NumberTitle','off');
tiledlayout(numel(testFilesB),1,'Padding','compact','TileSpacing','compact')

for k = 1:numel(testFilesB)
    f = testFilesB(k);

    % indices belonging to this file
    idx = find(filesPerSeqB == f);

    % order by target cycle for a smooth curve
    cycLocal = getFieldAsDoubleCol(infoTe(idx), 'CycleTarget');
    [~, ord] = sort(cycLocal);
    idx = idx(ord);

    % plot
    nexttile;
    plot(pred(idx),'o-','DisplayName','Step A'); hold on;      % baseline LSTM
    plot(pred_TF(idx),'s-','DisplayName','Step B (AR)');       % AR with GT-TF
    plot(Yte(idx),'x-','DisplayName','True');                  % ground truth
    ylabel('SoH'); title(string(f));
    legend('show','Location','best'); grid on;
end
xlabel('Sequence # (ordered by target cycle)');

%% STEP B — Error distribution (optional)
figure('Name','Error histograms: Step A vs Step B','NumberTitle','off');
edges = linspace(-0.15, 0.15, 60);
histogram(pred - Yte, edges, 'DisplayName','Step A'); hold on;
histogram(pred_TF - Yte, edges, 'DisplayName','Step B (AR)');
xlabel('Prediction error (ŷ - y)'); ylabel('Count'); grid on;
legend('show','Location','best');



%% Correlation sanity (unchanged)
vars = ["V_mean","V_std", "t_sec"];
Xtab = T{:, vars};
yall = T.SoH;
[Rp, pP] = corr(Xtab, yall, 'Rows','complete', 'Type','Pearson');
[Rs, pS] = corr(Xtab, yall, 'Rows','complete', 'Type','Spearman');
disp(table(vars', Rp, pP, Rs, pS, 'VariableNames', ...
    {'Feature','PearsonR','PearsonP','SpearmanR','SpearmanP'}));

%% SoH checks/plots (unchanged)
soh = T.SoH; isBad = isnan(soh) | isinf(soh);
fprintf('SoH NaN/Inf count: %d / %d (%.2f%%)\n', sum(isBad), numel(soh), 100*mean(isBad));
minS = min(soh, [], 'omitnan'); maxS = max(soh, [], 'omitnan');
xlo  = max(0, minS - 0.05); xhi = min(1.2, maxS + 0.05);
lb = 0; ub = 1.1;
outOfRange = (soh < lb | soh > ub) & ~isBad;
fprintf('Out-of-range SoH (<%.2f or >%.2f): %d (%.2f%%)\n', lb, ub, sum(outOfRange), 100*mean(outOfRange));
fprintf('SoH stats: min=%.4f, 1%%=%.4f, 25%%=%.4f, median=%.4f, 75%%=%.4f, 99%%=%.4f, max=%.4f\n', ...
    quantile(soh, [0 0.01 0.25 0.5 0.75 0.99 1], 'Method','approx'));

figure('Name','SoH Distribution (Overall)','NumberTitle','off');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; histogram(soh, 'BinMethod','fd','FaceAlpha',0.8); xlim([xlo xhi]); grid on; xlabel('SoH'); ylabel('Count'); title('Histogram of SoH');
nexttile; hold on; [xPDF,fPDF]=ksdensity(soh(~isBad)); plot(xPDF,fPDF,'LineWidth',1.5); xline(1.0,'--'); xlim([xlo xhi]); grid on; xlabel('SoH'); ylabel('Density'); title('Kernel Density');
nexttile; qqplot(soh(~isBad)); title('QQ Plot of SoH');
nexttile; boxplot(soh(~isBad),'Notch','on','Symbol','r.'); ylim([xlo xhi]); grid on; ylabel('SoH'); title('Boxplot (Overall)');

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
