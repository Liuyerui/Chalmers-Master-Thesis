%% DataLoading.m — XJTU -> cycles -> (ready for LSTM)
% Run from project root: /Users/axel/Documents/MATLAB/batterySOHEstimation

%clear; clc;

rootDir = fullfile(pwd, 'XJTU Battery Dataset');
outDir  = fullfile(pwd, 'processed');
%%
if ~exist(outDir,'dir'), mkdir(outDir); end

% Collect all .mat battery files recursively
fileList = dir(fullfile(rootDir, '**', '*.mat'));

% Optional: skip metadata PDFs or temperature compensation file
skipNames = ["Temperature_Compensation_Data.mat"];
fileList = fileList(~ismember(string({fileList.name}), skipNames));

% Options for cycle building
opts = struct('DischargeOnly', true, ...
              'UseAbsCurrent', true, ...
              'MinSamplesPerCycle', 10, ...
              'ClipVoltage', []);      % e.g., [3.0 4.2] if needed

allCycles = table();
logRows   = strings(0,1);

for k = 1:numel(fileList)
    f = fullfile(fileList(k).folder, fileList(k).name);
    try
        % 1) Load -> timetable with Time, Current, Voltage, (Temp) + CycleID
        tt = loadXJTUFileToTimetable(f);

        % 2) Ensure CycleID exists (create if missing)
        tt = ensureCycleID(tt);

        % 3) Build per-cycle features + SoH
        nominalCapAh = estimateNominalCapacity(tt);       % robust guess from first discharge
        cycles = buildCycles(tt, nominalCapAh, opts);

        % Prefer ground-truth discharge capacity from the file if present
        S = load(f);
        if isfield(S,'summary') && isfield(S.summary,'discharge_capacity_Ah')
            Qsum = S.summary.discharge_capacity_Ah(:);
            assert(height(cycles) == numel(Qsum), 'summary/cycles length mismatch for %s', f);
            cycles.Capacity_Ah = Qsum;  % override integrated capacity
        end
        
        % Robust SoH reference (ignore formation/conditioning)
        Q = cycles.Capacity_Ah(:);
        startIdx = 4;                        % skip first 3 cycles
        endIdx   = min(50, numel(Q));        % early plateau window
        cand = Q(startIdx:endIdx);
        if isempty(cand)
            Q_ref = max(Q);
        else
            k = min(5, numel(cand));
            Q_ref = median(maxk(cand, k));   % plateau-like reference
        end
        cycles.SoH = min(Q ./ Q_ref, 1);      % clamp >1 to 1

        % 4) Add identifiers
        [~, baseName, ~] = fileparts(f);
        cycles.File = repmat(string(baseName), height(cycles), 1);
        cycles.Batch = repmat(string(getBatchName(f)), height(cycles), 1);

        % 5) Save per-file cycles and append to combined table
        save(fullfile(outDir, baseName + "_cycles.mat"), 'cycles');
        allCycles = [allCycles; cycles]; %#ok<AGROW>

        logRows(end+1) = "OK  : " + f; %#ok<SAGROW>
    catch ME
        logRows(end+1) = "FAIL: " + f + " | " + ME.message; %#ok<SAGROW>
        warning("Failed %s: %s", f, ME.message);
    end
end

% Save combined
if ~isempty(allCycles)
    writetable(allCycles, fullfile(outDir, 'XJTU_all_cycles.csv'));
    save(fullfile(outDir, 'XJTU_all_cycles.mat'), 'allCycles');
end

% Print a short report
fprintf('\n=== LOAD REPORT ===\n');
fprintf('%s\n', logRows);
fprintf('Total files: %d | Succeeded: %d | Failed: %d\n', ...
    numel(fileList), sum(startsWith(logRows,"OK")), sum(startsWith(logRows,"FAIL")));

% Quick sanity prints
if ~isempty(allCycles)
    fprintf('\nCombined cycles: %d rows from %d files\n', height(allCycles), numel(unique(allCycles.File)));
    fprintf('Capacity range [Ah]: [%.3f, %.3f]\n', min(allCycles.Capacity_Ah), max(allCycles.Capacity_Ah));
    fprintf('SoH range: [%.3f, %.3f]\n', min(allCycles.SoH), max(allCycles.SoH));
end

%% (Optional) Make LSTM-ready sequences here or in a separate training script
% Example:
% W = 20; features = ["Capacity_Ah","V_mean","V_std","t_sec","T_mean"];
% [X, Y] = makeSequences(allCycles, W, features);
% ... split train/val/test, normalize, trainNetwork(...)


%% ---------- Local helpers below (or put each in its own .m file) ----------



function T = normalizeToTable(raw)
    % Map common XJTU-style field names to standard: Time, Current, Voltage, Temp (optional), CycleID (optional)
    % Accepts:
    % - struct with fields as vectors or cell arrays
    % - table/timetable with arbitrary column names

    if istable(raw) || istimetable(raw)
        T = timetable2table(raw); %#ok<TNMLP>
    elseif isstruct(raw)
        % If struct array with per-sample fields:
        try
            T = struct2table(raw);
        catch
            % Sometimes the struct holds nested fields; try to pick typical ones
            error('Unsupported struct layout; please inspect file schema.');
        end
    else
        error('Unsupported variable type');
    end

    % Candidate name lists
    timeNames    = ["Time","time","t","T","Timestamp","system_time"];
    currentNames = ["Current","current","I","curr","current_A"];
    voltageNames = ["Voltage","voltage","V","volt","voltage_V"];
    tempNames    = ["Temp","Temperature","temperature","temp","T_degC","temp_C", "temperature_C"];
    cycleNames   = ["CycleID","cycle","Cycle","cycle_id","nCycle","Step","cycleNumber"];

    % Helper to pick first present
    pick = @(cands) string(T.Properties.VariableNames(ismember(lower(string(T.Properties.VariableNames)), lower(cands))));
    % Map
    tn = pick(timeNames);    if isempty(tn), error('Time column not found'); end
    in = pick(currentNames); if isempty(in), error('Current column not found'); end
    vn = pick(voltageNames); if isempty(vn), error('Voltage column not found'); end
    cn = pick(cycleNames);   % may be empty
    qn = pick(tempNames);    % optional

    T = renamevars(T, tn(1), "Time");
    T = renamevars(T, in(1), "Current");
    T = renamevars(T, vn(1), "Voltage");
    if ~isempty(cn), T = renamevars(T, cn(1), "CycleID"); end
    if ~isempty(qn), T = renamevars(T, qn(1), "Temp");    end

    % Keep only relevant columns
    keep = ["Time","Current","Voltage","Temp","CycleID"];
    keep = keep(ismember(keep, string(T.Properties.VariableNames)));
    T = T(:, keep);
end

function tt = loadXJTUFileToTimetable(filePath)
    % 加载 .mat 文件 load '.mat' file
    raw = load(filePath);
    
    % XJTU 数据集包含在一个叫 'data' 或与文件名同名的结构体中
    % 这里做一个简单的解包逻辑
    raw = raw.('data');
    
    % ============================================================
    % 1. 调用已有的 normalizeToTable 将其转为标准 Table
    % ============================================================
    T = normalizeToTable(raw);
    
    % ============================================================
    % 2. 展开嵌套的 Cell 列 (Flattening)
    % ============================================================
    T = flattenNestedTable(T); % <--- 调用下方定义的辅助函数

    % 删除空行
    nanRows = cellfun(@(x) isnumeric(x) && isnan(x), T.Time);
    T(nanRows, :) = [];
    % 显示删除了多少行
    fprintf('删除了 %s 中 %d 行包含 NaN 的时间数据\n', filePath, sum(nanRows));

    % ============================================================
    % 3. 转换为 timetable
    % ============================================================
    % 将格式为 "yyyy-MM-dd,HH:mm:ss"的字符串转换为 datetime 对象
    timeData = T.Time;
    newTime = datetime(timeData, 'InputFormat', 'yyyy-MM-dd,HH:mm:ss');
    
    % 将转换后的时间赋值回 Table
    T.Time = newTime;
    
    tt = table2timetable(T, 'RowTimes', T.Time);
end

function T_flat = flattenNestedTable(T)
    vars = T.Properties.VariableNames;
    numRows = height(T); % 这里应该是 390
    
    % ============================================================
    % 在展开之前，强制创建一个基于行索引的 CycleID
    % ============================================================
    % 即使原始数据里没有 CycleID，我们也默认第 i 行就是第 i 个循环
    if ~ismember('CycleID', vars)
        T.CycleID = (1:numRows)'; 
        vars = [vars, {'CycleID'}]; % 更新变量列表
    end
    % ============================================================

    isSequence = false(1, numel(vars));
    for i = 1:numel(vars)
        col = T.(vars{i});
        if iscell(col) && ~all(cellfun(@isscalar, col)) % 只有非标量的 cell 才算序列
            isSequence(i) = true;
        end
    end
    
    newData = struct();
    
    % 获取每个循环的长度 (采样点数)
    % 假设 Time 列存在且是序列，用它来决定长度
    if ismember('Time', vars)
        lengths = cellfun(@(x) numel(x), T.Time);
    else
        % 如果没有 Time，尝试找任意一个序列列
        seqCols = vars(isSequence);
        lengths = cellfun(@(x) numel(x), T.(seqCols{1}));
    end

    for i = 1:numel(vars)
        name = vars{i};
        col = T.(name);
        
        if isSequence(i)
            % 序列列：垂直堆叠 (Flatten)
            temp = vertcat(col{:});
            % 处理特殊的时间嵌套格式
            if iscell(temp) && (isdatetime(temp{1}) || isduration(temp{1}))
                 try temp = [temp{:}]'; catch, temp = cellfun(@(x) x, temp); end
            end
            newData.(name) = temp;
        else
            % 标量列 (包括我们刚才创建的 CycleID)：按长度复制
            % 例如：第1行有100个点，那么 CycleID=1 就要重复100次
            newData.(name) = repelem(col, lengths);
        end
    end
    
    T_flat = struct2table(newData);
end

function tt = ensureCycleID(tt)
    % Pass-through if CycleID exists (ensure numeric)
    hasCycle = any(strcmp('CycleID', tt.Properties.VariableNames));
    if hasCycle
        cid = tt.CycleID;
        if iscell(cid), cid = cellfun(@double, cid); end
        if ~isnumeric(cid), cid = double(cid); end
        tt.CycleID = cid;
        return;
    end

    % Derive CycleID using row times + current
    trow = tt.Properties.RowTimes;
    if isdatetime(trow)
        t = seconds(trow - trow(1));
    elseif isduration(trow)
        t = seconds(trow - trow(1));
    else
        error('ensureCycleID: Unsupported RowTimes type.');
    end
    dt = [0; diff(t)];
    I  = tt.Current;

    gapReset   = dt > 60;                 % long pause
    timeReset  = [false; diff(t) < 0];   % safety (shouldn't happen after loader)
    signChange = [false; diff(sign(I)) ~= 0];

    boundaries = gapReset | timeReset | signChange;
    CycleID = cumsum(boundaries) + 1;
    tt.CycleID = CycleID;
end


function nominalCapAh = estimateNominalCapacity(tt)
    % Crude but robust estimate: integrate the first discharge cycle with enough samples
    optsLocal = struct('DischargeOnly', true, 'UseAbsCurrent', true, 'MinSamplesPerCycle', 10, 'ClipVoltage', []);
    cyclesTmp = buildCycles(tt, 1.0, optsLocal);   % temporary nominalCap=1 to get Capacity_Ah
    if isempty(cyclesTmp)
        nominalCapAh = 1.0;   % fallback; avoids division-by-zero
    else
        % Use the maximum capacity among the first few cycles (more stable than the very first)
        k = min(5, height(cyclesTmp));
        nominalCapAh = max(cyclesTmp.Capacity_Ah(1:k));
        if ~isfinite(nominalCapAh) || nominalCapAh <= 0
            nominalCapAh = max(cyclesTmp.Capacity_Ah);
        end
        if ~isfinite(nominalCapAh) || nominalCapAh <= 0
            nominalCapAh = 1.0;
        end
    end
end

function b = getBatchName(pathStr)
    % Extract "Batch-x" folder name if present
    parts = split(string(pathStr), filesep);
    isBatch = startsWith(parts, "Batch-");
    if any(isBatch)
        b = parts(find(isBatch,1,'last'));
    else
        b = "Batch-NA";
    end
end