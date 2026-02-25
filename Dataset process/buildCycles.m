function cycles = buildCycles(data, nominalCapAh, opts)
%BUILD CYCLE FEATURES + SOH FROM RAW SAMPLE DATA (DISCHARGE-ONLY BASELINE)
% data: timetable/table with row times (or column) "Time", columns: Current, Voltage, [Temp], CycleID
% opts: DischargeOnly=true, UseAbsCurrent=true, MinSamplesPerCycle=10, ClipVoltage=[]

if nargin < 3 || isempty(opts), opts = struct; end
if ~isfield(opts,'DischargeOnly'),      opts.DischargeOnly = true; end
if ~isfield(opts,'UseAbsCurrent'),      opts.UseAbsCurrent = true; end
if ~isfield(opts,'MinSamplesPerCycle'), opts.MinSamplesPerCycle = 10; end
if ~isfield(opts,'ClipVoltage'),        opts.ClipVoltage = []; end

% --- Ensure we have a timetable with row times ---
if istimetable(data)
    ttAll = data;
else
    % Table → timetable if possible
    if any(strcmp('Time', data.Properties.VariableNames))
        tcol = data.Time;
        if isdatetime(tcol)
            rowTimes = tcol;
        elseif isduration(tcol)
            rowTimes = tcol;
        elseif isnumeric(tcol)
            rowTimes = seconds(tcol);
        else
            error('Unsupported Time column type (use datetime/duration/numeric seconds).');
        end
        data.Time = [];  % remove to avoid duplicate
        ttAll = table2timetable(data, 'RowTimes', rowTimes);
    else
        error('buildCycles: Input must be a timetable or a table with a Time column.');
    end
end

% Required variables (row times handled separately)
required = {'Current','Voltage','CycleID'};
missing  = setdiff(required, ttAll.Properties.VariableNames);
if ~isempty(missing)
    error('Missing required columns: %s', strjoin(missing, ', '));
end

G = findgroups(ttAll.CycleID);
n = max(G);

cap      = nan(n,1);
meanV    = nan(n,1);
stdV     = nan(n,1);
tCycle   = nan(n,1);
tempMean = nan(n,1);
valid    = false(n,1);
hasTemp  = any(strcmp('Temp', ttAll.Properties.VariableNames));

for i = 1:n
    idx = (G == i);
    tt  = ttAll(idx,:);

    if height(tt) < opts.MinSamplesPerCycle, continue; end

    % Optional voltage clipping
    if ~isempty(opts.ClipVoltage)
        vmin = opts.ClipVoltage(1); vmax = opts.ClipVoltage(2);
        keepV = (tt.Voltage >= vmin) & (tt.Voltage <= vmax);
        tt = tt(keepV,:);
        if height(tt) < opts.MinSamplesPerCycle, continue; end
    end

    % Row times (duration or datetime) → seconds, then diffs
    trow = tt.Properties.RowTimes;
    if isdatetime(trow)
        tsec = seconds(trow - trow(1));
    else
        % duration
        tsec = seconds(trow - trow(1));
    end

    dt = diff(tsec);                 % [N-1 x 1]
    I  = tt.Current(1:end-1);
    V  = tt.Voltage(1:end-1);
    if hasTemp, T = tt.Temp(1:end-1); end

    % Keep only strictly positive, finite dt
    tol = 1e-9;
    pos = isfinite(dt) & dt > tol;
    dt = dt(pos); I = I(pos); V = V(pos);
    if hasTemp, T = T(pos); end

    % Discharge-only segment
    if opts.DischargeOnly
        dis = I < 0;
        I = I(dis); dt = dt(dis); V = V(dis);
        if hasTemp, T = T(dis); end
    end

    if isempty(I) || numel(I) < opts.MinSamplesPerCycle, continue; end

    % Capacity (Ah)
    if opts.UseAbsCurrent
        cap(i) = sum(abs(I) .* dt) / 3600;
    else
        cap(i) = abs(sum(I .* dt) / 3600);
    end

    % Features computed on the same discharge segment
    meanV(i)  = mean(V, 'omitnan');
    stdV(i)   = std(V,  'omitnan');
    tCycle(i) = sum(dt);                              % discharge duration [s]
    if hasTemp, tempMean(i) = mean(T, 'omitnan'); end

    valid(i) = isfinite(cap(i)) && cap(i) > 0;
end

keep  = valid;
Cycle = find(keep)';                   % original cycle indices
SoH   = cap(keep) / nominalCapAh;      % will be overwritten in DataLoading.m anyway

cycles = table( ...
    Cycle(:), cap(keep), SoH(:), meanV(keep), stdV(keep), tCycle(keep), tempMean(keep), ...
    'VariableNames', {'Cycle','Capacity_Ah','SoH','V_mean','V_std','t_sec','T_mean'});
end
