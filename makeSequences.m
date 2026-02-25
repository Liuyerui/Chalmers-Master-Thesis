function [X, Y, info, Y_last] = makeSequences(cycles, W, featureNames, varargin)
% MAKESEQUENCES  Build sequence-to-one windows per File (no cross-file leakage).
%
% [X, Y, info, Y_last] = makeSequences(cycles, W, featureNames, Name,Value...)
%
% Inputs
%   cycles        : table with at least variables: File, Cycle, SoH, and featureNames
%   W             : window length (use last W cycles -> predict SoH at next)
%   featureNames  : string/cellstr of feature variable names (inputs only)
%
% Name-Value (all optional)
%   'IncludeFiles' : only use these Files (string/cellstr)
%   'ExcludeFiles' : exclude these Files (string/cellstr)
%   'SortBy'       : variables to sort within each File (default ["File","Cycle"])
%
% Outputs
%   X       : {N x 1} cell, each is [numFeat x W] double
%   Y       : [N x 1] double, target SoH at t
%   info    : struct array (File, CycleTarget, RowTarget)
%   Y_last  : [N x 1] double, Baseline A (SoH at t-1, i.e., last in window)
%
% Notes
%   - Windows are formed *within* each File only.
%   - Target Y is the SoH at the next cycle after the window.

    p = inputParser;
    addParameter(p, 'IncludeFiles', [], @(x) isstring(x) || iscellstr(x) || isempty(x));
    addParameter(p, 'ExcludeFiles', [], @(x) isstring(x) || iscellstr(x) || isempty(x));
    addParameter(p, 'SortBy', ["File","Cycle"]);
    parse(p, varargin{:});
    inc = p.Results.IncludeFiles;
    exc = p.Results.ExcludeFiles;
    sortBy = p.Results.SortBy;

    % Filter by include/exclude
    if ~isempty(inc)
        cycles = cycles(ismember(string(cycles.File), string(inc)), :);
    end
    if ~isempty(exc)
        cycles = cycles(~ismember(string(cycles.File), string(exc)), :);
    end

    % Ensure sorted
    cycles = sortrows(cycles, sortBy);

    % Group by File
    [G, files] = findgroups(cycles.File);

    X = {}; Y = []; Y_last = [];
    info = struct('File', strings(0,1), 'CycleTarget', [], 'RowTarget', []);

    for k = 1:numel(files)
        idx = find(G == k);
        T = cycles(idx, :);
        if height(T) <= W, continue; end

        Xmat = T{:, featureNames};
        SoH  = T.SoH;

        for t = (W+1):height(T)
            X{end+1,1} = Xmat(t-W:t-1, :)';      % [features x W]
            Y(end+1,1) = SoH(t);                 % target at t
            Y_last(end+1,1) = SoH(t-1);          % Baseline A (last value)

            info(end+1,1).File         = string(T.File(t));
            info(end).CycleTarget      = T.Cycle(t);
            info(end).RowTarget        = idx(t); % row in original table
        end
    end
end
