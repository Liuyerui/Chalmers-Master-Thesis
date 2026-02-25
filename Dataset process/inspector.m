arbitraryFile = load("XJTU Battery Dataset/Batch-1/2C_battery-1.mat");
arbitraryFile2 = load("XJTU Battery Dataset/Batch-2/3C_battery-3.mat");
data    = arbitraryFile2.data;
summary = arbitraryFile2.summary;               % ✅ top-level, not data(i).summary

numCycles = numel(data);
if isfield(summary,'discharge_capacity_Ah')
    cycleCapacities = summary.discharge_capacity_Ah(:);
else
    % fallback: derive from cumulative capacity_Ah per cycle
    cycleCapacities = zeros(numCycles,1);
    for i = 1:numCycles
        di = data(i);
        I  = di.current_A(:);
        Q  = di.capacity_Ah(:);
        isDis = I < 0;
        cycleCapacities(i) = max(Q(isDis)) - min(Q(isDis));
    end
end

plot(1:numCycles, cycleCapacities, '-o');
xlabel('Cycle number'); ylabel('Discharge capacity [Ah]');
title('Capacity fade over cycles');
