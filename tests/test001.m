cd('D:\postgraduate\master_thesis\Chalmers-Master-Thesis')

S = load('processed\2C_battery-1_cycles.mat');
cycles = S.cycles;

figure;
subplot(2,1,1);
plot(cycles.Cycle, cycles.Capacity_Ah, '-o');
xlabel('Cycle'); ylabel('Capacity [Ah]');
grid on;
title('Capacity vs Cycle');

subplot(2,1,2);
plot(cycles.Cycle, cycles.SoH, '-o');
xlabel('Cycle'); ylabel('SoH');
grid on;
title('SoH vs Cycle');