fis = readfis('human_states_following_v8.fis');

% Key point sampling: covers the boundaries and centers of each MF
% following_time: short[0-20], medium[10-50], long[40-60]
ft      = [0, 5, 15, 25, 35, 45, 55, 60];
% hhd: close[0-0.6], medium[0.5-1], far[0.9-4]
hhd     = [0, 0.3, 0.55, 0.75, 0.95, 2, 4];
% hrd: close[0-1.2], medium[1-2.2], far[2-5]
hrd     = [0, 0.5, 1.1, 1.6, 2.1, 3, 4];
% density: low[0-1], medium[2-4], crowded[5-10]
density = [0, 0.5, 2, 3, 5, 7, 10];

[FT, HHD, HRD, D] = ndgrid(ft, hhd, hrd, density);
inputs = [FT(:), HHD(:), HRD(:), D(:)];
fprintf('Total input combinations: %d\n', size(inputs,1)); % 8*7*7*7 = 2744 sets

outputs = evalfis(fis, inputs);

results = array2table([inputs, outputs], ...
    'VariableNames', {'following_time','hhd','hrd','density', ...
                      'engaged','overwhelmed','distracted','impatient'});

% ---- Dead Zone Detection ----
% All outputs are very low: indicates no rules were activated
all_out_max = max(outputs, [], 2);
dead_idx = abs(all_out_max - 0.5) < 1e-5;
dead_zones = results(dead_idx, :);
fprintf('Suspected dead zones (No rules activated, outputs default to 0.5): %d sets\n', sum(dead_idx));

% ---- Logic Conflict Detection ----
% engaged and overwhelmed are high simultaneously
conflict1 = results(outputs(:,1) > 0.6 & outputs(:,2) > 0.6, :);
fprintf('Conflict: engaged and overwhelmed both high: %d sets\n', height(conflict1));

% ---- Export ----
writetable(results, 'fis_sweep_results.csv');
writetable(dead_zones, 'dead_zones.csv');
disp('Export complete');