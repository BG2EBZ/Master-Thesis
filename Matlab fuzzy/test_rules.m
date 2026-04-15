fis = readfis('human_states_following.fis');

% 关键点采样：覆盖每个MF的边界和中心
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
fprintf('总共 %d 组输入组合\n', size(inputs,1)); % 8*7*7*7 = 2744组

outputs = evalfis(fis, inputs);

results = array2table([inputs, outputs], ...
    'VariableNames', {'following_time','hhd','hrd','density', ...
                      'overwhelmed','distracted','impatient','engaged'});

% ---- 盲区检测 ----
% 所有输出都很低：说明没有规则被激活
all_out_max = max(outputs, [], 2);
dead_idx = all_out_max < 0.1;
dead_zones = results(dead_idx, :);
fprintf('疑似盲区（所有输出<0.1）：%d 组\n', sum(dead_idx));

% ---- 逻辑冲突检测 ----
% engaged 和 overwhelmed 同时高
conflict1 = results(outputs(:,1) > 0.6 & outputs(:,4) > 0.6, :);
fprintf('engaged与overwhelmed同时高：%d 组\n', height(conflict1));

% ---- 导出 ----
writetable(results, 'fis_sweep_results.csv');
writetable(dead_zones, 'dead_zones.csv');
disp('导出完成');