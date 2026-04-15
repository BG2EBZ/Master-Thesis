fis = readfis('human_states_following_v6.fis');

% Representative values for each Membership Function (MF) - using center of plateaus
% following_time: short[0-10], medium[20-40], long[50-60]
% hhd:            close[0-0.5], medium[0.6-0.9], far[1-4]
% hrd:            close[0-1],   medium[1.2-2],   far[2.2-5]
% density:        low[0-1],     medium[2-4],     crowded[5-10]

ft_vals      = [5,    30,  55 ];   % short, medium, long
hhd_vals     = [0.25, 0.75, 2.0];  % close, medium, far
hrd_vals     = [0.5,  1.6,  3.5];  % close, medium, far
density_vals = [0.5,  3.0,  7.0];  % low, medium, crowded

ft_labels      = {'short','medium','long'};
hhd_labels     = {'close','medium','far'};
hrd_labels     = {'close','medium','far'};
density_labels = {'low','medium','crowded'};
state_labels   = {'overwhelmed','distracted','impatient','engaged'};

% Construct 81 sets of inputs
n = 0;
results = struct();

for i = 1:3
    for j = 1:3
        for k = 1:3
            for l = 1:3
                n = n + 1;
                input = [ft_vals(i), hhd_vals(j), hrd_vals(k), density_vals(l)];
                output = evalfis(fis, input);

                % Identify dominant state (highest output value)
                [max_val, max_idx] = max(output);

                results(n).following_time = ft_labels{i};
                results(n).hhd            = hhd_labels{j};
                results(n).hrd            = hrd_labels{k};
                results(n).density        = density_labels{l};
                results(n).overwhelmed    = output(1);
                results(n).distracted     = output(2);
                results(n).impatient      = output(3);
                results(n).engaged         = output(4);
                results(n).dominant_state = state_labels{max_idx};
                results(n).dominant_value = max_val;
            end
        end
    end
end

% Convert to table
T = struct2table(results);
disp(T)

% Export to CSV
writetable(T, 'fis_81_combinations.csv');
fprintf('Done. Total %d combinations exported to fis_81_combinations.csv\n', n);

% Statistical distribution by dominant state
fprintf('\n=== Dominant State Distribution ===\n');
for s = state_labels
    count = sum(strcmp(T.dominant_state, s{1}));
    fprintf('  %s: %d sets (%.1f%%)\n', s{1}, count, count/81*100);
end