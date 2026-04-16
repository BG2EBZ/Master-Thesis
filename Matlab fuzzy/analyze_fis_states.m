%% analyze_fis_states.m
% Analyze a Mamdani FIS with output order:
% [engaged, overwhelmed, distracted, impatient]
%
% This script computes:
% 1) dominant state distribution
% 2) ambiguous / near-tie rate
% 3) low-confidence rate
% 4) flat-output rate
% 5) pairwise conflict rates
%
% It supports two input-generation modes:
%   - grid sweep over representative / boundary points
%   - random sampling over the full input ranges
%
% Outputs:
%   fis_analysis_all.csv
%   fis_analysis_ambiguous.csv
%   fis_analysis_low_confidence.csv
%   fis_analysis_flat.csv
%   fis_analysis_conflicts.csv
%   fis_analysis_summary.csv
%
% ------------------------------------------------------------
clear; clc;

%% ---------------- USER SETTINGS ----------------
fis_file = 'human_states_following_v8.fis';

% Input generation mode: 'grid' or 'random'
mode = 'grid';

% Thresholds
high_th   = 0.60;   % state considered "high"
lowconf_th = 0.55;  % max output below this => low confidence
margin_th = 0.05;   % top1-top2 below this => ambiguous / near-tie
spread_th = 0.10;   % max-min below this => flat output region

% Random mode settings
N_random = 5000;
rng(42); % for reproducibility

% Grid mode settings: representative + boundary-aware sample points
ft_grid      = [0, 5, 15, 25, 35, 45, 55, 60];
hhd_grid     = [0, 0.3, 0.55, 0.75, 0.95, 2.0, 4.0];
hrd_grid     = [0, 0.5, 1.1, 1.6, 2.1, 3.0, 4.0];
density_grid = [0, 0.5, 2.0, 3.0, 5.0, 7.0, 10.0];

% Output order in the current FIS
state_labels = {'engaged','overwhelmed','distracted','impatient'};

%% ---------------- LOAD FIS ----------------
fis = readfis(fis_file);

%% ---------------- GENERATE INPUTS ----------------
switch lower(mode)
    case 'grid'
        [FT, HHD, HRD, DENS] = ndgrid(ft_grid, hhd_grid, hrd_grid, density_grid);
        inputs = [FT(:), HHD(:), HRD(:), DENS(:)];
        fprintf('Mode: GRID\n');
        fprintf('Total grid combinations: %d\n', size(inputs,1));

    case 'random'
        ft_rand      = 60 * rand(N_random, 1);
        hhd_rand     =  4 * rand(N_random, 1);
        hrd_rand     =  4 * rand(N_random, 1);
        density_rand = 10 * rand(N_random, 1);
        inputs = [ft_rand, hhd_rand, hrd_rand, density_rand];
        fprintf('Mode: RANDOM\n');
        fprintf('Total random samples: %d\n', size(inputs,1));

    otherwise
        error('Unknown mode. Use ''grid'' or ''random''.');
end

%% ---------------- EVALUATE FIS ----------------
outputs = evalfis(fis, inputs);

% Safety check
if size(outputs,2) ~= 4
    error('Expected 4 outputs, but got %d. Check FIS output order.', size(outputs,2));
end

engaged     = outputs(:,1);
overwhelmed = outputs(:,2);
distracted  = outputs(:,3);
impatient   = outputs(:,4);

%% ---------------- DERIVED METRICS ----------------
[max_out, max_idx] = max(outputs, [], 2);
[min_out, ~]       = min(outputs, [], 2);

sorted_out = sort(outputs, 2, 'descend');
top1   = sorted_out(:,1);
top2   = sorted_out(:,2);
margin = top1 - top2;
spread = max_out - min_out;

tie_tol = 1e-6;   % near-tie tolerance

% If top1 and top2 are nearly tied, force engaged
near_tie_idx = margin < tie_tol;
max_idx(near_tie_idx) = 1;      % engaged
max_out(near_tie_idx) = outputs(near_tie_idx, 1);

% Dominant state names
dominant_state = strings(size(max_idx));
for i = 1:numel(state_labels)
    dominant_state(max_idx == i) = state_labels{i};
end

% Core flags
low_conf_flag  = max_out < lowconf_th;
ambiguous_flag = margin < margin_th;
flat_flag      = spread < spread_th;

% Pairwise conflicts (you can adjust which pairs are meaningful)
conf_eng_ovw = engaged     > high_th & overwhelmed > high_th;
conf_eng_dis = engaged     > high_th & distracted  > high_th;
conf_eng_imp = engaged     > high_th & impatient   > high_th;
conf_ovw_dis = overwhelmed > high_th & distracted  > high_th;
conf_ovw_imp = overwhelmed > high_th & impatient   > high_th;
conf_dis_imp = distracted  > high_th & impatient   > high_th;

any_conflict_flag = conf_eng_ovw | conf_eng_dis | conf_eng_imp | ...
                    conf_ovw_dis | conf_ovw_imp | conf_dis_imp;

%% ---------------- BUILD MAIN TABLE ----------------
T = table();
T.following_time = inputs(:,1);
T.hhd            = inputs(:,2);
T.hrd            = inputs(:,3);
T.density        = inputs(:,4);

T.engaged        = engaged;
T.overwhelmed    = overwhelmed;
T.distracted     = distracted;
T.impatient      = impatient;

T.max_out        = max_out;
T.top2_out       = top2;
T.margin         = margin;
T.spread         = spread;
T.dominant_state = dominant_state;

T.low_conf_flag  = low_conf_flag;
T.ambiguous_flag = ambiguous_flag;
T.flat_flag      = flat_flag;

T.conf_eng_ovw   = conf_eng_ovw;
T.conf_eng_dis   = conf_eng_dis;
T.conf_eng_imp   = conf_eng_imp;
T.conf_ovw_dis   = conf_ovw_dis;
T.conf_ovw_imp   = conf_ovw_imp;
T.conf_dis_imp   = conf_dis_imp;
T.any_conflict   = any_conflict_flag;

%% ---------------- SUBTABLES ----------------
T_ambiguous     = T(T.ambiguous_flag, :);
T_low_conf      = T(T.low_conf_flag, :);
T_flat          = T(T.flat_flag, :);
T_conflicts     = T(T.any_conflict, :);

%% ---------------- SUMMARY STATISTICS ----------------
N = height(T);

% Core rates
ambiguous_rate = mean(T.ambiguous_flag);
low_conf_rate  = mean(T.low_conf_flag);
flat_rate      = mean(T.flat_flag);
any_conf_rate  = mean(T.any_conflict);

% Conflict rates by pair
conf_rates = [ ...
    mean(T.conf_eng_ovw), ...
    mean(T.conf_eng_dis), ...
    mean(T.conf_eng_imp), ...
    mean(T.conf_ovw_dis), ...
    mean(T.conf_ovw_imp), ...
    mean(T.conf_dis_imp)  ...
];
conf_names = {'engaged_overwhelmed', 'engaged_distracted', 'engaged_impatient', ...
              'overwhelmed_distracted', 'overwhelmed_impatient', 'distracted_impatient'};

% Dominant state distribution
[state_group, ~, gid] = unique(T.dominant_state);
dom_counts = accumarray(gid, 1);
dom_rates  = dom_counts / N;

% Summary table
summary_metric = [ ...
    "total_samples";
    "ambiguous_rate";
    "low_confidence_rate";
    "flat_output_rate";
    "any_conflict_rate";
    "high_threshold";
    "lowconf_threshold";
    "margin_threshold";
    "spread_threshold" ];

summary_value = [ ...
    N;
    ambiguous_rate;
    low_conf_rate;
    flat_rate;
    any_conf_rate;
    high_th;
    lowconf_th;
    margin_th;
    spread_th ];

Summary = table(summary_metric, summary_value, 'VariableNames', {'metric','value'});

% Add dominant state distribution rows
DomSummary = table( ...
    "dominant_" + string(state_group), ...
    dom_rates, ...
    'VariableNames', {'metric','value'});

% Add pairwise conflict rows
ConfSummary = table( ...
    "conflict_" + string(conf_names(:)), ...
    conf_rates(:), ...
    'VariableNames', {'metric','value'});

Summary = [Summary; DomSummary; ConfSummary];

%% ---------------- PRINT SUMMARY ----------------
fprintf('\n========== FIS ANALYSIS SUMMARY ==========' );
fprintf('\nTotal samples           : %d', N);
fprintf('\nAmbiguous rate          : %.4f (%.2f%%)', ambiguous_rate, ambiguous_rate*100);
fprintf('\nLow-confidence rate     : %.4f (%.2f%%)', low_conf_rate,  low_conf_rate*100);
fprintf('\nFlat-output rate        : %.4f (%.2f%%)', flat_rate,      flat_rate*100);
fprintf('\nAny-conflict rate       : %.4f (%.2f%%)\n', any_conf_rate, any_conf_rate*100);

fprintf('\n--- Dominant State Distribution ---\n');
for i = 1:numel(state_group)
    fprintf('  %-14s : %6d (%.2f%%)\n', state_group(i), dom_counts(i), dom_rates(i)*100);
end

fprintf('\n--- Pairwise Conflict Rates ---\n');
for i = 1:numel(conf_names)
    fprintf('  %-24s : %.4f (%.2f%%)\n', conf_names{i}, conf_rates(i), conf_rates(i)*100);
end

%% ---------------- EXPORT CSV ----------------
writetable(T,            'fis_analysis_all.csv');
writetable(T_ambiguous,  'fis_analysis_ambiguous.csv');
writetable(T_low_conf,   'fis_analysis_low_confidence.csv');
writetable(T_flat,       'fis_analysis_flat.csv');
writetable(T_conflicts,  'fis_analysis_conflicts.csv');
writetable(Summary,      'fis_analysis_summary.csv');

fprintf('\nCSV exported:\n');
fprintf('  fis_analysis_all.csv\n');
fprintf('  fis_analysis_ambiguous.csv\n');
fprintf('  fis_analysis_low_confidence.csv\n');
fprintf('  fis_analysis_flat.csv\n');
fprintf('  fis_analysis_conflicts.csv\n');
fprintf('  fis_analysis_summary.csv\n');

%% ---------------- OPTIONAL QUICK VISUALS ----------------
% Uncomment if you want quick histograms in MATLAB.
%
% figure; histogram(T.margin, 30); title('Margin (top1 - top2)');
% xlabel('margin'); ylabel('count');
%
% figure; histogram(T.max_out, 30); title('Max Output Distribution');
% xlabel('max output'); ylabel('count');
%
% figure; histogram(categorical(T.dominant_state)); title('Dominant State Distribution');
% xlabel('state'); ylabel('count');

