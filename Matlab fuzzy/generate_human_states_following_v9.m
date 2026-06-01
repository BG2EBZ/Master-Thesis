%% generate_human_states_following_v9.m
% Build the following+normal fuzzy inference system defined in
% museum_env_package/museum_env/fuzzy/human_states.py and export it as a
% reusable .fis file.

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
output_file = fullfile(script_dir, 'human_states_following_v9.fis');

fis = newfis( ...
    'human_states_following_v9', ...
    'mamdani', ...
    'min', ...
    'max', ...
    'min', ...
    'max', ...
    'centroid' ...
);

% Inputs: following_time, hhd, hrd, density, angle.
fis = addvar(fis, 'input', 'following_time', [0 60]);
fis = addmf(fis, 'input', 1, 'short', 'trapmf', [0 0 15 25]);
fis = addmf(fis, 'input', 1, 'medium', 'trapmf', [15 25 35 45]);
fis = addmf(fis, 'input', 1, 'long', 'trapmf', [35 45 60 60]);

fis = addvar(fis, 'input', 'hhd', [0 4]);
fis = addmf(fis, 'input', 2, 'close', 'trapmf', [0 0 0.5 0.7]);
fis = addmf(fis, 'input', 2, 'medium', 'trapmf', [0.5 0.7 1.0 1.2]);
fis = addmf(fis, 'input', 2, 'far', 'trapmf', [1.0 1.2 4.0 4.0]);

fis = addvar(fis, 'input', 'hrd', [0 4]);
fis = addmf(fis, 'input', 3, 'close', 'trapmf', [0 0 0.8 1.0]);
fis = addmf(fis, 'input', 3, 'medium', 'trapmf', [0.8 1.0 2.0 2.2]);
fis = addmf(fis, 'input', 3, 'far', 'trapmf', [2.0 2.2 5.0 5.0]);

fis = addvar(fis, 'input', 'density', [0 10]);
fis = addmf(fis, 'input', 4, 'low', 'trapmf', [0 0 3 3]);
fis = addmf(fis, 'input', 4, 'medium', 'trapmf', [4 4 7 7]);
fis = addmf(fis, 'input', 4, 'crowded', 'trapmf', [8 8 10 10]);

fis = addvar(fis, 'input', 'angle', [-180 180]);
fis = addmf(fis, 'input', 5, 'ahead', 'trapmf', [-60 -45 45 60]);

% Outputs: engaged, overwhelmed, distracted, impatient, curiosity.
output_names = {'engaged', 'overwhelmed', 'distracted', 'impatient', 'curiosity'};
for output_idx = 1:numel(output_names)
    fis = addvar(fis, 'output', output_names{output_idx}, [0 1]);
    fis = addmf(fis, 'output', output_idx, 'low', 'trapmf', [0 0 0.2 0.5]);
    fis = addmf(fis, 'output', output_idx, 'medium', 'trimf', [0.2 0.5 0.8]);
    fis = addmf(fis, 'output', output_idx, 'high', 'trapmf', [0.5 0.8 1.0 1.0]);
end

% Each row is:
% [following_time hhd hrd density angle engaged overwhelmed distracted impatient curiosity weight and_or]
rules = [
    0 1 1 3 0, 0 3 0 0 0, 1 1;  % overwhelmed high
    0 3 3 1 0, 0 0 3 0 0, 1 1;  % distracted high
    0 2 3 1 0, 0 0 3 0 0, 1 1;
    0 3 2 1 0, 0 0 3 0 0, 1 1;
    0 2 2 1 0, 0 0 3 0 0, 1 1;
    0 1 1 1 0, 0 0 0 3 0, 1 1;  % impatient high
    0 1 1 2 0, 0 0 0 3 0, 1 1;
    0 0 1 0 1, 0 0 0 0 3, 1 1;  % curiosity high
    0 0 2 0 1, 0 0 0 0 3, 1 1;
    0 0 3 0 1, 0 0 0 0 3, 1 1;
    0 0 0 1 0, 3 0 0 0 0, 1 1;  % engaged high
    0 0 0 2 0, 3 0 0 0 0, 1 1;
    0 2 0 3 0, 2 0 0 0 0, 1 1;  % engaged medium
    0 3 0 3 0, 2 0 0 0 0, 1 1;
    0 1 2 3 0, 2 0 0 0 0, 1 1;
    0 1 3 3 0, 2 0 0 0 0, 1 1;
    3 1 1 0 0, 1 0 0 0 0, 1 1;  % engaged low
    1 0 0 0 0, 3 0 0 0 0, 1 1;  % engaged high
    3 0 3 1 0, 1 0 0 0 0, 1 1;  % engaged low
    0 2 0 0 0, 3 0 0 0 0, 1 1;  % engaged high
    0 0 2 0 0, 3 0 0 0 0, 1 1;
    0 0 0 0 1, 2 0 0 0 0, 1 1   % engaged medium
];

fis = addrule(fis, rules);
writefis(fis, output_file);

fprintf('Exported FIS to:\n  %s\n', output_file);
fprintf('Inputs: %d, Outputs: %d, Rules: %d\n', ...
    numel(fis.input), numel(fis.output), numel(fis.rule));
