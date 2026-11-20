function run_refs(matdir, outfile, solvers, nrestart, pattern, root)
% Run the hypergraph-matching solvers on exported matching tensors (export_mat.py) with random restarts.
%   matdir  : folder of <instance>.mat files written by export_mat.py
%   outfile : result file (MAT, -v7)
%   solvers : cell array of 'adapt_bcagm3' | 'bcagm3' | 'rrwhm'
%   nrestart: restarts per (instance, solver); restart 1 starts from the all-ones X0 of the solver code,
%             restarts >= 2 from X0 = 0.5 + rand (positive, different per restart, fixed seed 12345).
%   pattern : file pattern inside matdir (default '*.mat')
%   root    : folder of the third-party solver code (default: third_party/hypergraph_mirror next to this file)
% Timing per restart covers the solver call plus the Hungarian discretization (not file loading).
% Output: for every instance a struct with, per solver, perms (R x n; perm(i) = target of source i, stored 0-based),
% times (R x 1, seconds) and score (R x 1, the solver's own score 6 Psi of the discretized assignment).
here = fileparts(mfilename('fullpath'));
if nargin < 5 || isempty(pattern), pattern = '*.mat'; end
if nargin < 6 || isempty(root), root = fullfile(here, 'third_party', 'hypergraph_mirror'); end
addpath(fullfile(root, 'methods', 'BCAGM'), fullfile(root, 'methods', 'BCAGM', 'cpp'), ...
        fullfile(root, 'methods', 'RRWHM'), fullfile(root, 'utils'));
files = dir(fullfile(matdir, pattern));
res = struct();
rng(12345);
for f = 1:numel(files)
    S = load(fullfile(files(f).folder, files(f).name));
    n = double(S.nP);
    problem.indH3 = S.indH3; problem.valH3 = S.valH3;
    alg.bOrder = [0 0 1];
    name = matlab.lang.makeValidName(erase(files(f).name, '.mat'));
    for s = 1:numel(solvers)
        sv = solvers{s};
        perms = zeros(nrestart, n); times = zeros(nrestart, 1); score = zeros(nrestart, 1);
        for r = 1:nrestart
            if r == 1, X0 = ones(n, n); else, X0 = 0.5 + rand(n, n); end
            problem.X0 = X0;
            tic;
            switch sv
                case 'adapt_bcagm3'
                    X = bcagm3(alg, problem, repmat(X0(:), 3, 1), 50, 1);
                case 'bcagm3'
                    X = bcagm3(alg, problem, repmat(X0(:), 3, 1), 50, 0);
                case 'rrwhm'
                    evalc('X = RRWHM(alg, problem);');
                otherwise
                    error('unknown solver %s', sv);
            end
            % Hungarian discretization of the (targets x sources) matrix; maximize
            C = max(X(:)) - X;
            M = matchpairs(C, 1e9);               % rows = targets, cols = sources
            times(r) = toc;
            p = zeros(1, n); p(M(:, 2)) = M(:, 1) - 1;
            perms(r, :) = p;
            lin = (0:n-1) * n + p;                % assignment code i*n + a
            score(r) = sum_score(S.indH3, S.valH3, lin, n);
        end
        res.(name).(sv).perms = perms;
        res.(name).(sv).times = times;
        res.(name).(sv).score = score;
    end
    fprintf('%s done\n', name);
end
save(outfile, 'res', '-v7');
end

function sc = sum_score(ind, val, lin, n)
x = zeros(n * n, 1); x(lin + 1) = 1;
sc = sum(val .* x(ind(:, 1) + 1) .* x(ind(:, 2) + 1) .* x(ind(:, 3) + 1));
end
