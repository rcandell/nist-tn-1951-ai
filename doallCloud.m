function doallCloud()
% NIST-TN-1951-AI: doallCloud.m
% Description: Main script for cloud-based processing and analysis of NIST 1951 industrial wireless propagation
%              measurements using AI-driven techniques. This script manages cloud-integrated workflows, including
%              data preprocessing, AI model execution, and result generation for industrial wireless scenarios.
% Author: Rick Candell, NIST
% Contact: For inquiries, visit https://www.nist.gov/programs-projects/wireless-systems-industrial-environments
% Dependencies: Requires code from https://github.com/rcandell/IndustrialWirelessAnalysis
% Citation: If you use or extend this code, please cite https://doi.org/10.18434/T4359D
% Disclaimer: Certain tools, equipment, or materials identified do not imply NIST endorsement.
% Created: July 2025
% License: GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

MANIFEST_PATH = './manifest.xlsx';
TEST=false;
if TEST
    doall_test(MANIFEST_PATH)
else
    doall_straight(MANIFEST_PATH)
end

end

function doall_test(manifest_path)
    global TEST_DATA
    if isempty(TEST_DATA)
        TEST_DATA = load("dataCloud\Tx3_5G_3115Vpol_Cloudat29_run1_pp.mat");
    end
    top_dirs = { './dataCloud'};
    for ii = 1:length(top_dirs)
        
        cd(top_dirs{ii})
    
        analyze_cwd('all', manifest_path, TEST_DATA);
        
        cd('..\')
    
    end
    
end

function doall_straight(manifest_path)
    % Analyze complex impulse responses from measurements
    % Author: Rick Candell
    % Organization: National Institute of Standards and Technology
    % Email: rick.candell@nist.gov
    
    %% mobile measurements
    delete('stats.dat')
    top_dirs = { './dataCloud'};
    figvis = false;
    for ii = 1:length(top_dirs)
        
        cd(top_dirs{ii})
    
        analyze_cwd('all', manifest_path);
        
        cd('..\')
    
    end

end

function analyze_cwd(OPTS, manifest_path, TEST_DATA)
% Analyze complex impulse responses from measurements
% Author: Rick Candell
% Organization: National Institute of Standards and Technology
% Email: rick.candell@nist.gov

    set(0,'DefaultFigureWindowStyle','docked')
    set(0, 'DefaultFigureVisible', 'off')
    
    TESTING = false;
    if nargin == 3
        TESTING = true;
    end

    if nargin < 2
        error('you forgot the manifest path')
    else
        opts = detectImportOptions(manifest_path);
        opts.VariableTypes = {'double', 'string', 'string', 'double', 'double', 'string', 'double', 'double', 'double'};
        manifest_tbl = readtable(manifest_path, opts);
    end
    
    if nargin < 1 || isempty(OPTS)
        OPTS = ...
        [ ...   
            1; ...  % compute path gain
            1; ...  % K factor
            1; ...  % delay spread
            1; ...  % compute average CIR from data
            1; ...  % NTAP estimation
            1; ...  % write stats file
            0; ...  % do plots
        ]; 
        disp('using the following options')
        disp(OPTS)
    elseif strcmp(OPTS,'all')
        disp('enabling all options')
        OPTS = ...
        [ ...   
            1; ...  % compute path gain
            1; ...  % K factor
            1; ...  % delay spread
            1; ...  % compute average CIR from data
            0; ...  % ntap approx
            1; ...  % write stats file        
            0; ...  % do plots
        ];     
    end
    
    pattern = '*.mat';
    C = MeasurementRun(OPTS, manifest_tbl);
    if TESTING
        C.estimate_cloud_cwd(pattern, TEST_DATA);
    else
        C.estimate_cloud_cwd(pattern);
    end
    
    set(0, 'DefaultFigureVisible', 'on')

end