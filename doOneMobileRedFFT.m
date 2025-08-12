function doOneMobileRedFFT()
% NIST-TN-1951-AI: doallMobileRedFFT.m
% Description: Main script for extracting the fft information of the cirs.
% Author: Rick Candell, NIST
% Contact: For inquiries, visit https://www.nist.gov/programs-projects/wireless-systems-industrial-environments
% Dependencies: Requires code from https://github.com/rcandell/IndustrialWirelessAnalysis
% Citation: If you use or extend this code, please cite https://doi.org/10.18434/T4359D
% Disclaimer: Certain tools, equipment, or materials identified do not imply NIST endorsement.
% Created: July 2025
% License: GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

% files = dir("dataMobile/AAPlantD1_2GHz_TX1_hpol_run4_pp.mat");
top_dir = "dataMobile";
files = dir(top_dir + "/*.mat");
MANIFEST_PATH = './manifest.xlsx';

for ii = 1:length(files)
    MEAS_FILE_PATH = top_dir + '/' + files(ii).name;
    % do_one_straight(MANIFEST_PATH, MEAS_FILE_PATH)
    cd(top_dir);
    analyze_cwd('all', MANIFEST_PATH, MEAS_FILE_PATH);
    cd('..\');
end

end

function do_one_straight(manifest_path, meas_file_path)
    % Analyze complex impulse responses from measurements
    % Author: Rick Candell
    % Organization: National Institute of Standards and Technology
    % Email: rick.candell@nist.gov
    
    %% mobile measurements
    delete('stats.dat')
    top_dirs = { './dataMobile'};
    for ii = 1:length(top_dirs)
        
        cd(top_dirs{ii})
    
        analyze_cwd('all', manifest_path, meas_file_path);
        
        cd('..\')
    
    end

end

function analyze_cwd(OPTS, manifest_path, meas_file_path)
% Analyze complex impulse responses from measurements
% Author: Rick Candell
% Organization: National Institute of Standards and Technology
% Email: rick.candell@nist.gov

    set(0,'DefaultFigureWindowStyle','docked')
    set(0, 'DefaultFigureVisible', 'on')

    if nargin < 3
        error('you forgot the path to the measurement file');
    end

    if nargin < 2
        error('you forgot the manifest path')
    else
        opts = detectImportOptions(manifest_path);
        opts.VariableTypes = {'double', 'string', 'string', 'string', 'double', 'string', 'double', 'double', 'double'};;
        manifest_tbl = readtable(manifest_path, opts);
    end
    
    if nargin < 1 || isempty(OPTS)
        OPTS = ...
        [ ...   
            1; ...  % compute path gain
            1; ...  % K factor
            1; ...  % delay spread
            1; ...  % compute average CIR from data
            0; ...  % NTAP estimation
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

    try
        TEST_DATA = evalin('base', 'TEST_DATA');
        TESTING = 1;
    catch m
        TESTING = 0;
    end
    
    [meas_file_dir, meas_file_name, ext] = fileparts(meas_file_path);
    pattern = meas_file_name + ext;
    C = MeasurementRun(OPTS, manifest_tbl);
    C.redcir_output_path = 'ffts/' + meas_file_name + '.csv';

    if TESTING
        C.make_ffts_cwd(pattern, TEST_DATA); 
    else
        C.make_ffts_cwd(pattern);
    end
    
    set(0, 'DefaultFigureVisible', 'on')

end