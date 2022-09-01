clc
clear
close all
tic;

%% initialization
% add EPANET matlab toolkit function
addpath(genpath( './' ))
% remove the old toolkit 
rmpath(genpath('./epanet_matlab_toolkit'))
% Add .inp directory
network_path = '/network';
addpath(genpath( network_path ))
disp('All files and folder Paths Loaded.');

%% Read WDN
wdn = epanet('L-TOWN.inp');

%% Run hydraulics
hyd_res = runHydraulicsComplete(wdn);

%% Base case (2019) results
P = hyd_res.Pressure;
F = hyd_res.Flow;
D = hyd_res.Demand;
QN = hyd_res.NodeQuality;
a = hyd_res.Times;
b = [0:3600:604800];
c = find(ismember(a, b)); % hourly indices

%% Plot original consumption (hourly for one week)
hourlyDem = D(c, :);
waterConsumption(hourlyDem)

%% Get nodal IDs
nodeID = wdn.getNodeNameID;
linkID = wdn.getLinkNameID;

%% Read Lakewood water consumption data
input_path = '../data/input/';
fileName = 'week20r.csv';
filename = fullfile(input_path, fileName);
T_2020_r = readTable(filename);
fileName = 'week20nr.csv';
filename = fullfile(input_path, fileName);
T_2020_nr = readTable(filename);
% check the existance of input folder
if ~exist(input_path, 'dir')
    mkdir(input_path)
end

%% create a new pattern for residential and non residential customers
r_2020 = newPattern(T_2020_r);
nr_2020 = newPattern(T_2020_nr);

%% assign 2020 patterns to the network
setTimeSimulationDuration(wdn, 3600*24*7);
setPatternMatrix(wdn, [r_2020'; nr_2020']);
scenarioOne_patterns = wdn.getPattern;

%% Run hydraulics
hyd_res_np = runHydraulicsComplete(wdn);
P_np = hyd_res_np.Pressure;
F_np = hyd_res_np.Flow;
D_np = hyd_res_np.Demand;
QN_np = hyd_res_np.NodeQuality;
a_np = hyd_res_np.Times;
b_np = [0:3600:604800];
c_np = find(ismember(a_np, b_np)); % hourly indices

%% Plot new consumption (hourly for one week)
% % % waterConsumption(D_np)

%% Plot pressure analyses
% average pressure
% plotPressures(P(c, :), P_np(c_np, :))
% min pressure
nnodes = wdn.getNodeCount;
plotMinPressures(P, P_np)
% max pressure
plotMaxPressures(P, P_np)

%% Get the nonzero demand indices
indices_nzbd = find(wdn.getNodeBaseDemands{1}(1:920));

%% Get the water age of the system
[w_age, WA_ij] = waterAge(QN, indices_nzbd, D);
[w_age_np, WA_ij_np] = waterAge(QN_np, indices_nzbd, D_np);

%% Plot the max water age for each node
% Match the indices of nonzero demand nodes between the original and np
plotWaterAge(WA_ij, WA_ij_np)

%% Save input file
wdn.saveInputFile(fullfile('./network', 'ky10townNewCase.inp'));
wdn.unload
%%%% RUN HYDRAULICS WITH THE NEW DEMAND PATTERN %%%%%%%%%%%%%%%%

%% FUNCTIONS
function T = readTable(filename)
% Read water demand as a table
T = readtable(filename);
% Delete first row
T([1], :) = [];
% Check the format of each variable (columns represent smartMeters)
idx = find(varfun(@(x) ~isa(x, 'double'),T(:, [2:end]),'OutputFormat','uniform'));
% Delete column type cell
T(:, [idx + 1]) = [];
% Convert Lakewood units (cubic feet to liters)
cf2liters = 28.3168;
T{:, 2:end} = T{:, 2:end} .* cf2liters;
end

function new_pattern = newPattern(T)
% create new pattern based on customer type
T.TestAvg = mean(T{:, 2:end}, 2);
new_pattern = (T.TestAvg / mean(T.TestAvg)) * 0.60;
end

function plotPatterns(r, nr)
plot(r, 'DisplayName', 'Residential normalized patterns');
hold on;
plot(nr, 'DisplayName', 'Non-residential normalized patterns')
hold off;
xticks(1:6:170)
x_labels = repmat(0:6:18, 1, 7);
xticklabels({x_labels})
xlabel('Time (Hours)'); ylabel('Multiplier');
legend
end

% runHydraulics function (faster)
function hydr_response = runHydraulics(wn)
hydr_response = wn.getComputedTimeSeries;
end

% runHydraulics step by step
function hydr_response = runHydraulicsComplete(d)
% Set time hydraulic and quality steps
etstep = 300;
d.setTimeReportingStep(etstep);
d.setTimeHydraulicStep(etstep);
d.setTimeQualityStep(etstep);
% Hstep = min(Pstep,Hstep)
% Hstep = min(Rstep,Hstep)
% Hstep = min(Qstep,Hstep)

% Hydraulic and Quality analysis step-by-step
d.openHydraulicAnalysis;
d.openQualityAnalysis;
d.initializeHydraulicAnalysis(0);
d.initializeQualityAnalysis(d.ToolkitConstants.EN_NOSAVE);

tstep = 1;
P = []; D = []; H = []; F = []; S = []; E = [];
QN = []; QL = []; T = [];
while (tstep>0)
    t  = d.runHydraulicAnalysis;
    qt = d.runQualityAnalysis;
    
    P  = [P; d.getNodePressure];
    D  = [D; d.getNodeActualDemand];
    H  = [H; d.getNodeHydaulicHead];
    F  = [F; d.getLinkFlows];
    S  = [S; d.getLinkStatus];
    E  = [E; d.getLinkEnergy];
    
    QN = [QN; d.getNodeActualQuality];
    QL = [QL; d.getLinkActualQuality];
    T  = [T; t];
    
    tstep = d.nextHydraulicAnalysisStep;
    qtstep = d.nextQualityAnalysisStep;
end
d.closeQualityAnalysis;
d.closeHydraulicAnalysis;
hydr_response.Pressure = P;
hydr_response.Demand = D;
hydr_response.Head = H;
hydr_response.NodeQuality = QN;
hydr_response.Flow = F;
hydr_response.Energy = E;
hydr_response.Times = T;
end

function waterConsumption(D)
figure()
% produced = abs(D(:, 783:785));
consumed = mean(D(:, 1:782), 2) * 1000;
% Plot hourly consumption over the period of simulation
% plot(produced, 'DisplayName', 'Produced');
% hold on;
plot(consumed, 'DisplayName', 'Consumed');
hold off;
xticks(1:6:170)
% x_labels = repmat(0:6:18, 1, 7);
xticklabels(repmat({'00:00', '06:00', '12:00', '18:00'}, 1, 7))
xtickangle(45)
% xticklabels({x_labels})
xlabel('Time (Hours)'); ylabel('Simulated demand (liters)');
legend
end

function plotPressures(P, P_np)
% Average pressure of each time step over the nodes
figure()
plot(mean(P(:, 1:920), 2), 'DisplayName', 'P Normal Operating Conditions'); hold on;
plot(mean(P_np(:, 1:920), 2), 'DisplayName', 'P Covid-19')
xlim([0 size(P,1)])
xlabel('Time (Hours)')
ylabel('Average System Pressure (m)')
legend
% Average pressure of each node over the simulation
% figure()
% plot(mean(P(:, 1:399)), 'DisplayName', 'P original'); hold on;
% plot(mean(P_np(:, 1:399)), 'DisplayName', 'P new dem pattern')
% xlabel('Node (id)')
% ylabel('Pressure (m)')
% legend
end

function plotMinPressures(P, P_np)
% Average pressure of each time step over the nodes
figure()
plot(min(P(:, 1:920)), 'DisplayName', 'Min P Normal Operating Conditions'); hold on;
plot(min(P_np(:, 1:920)), 'DisplayName', 'Min P Covid-19')
xlim([0 size(P,2)])
xlabel('Node')
ylabel('Minimum Pressure (m)')
legend
end

function plotMaxPressures(P, P_np)
% Average pressure of each time step over the nodes
figure()
plot(max(P(:, 1:920)), 'DisplayName', 'Max P Normal Operating Conditions'); hold on;
plot(max(P_np(:, 1:920)), 'DisplayName', 'Max P Covid-19')
xlim([0 size(P,2)])
xlabel('Node')
ylabel('Maximum Pressure (m)')
legend
end

%% Get water age for non zero demand nodes
function [WA, WA_ij] = waterAge(QN, dem_indices, D)
D = D(:, dem_indices);
Wth = 48; % hours defined in BWN-II
WA_ij = QN(:, dem_indices);
WA = zeros(1);
for i = 1:size(WA_ij , 2)
WA(i) = sum(D(WA_ij(:,i) > Wth, i) .* (WA_ij(WA_ij(:,i) > Wth, i) - Wth)) ...
    / sum(sum(D));
end
WA = sum(WA);
end

function plotWaterAge(wAge, wAge_np)
% plot the total water age of the non zero demand nodes
% find only non-zero demand nodes
figure()
plot(max(wAge), 'DisplayName', 'WA Normal Operating Conditions');
hold on;
plot(max(wAge_np), 'DisplayName', 'WA Covid-19')
hold off;
xlabel('Node (index)')
ylabel('Water Age (hours)')
legend
end
