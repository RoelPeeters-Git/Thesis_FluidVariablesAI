function [theta, relRMSerror, output_sim] = fcheck_linearity(inputFile,outputFile)

if nargin < 2
input  = load('Uy_grid.csv');
output = load('Cy_grid.csv');
elseif nargin == 2
input = load(inputFile);
output = load(outputFile);
end

% Cy = Uy*theta
% Uy-1*Cy = theta
% Uy \ Cy = theta

% A*X = B => X = A\B

theta = input \ output;
output_sim = input*theta;
relRMSerror = rms(output_sim-output)/rms(output)
end