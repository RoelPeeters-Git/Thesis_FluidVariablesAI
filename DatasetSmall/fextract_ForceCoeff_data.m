function fextract_ForceCoeff_data(readFile,writeFile,forceCoeff)

% forceCoeff:
%   - Cy
%   - Cx

load(readFile)
switch forceCoeff
    case 'Cy'
        fCoeff = forceCoeffs(:,4);
    case 'Cx'
        fCoeff = forceCoeffs(:,3);
end

clearvars -except fCoeff writeFile
csvwrite([ writeFile '.csv'],fCoeff)