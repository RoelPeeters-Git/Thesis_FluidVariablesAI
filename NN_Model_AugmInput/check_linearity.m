[theta, RMStrain, sim_ftrain] = fcheck_linearity('Utrain.csv', 'ftrain.csv');


Udev = load('Udev.csv');
fdev = load('fdev.csv');
Utest = load('Utest.csv');
ftest = load('ftest.csv');

devsim = Udev*theta;
testsim = Utest*theta;

rRmsdev = rms(devsim-fdev)/rms(fdev);
rRmstest = rms(testsim-ftest)/rms(ftest);