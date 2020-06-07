
[theta, RMStrain, output_sim] = fcheck_linearity('Utrain.csv','ftrain.csv');
testinput = load('Utest.csv');
testoutput = load('ftest.csv');

testsim = testinput*theta;
error = testoutput-testsim;
RMStest = rms(error)/rms(testoutput);

save('lincheck_result')