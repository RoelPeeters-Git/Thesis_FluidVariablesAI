simtestdata = input_test * theta;
testerror = output_test - simtestdata;
relRMS_testdata = rms(testerror)/rms(output_test);
