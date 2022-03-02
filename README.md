# avazu-ctr-prediction
avazu-ctr-prediction challange from kaggle

I loaded the data and did two version of outlier detection. One is using the moving average over all data point straight. The second version is basically a outlier detection for all hours of the day by stackign all the ten days. As well I set it up to runs once with a simple detrend (delta aka difference) and a second time straight as well with no further processing. The window size I used is 6 hours. 
