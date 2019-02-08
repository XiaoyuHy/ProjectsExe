08/02/2019:

When comparing DA and kriging, could plot the RMSE diffrence  between the two models; apart from plot the mean diffrence, plot the 0.975 quartile and 0.025 quartile to account for the case 
where the distribution of the diffrence  is skewed.

By plotting the diffrence between the two models (see DataImogenFrGridMoNotCentre/RMSEdiff_outSample.png), find the the confidence interval include 0, inicating both model perform 
equally well in the case of application where we only have 128 observations.

So decided to plot the predictions and diagnosis for one particular seed.

For the simulated data, if the optimisation is not correct. could result in 0.%  out-of-sample prediction accuracy, the same for both the DA and kriging models
form example for SEED 222, have to use 7 repeats of BFGS optimisation for the Kriging model; for DA, see logs/output219_repeat2_numMo50 on callisto cluster, 
use 4 repeats of BFGS optimisation, and the 3rd and 4th repeat have similar minus_log_like, but quite different parameters for sigma2, phi2

23/01/2019:
for France Data, Gammatranformed data, Ztilde10by10
for numMo 50, for the data assimilation model, the global maximum out of 3 rounds of optimisation gives lower RMSE and higher prediction accuracy compared to one round of optimisation\
 which are trials of LBFGSB + BFGS that stopts whenever the gradients rounded to 2 decimal points equal to 0.

The same applies to the kriging model.

11/01/2019:
for FrData SEED 132/135/136 numMo 50
2 repeats of BFGS give differnt results compared to 3 repeats of LBFGSB + BFGS
The former is more accuate

for FrData SEED 124 numMo 50
LBFGSB + BFGS not working,
but 3 repeats BFGS working
Again suggests initialisation is very import for the optimisation

10/01/2019:
for simData SEED 203 numMo 50
6 repeats of BFGS optimisation still NOT working
but one of 3 repeats of LBFGSB + BFGS works
Again suggests initialisation is very import for the optimisation

06/01/2019:
From below for seed 203 numMo 500,
slightly differnt optimal results give quite different out-of-sample prediction accuracy
Again, accurate optimisation is very important for the prediction accuracy

results form cluster:
Output: /home/xx249/bayesMelding/dataSimulated/numObs_200_numMo_500/seed203/
starting optimising when withPrior is False & gpdtsMo is True& useGradsFlag is True
Starting the 0 round of optimisation in optim_NotRndStart
initial theta from previous optimisation  for BFGS is [  4.92842469  -2.03686372 -38.36460341 -12.97735356  -6.45776872
   0.98971865  -0.26982794  -0.06551927  -2.1813547 ]
theta from optimisation with BFGS when numMo is 500 is [  4.97697012  -2.01003509 -38.36460341 -12.97735461  -6.45776872
   0.95118693  -0.1822221   -0.04023809  -1.58667615]
grads at theta from optimisation with BFGS when numMo is 500 is [-2.45789630e-06  3.90272942e-06 -3.57127695e-33 -1.08508283e-07
 -5.15617631e-13 -5.14253816e-06  2.31820622e-07  3.08987503e-07
 -7.14406834e-08]
np.max(np.abs(variance_log_covPars)) in optim_NotRndStart - firstPart is 1.0741169872840657
BFGS optmisation converged successfully when numMo is 500
minus_log_like with BFGS is when numMo is 500 is 2575.073518756344
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :[array([1.45034279e+02, 1.33983973e-01, 2.18003988e-17, 2.31209942e-06,
       1.56829111e-03]), array([ 0.95118693, -0.1822221 , -0.04023809, -1.58667615])]
Optim status with BFGS :True
Optim message with BFGS :Optimization terminated successfully.
minus_log_like for 1 rounds is [2575.073518756344]
log_cov_parameters plus model bias after optimisation  is :[  4.97697012  -2.01003509 -38.36460341 -12.97735461  -6.45776872
   0.95118693  -0.1822221   -0.04023809  -1.58667615]
parameters after optimisation withPrior is  :[array([1.45034279e+02, 1.33983973e-01, 2.18003988e-17, 2.31209942e-06,
       1.56829111e-03]), array([ 0.95118693, -0.1822221 , -0.04023809, -1.58667615])]
covariance of pars after optimisation  :[[ 4.38673676e-03  1.13933248e-03  3.07596290e-35  3.76925824e-03
   1.78996744e-08 -5.59804188e-04 -1.36373398e-03  6.63548129e-04
  -5.36228497e-03]
 [ 1.13933248e-03  1.71513863e-03  3.31621347e-35 -5.19157549e-03
  -2.46546815e-08 -6.97361264e-05  4.75157770e-04  6.84231232e-04
   7.33062321e-03]
 [ 3.07596290e-35  3.31621347e-35  1.00000000e+00 -1.92363769e-35
  -9.14460997e-41  2.18108026e-35 -9.52333839e-35 -1.70493075e-35
  -7.92998002e-34]
 [ 3.76925824e-03 -5.19157549e-03 -1.92363769e-35  1.07411699e+00
   3.51961819e-07  1.24718532e-03  8.17083330e-03 -1.57220180e-03
   5.72052626e-02]
 [ 1.78996744e-08 -2.46546815e-08 -9.14460997e-41  3.51961819e-07
   1.00000000e+00  5.92292026e-09  3.88019371e-08 -7.46630300e-09
   2.71657133e-07]
 [-5.59804188e-04 -6.97361264e-05  2.18108026e-35  1.24718532e-03
   5.92292026e-09  3.50722667e-04  1.05594292e-03 -1.83377068e-04
   6.21230305e-03]
 [-1.36373398e-03  4.75157770e-04 -9.52333839e-35  8.17083330e-03
   3.88019371e-08  1.05594292e-03  2.15644658e-02  2.52592238e-03
   1.74586663e-01]
 [ 6.63548129e-04  6.84231232e-04 -1.70493075e-35 -1.57220180e-03
  -7.46630300e-09 -1.83377068e-04  2.52592238e-03  1.15183823e-02
   6.79836008e-02]
 [-5.36228497e-03  7.33062321e-03 -7.92998002e-34  5.72052626e-02
   2.71657133e-07  6.21230305e-03  1.74586663e-01  6.79836008e-02
   1.68531691e+00]]
Optim status  :True
Optim message :Optimization terminated successfully.
running time for optimisation using simData is True :57590.425080396235 seconds
output_folder in gpGaussLikeFuns is /home/xx249/bayesMelding/dataSimulated/numObs_200_numMo_500/seed203/
Out-of-sample RMSE for seed203 is :8.83083447243669
Out of sample average width of the prediction variance for seed 203 is 9.050131647232158
Out of sample prediction accuracy is 96.2%
In-sample RMSE for seed203 is :1.5276236883173303e-07
In-sample average width of the prediction variance for seed 203 is 0.0014142135409428195
In-sample prediction accuracy is 9.5%

Results from MacBook
Xiaoyus-MacBook-Pro:bayesMelding xx249$ python bayesMelding.py -numMo 500 -repeat 1 -SEED 203
Output: /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_500/seed203/
(200, 2)
starting optimising when withPrior is False & gpdtsMo is True& useGradsFlag is True
Starting the 0 round of optimisation in optim_NotRndStart
initial theta from previous optimisation  for BFGS is [  4.92842469  -2.03686372 -38.36460341 -12.97735356  -6.45776872
   0.98971865  -0.26982794  -0.06551927  -2.1813547 ]
theta from optimisation with BFGS when numMo is 500 is [  4.97697012  -2.01003508 -38.36460341 -12.97735448  -6.45776872
   0.95118693  -0.1822221   -0.04023809  -1.58667616]
grads at theta from optimisation with BFGS when numMo is 500 is [-4.98045779e-07  7.84480164e-07 -3.57127700e-33 -1.08508297e-07
 -5.15628992e-13  1.03406808e-06 -4.89978669e-07 -1.16047829e-08
  2.18853833e-08]
np.max(np.abs(variance_log_covPars)) in optim_NotRndStart - firstPart is 1.0000415907243376
BFGS optmisation converged successfully when numMo is 500
minus_log_like with BFGS is when numMo is 500 is 2575.073518756449
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :[array([1.45034279e+02, 1.33983974e-01, 2.18003988e-17, 2.31209971e-06,
       1.56829111e-03]), array([ 0.95118693, -0.1822221 , -0.04023809, -1.58667616])]
Optim status with BFGS :True
Optim message with BFGS :Optimization terminated successfully.
minus_log_like for 1 rounds is [2575.073518756449]
log_cov_parameters plus model bias after optimisation  is :[  4.97697012  -2.01003508 -38.36460341 -12.97735448  -6.45776872
   0.95118693  -0.1822221   -0.04023809  -1.58667616]
parameters after optimisation withPrior is  :[array([1.45034279e+02, 1.33983974e-01, 2.18003988e-17, 2.31209971e-06,
       1.56829111e-03]), array([ 0.95118693, -0.1822221 , -0.04023809, -1.58667616])]
covariance of pars after optimisation  :[[ 4.36035115e-03  1.15913759e-03  3.66775261e-35 -9.39730309e-05
  -4.46435051e-10 -5.61071508e-04 -1.43658652e-03  6.69856646e-04
  -5.90234253e-03]
 [ 1.15913759e-03  1.71042208e-03  2.48605370e-35 -1.51825994e-04
  -7.21245209e-10 -7.37418004e-05  5.41631984e-04  6.82642998e-04
   7.83677273e-03]
 [ 3.66775261e-35  2.48605370e-35  1.00000000e+00  3.91015153e-38
   1.85676878e-43  2.39575682e-35 -8.32480184e-35 -1.96441075e-35
  -7.09297138e-34]
 [-9.39730309e-05 -1.51825994e-04  3.91015153e-38  1.00004159e+00
   1.97575942e-10  4.34068302e-05 -1.66566778e-04 -5.81462005e-06
  -1.71493263e-03]
 [-4.46435051e-10 -7.21245209e-10  1.85676878e-43  1.97575942e-10
   1.00000000e+00  2.06204086e-10 -7.91306922e-10 -2.76238957e-11
  -8.14705227e-09]
 [-5.61071508e-04 -7.37418004e-05  2.39575682e-35  4.34068302e-05
   2.06204086e-10  3.53098554e-04  1.04659038e-03 -1.84624270e-04
   6.13646602e-03]
 [-1.43658652e-03  5.41631984e-04 -8.32480184e-35 -1.66566778e-04
  -7.91306922e-10  1.04659038e-03  2.13775803e-02  2.54707571e-03
   1.73216657e-01]
 [ 6.69856646e-04  6.82642998e-04 -1.96441075e-35 -5.81462005e-06
  -2.76238957e-11 -1.84624270e-04  2.54707571e-03  1.15178607e-02
   6.81445348e-02]
 [-5.90234253e-03  7.83677273e-03 -7.09297138e-34 -1.71493263e-03
  -8.14705227e-09  6.13646602e-03  1.73216657e-01  6.81445348e-02
   1.67529268e+00]]
Optim status  :True
Optim message :Optimization terminated successfully.
running time for optimisation using simData is True :13242.789096132 seconds
output_folder in gpGaussLikeFuns is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_500/seed203/
Out-of-sample RMSE for seed203 is :30.206927021521754
Out of sample average width of the prediction variance for seed 203 is 9.050131621829188
Out of sample prediction accuracy is 16.9%
In-sample RMSE for seed203 is :1.527617678580572e-07
In-sample average width of the prediction variance for seed 203 is 0.0014142107973472307
In-sample prediction accuracy is 9.5%

04/01/2019:
For nonFrGammatranformed data, seed 215, numObs 200, numMo 50
if optim using the maximum likelihood of 3 repeats of BFGS , see optim_RndStart1() in bayesMelding.py
theta from optimisation is [ 4.67546444e+00 -6.21594476e+00 -1.18111700e+00  3.21001515e+00
 -3.33872692e-03  8.06445586e+00 -1.92045895e-01  1.61100115e-01
 -2.38249403e+00]
[1.07282382e+02 1.99732846e-03 3.06935699e-01 2.47794616e+01 9.96666840e-01]
output_folder in gpGaussLikeFuns is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_50/seed215/
Out-of-sample RMSE for seed215 is :10.583348962294682
Out of sample average width of the prediction variance for seed 215 is 10.355503412006598
Out of sample prediction accuracy is 95.8%
In-sample RMSE for seed215 is :0.009091857139782281
In-sample average width of the prediction variance for seed 215 is 0.43397970814055
In-sample prediction accuracy is 100.0%

both out-of-sample and in-sample predic accuracy are high


however, 
if optim using LBFGSB + BFGS , see optim_RndStart() in bayesMelding.py
theta from optimisation is [ 4.67657421 -3.62373445 -9.87660928  3.15918724  0.0151728  -1.45988801
 -0.19728132  0.17356637 -2.37786028]
[1.07401506e+02 2.66828443e-02 5.13621374e-05 2.35514464e+01
 1.01528849e+00]
output_folder in gpGaussLikeFuns is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_50/seed215/
Out-of-sample RMSE for seed215 is :10.557537172664071
Out of sample average width of the prediction variance for seed 215 is 10.321073336083975
Out of sample prediction accuracy is 95.7%
In-sample RMSE for seed215 is :9.681086668117495e-08
In-sample average width of the prediction variance for seed 215 is 0.001416054929103816
In-sample prediction accuracy is 8.0%

but for numObs 200, numM0 300
log_cov_parameters plus model bias after optimisation  is :
[ 4.67626715 -6.51904851 -9.87660928  4.54828711 -1.82170138  2.40385685 -0.30699304  0.10963264 -2.19896389]
parameters after optimisation withPrior is  :
[array([1.07368533e+02, 1.47507196e-03, 5.13621374e-05, 9.44704523e+01, 1.61750318e-01]), array([ 2.40385685, -0.30699304,  0.10963264, -2.19896389])]
Out-of-sample RMSE for seed215 is :10.58006444659518
Out of sample average width of the prediction variance for seed 215 is 10.359657086483846
Out of sample prediction accuracy is 95.8%
In-sample RMSE for seed215 is :9.675991594001949e-08
In-sample average width of the prediction variance for seed 215 is 0.0014160777204424596
In-sample prediction accuracy is 8.0%

for Kriging, if 
optim using the maximum likelihood of 3 repeats of BFGS, see optimise2(), 
tarting optimising when withPrior is False& useGradsFlag is True
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[ 1.4268279  -1.44999984 -1.3393216 ]
The 0 round of optimisation
grads at theta from the 0 round of optimisation with BFGS is [ 8.27877940e-07  5.01534863e-06 -1.14138479e-13]
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 744.3199225522806
parameters after optimisation withPrior is False with BFGS :[1.07613848e+02 1.45645712e-01 1.21855099e-06]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[-1.03791445  0.50831444  1.1968809 ]
The 1 round of optimisation
grads at theta from the 1 round of optimisation with BFGS is [-7.36036085e-07  1.81136396e-09 -5.13467924e-06]
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 with BFGS is 751.1231936937489
parameters after optimisation withPrior is False with BFGS :[8.13413901e-01 9.54115021e+04 1.03228533e+01]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[-0.22058733 -0.66789006  0.0121121 ]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[-1.01740837 -0.90899626 -0.55198177]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[-0.34155135 -0.42760053 -0.39627561]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[ 0.48606193 -0.87184021  0.92431956]
The 2 round of optimisation
grads at theta from the 2 round of optimisation with BFGS is [-2.74336969e-07  4.29184715e-11  5.38660382e-06]
BFGS optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 2 with BFGS is 751.1231936928635
parameters after optimisation withPrior is False with BFGS :[8.13411874e-01 6.19842034e+05 1.03228530e+01]
minus_log_like for repeat 3 is [array(744.31992255) array(751.12319369) array(751.12319369)]
parameters after optimisation with BFGS is [1.07613848e+02 1.45645712e-01 1.21855099e-06]
running time for optimisation is 0.5265904840000002 seconds
output_folder in predic_gpRegression is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/kriging/numObs_200/seed215/
Out-of-sample RMSE for seed215 is :9.837247692542043
Out of sample width of the prediction variance for seed 215 is 9.432772320474536
Out of sample prediction accuracy is 95.2%
len of mu_star is 1000000
In-sample RMSE for seed215 is :1.0498392818564716e-07
In sample average width of the prediction variance for seed 215 is 0.0014142143845422572
In sample prediction accuracy is 8.0%

22:18, 04/01/2019 but the in-sample prediction is low, as for all other 19 seeds (both Kriging and data assimilation)
for the FR Gammatranformed data, the in-sample prediction accuracy is 100%.
need to discuss with Ben and Theo next week for these more skewed Gammatranformed data.

optim using the maximum likelihood of 3 repeats of LBFGS + BFGS, see optimise1(), 
starting optimising when withPrior is False& useGradsFlag is True
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[ 1.4268279  -1.44999984 -1.3393216 ]
The 0 round of optimisation
theta from the 0 round of optimisation with LBFGSB is [ 4.44421932 -4.88240807  1.55097588]
grads at theta from the 0 round of optimisation with LBFGSB is [-5.22564960e-08  3.55501391e-10 -2.74056049e-08]
initial theta from LBFGSB optimisation for BFGS is [ 4.44421932 -4.88240807  1.55097588]
theta from the 0 round of optimisation with BFGS is [ 4.44421932 -4.88240807  1.55097588]
grads at theta from the 0 round of optimisation with BFGS is [-5.22564960e-08  3.55501391e-10 -2.74056049e-08]
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 751.4201738490478
parameters after optimisation withPrior is False with BFGS :[8.51333896e+01 7.57874192e-03 4.71607025e+00]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[-1.03791445  0.50831444  1.1968809 ]
The 1 round of optimisation
theta from the 1 round of optimisation with LBFGSB is [ 3.46867268 -4.9967564   2.16061142]
grads at theta from the 1 round of optimisation with LBFGSB is [-9.83332709e-06  1.68449738e-10 -4.61305570e-05]
initial theta from LBFGSB optimisation for BFGS is [ 3.46867268 -4.9967564   2.16061142]
theta from the 1 round of optimisation with BFGS is [ 3.46867263 -4.9967564   2.1606112 ]
grads at theta from the 1 round of optimisation with BFGS is [-3.85822929e-09  1.68450009e-10 -1.89716047e-08]
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 with BFGS is 751.420173849056
parameters after optimisation withPrior is False with BFGS :[3.20941133e+01 6.75983770e-03 8.67643907e+00]
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[-0.22058733 -0.66789006  0.0121121 ]
The 2 round of optimisation
theta from the 2 round of optimisation with LBFGSB is [ 3.53162989 -4.99959947  2.14656421]
grads at theta from the 2 round of optimisation with LBFGSB is [-5.68031048e-06  1.80418930e-10 -2.43274793e-05]
initial theta from LBFGSB optimisation for BFGS is [ 3.53162989 -4.99959947  2.14656421]
theta from the 2 round of optimisation with BFGS is [ 3.53162986 -4.99959947  2.14656409]
grads at theta from the 2 round of optimisation with BFGS is [8.59883045e-08 1.80419090e-10 3.66934131e-07]
BFGS optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 2 with BFGS is 751.4201738489263
parameters after optimisation withPrior is False with BFGS :[3.41796302e+01 6.74064626e-03 8.55541219e+00]
minus_log_like for repeat 3 is [array(751.42017385) array(751.42017385) array(751.42017385)]
parameters after optimisation with BFGS is [8.51333896e+01 7.57874192e-03 4.71607025e+00]
running time for optimisation is 0.39931688 seconds
output_folder in predic_gpRegression is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/kriging/numObs_200/seed215/
Out-of-sample RMSE for seed215 is :10.58283371148508
Out of sample width of the prediction variance for seed 215 is 9.224918107154982
Out of sample prediction accuracy is 93.4%
len of mu_star is 1000000
In-sample RMSE for seed215 is :2.1463945206882005
In sample average width of the prediction variance for seed 215 is 6.314715341343104
In sample prediction accuracy is 100.0%

both the in-sampe predic accuracy and out-of-sample prediction accuracy is high, 

data assimilation
1.07282382e+02 1.99732846e-03 3.06935699e-01 2.47794616e+01 9.96666840e-01
Kriging
8.51333896e+01 7.57874192e-03 4.71607025e+00

again, data assimilation and kriging should have similar estimate of phi1,
and  seems to suggest a better local maximum of likelihood
is key to the results!

31/12/2018
for seed 205
kriging results fo sigma1, phi1, sigma2 1.45715993e+02 1.31570817e-01 4.67248872e-05

data assimilation method results are [1.40798796e+02, 1.35538019e-01, 1.51500068e-10, 1.66754626e-07,
       9.90142245e+01]), array([ 0.94213457, -0.07243488, -0.04460433, -0.64160505]) - also have lower rmse, var and similar prediction accuracy
both models have very low in-sample prediction accuracy

but for seed 219/209, 
where the estimated for sigma1, phi1,sigma2 are differnt between Kriging and datafusion,
kriging has lower Out-of-sample predction accuracy, higer rmse but lower variance and very high in-sample prediction accuracy,
whereas datafusion model has higher Out-of-sample prediction accuracy, lower RMSE but higher variance and very low in-sample prediction accuracy
for seed 209
kriging :[3.44854587e+01 6.73794700e-03 8.62405136e+00]
data assimilation :[array([1.16670249e+02, 1.41655132e-01, 8.37519093e-01, 1.71368947e-06,
       2.03572332e-04]), array([1.06496527, 0.41564109, 0.12341285, 2.88831957])]

after 3 repeats optimisation, optimal estimate of Kriging is [101.14760311   0.14153578   3.19010816],
 which is similar to the above estimates of data assimilation method, and in this case, data assimilation has lower rmse, var 
 and higher prediction accuracy, bother methods have high in-sample prediction accuracy

for seed 219
Kriging:[3.67312979e+01 6.73794700e-03 8.65446710e+00]

data assimilation :[array([1.23472507e+02, 1.39771928e-01, 9.28723954e-06, 2.89077124e-16,
       4.77621831e-01]), array([ 0.89317192, -0.09874212, -0.01474326, -0.90557679])]

after 3 repeats optimisation, optimal estimate of Kriging is [1.10501026e+02 1.42842782e-01 9.00827476e-03],
 which is similar to the above estimates of data assimilation method, and in this case, data assimilation has lower rmse, var 
 and similar prediction accuracy, bother methods have low in-sample prediction accuracy

25/12/2018:
For newly Gammatranformed simulated data (more skewed data), for seed 210,
where kriging and datafusion model have similar estimates for sigma1, phi1,sigma2,
both models have very low in-sample prediction accuracy, for the Out-of-sample prediction,
datafusion model have lower rmse, var and similar prediction accuracy

20/12/2018:
Gammatranformed (Not from fitting of FR data) scale0.2, shape 2.0, without adding error to tildZs
SEED 206:
starting optimising when withPrior is False & gpdtsMo is True& useGradsFlag is True
Starting the 0 round of optimisation in optim_RndStart
initial theta in optim_RndStart :[ 0.43532072 -0.13745337 -1.15993822  1.5304582  -1.22343392  0.
  0.          0.          0.        ]
The 0 repeat of optimisation in optim_RndStart
theta from the 0 repeat of optimisation with LBFGSB is [ 4.6        -1.77269053  1.28106154 -2.3        -2.3         1.11679843
  0.02359504 -0.0727561   0.14973892]
grads at theta from the 0 repeat of optimisation with LBFGSB is [ 4.42186399e+00 -9.59990564e-04  5.79552947e-03 -5.09456417e-03
 -2.17903209e-03  3.72064626e-03 -2.67290901e-03  3.94642116e-03
 -1.85682058e-05]
initial theta from LBFGSB optimisation for BFGS is [ 4.6        -1.77269053  1.28106154 -2.3        -2.3         1.11679843
  0.02359504 -0.0727561   0.14973892]
theta from the 0 repeat of optimisation with BFGS is [ 4.67825492e+00 -1.77690451e+00  1.17548326e+00 -1.40056927e+01
 -7.11328510e+00  1.08655817e+00 -3.30813448e-03 -6.23903972e-02
 -7.49099398e-02]
grads at theta from the 0 repeat of optimisation with BFGS is [ 7.85381367e-06 -1.55344586e-06 -1.67238508e-06 -5.13805325e-10
 -4.37548158e-19  4.93587953e-06 -1.42462097e-06 -3.62057091e-07
  3.03731334e-07]
np.max(np.abs(variance_log_covPars)) in optim_RndStart -firstPart is 658273219.5746543
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 933.227293907447
parameters after optimisation with BFGS :[array([1.07582169e+02, 1.69160974e-01, 3.23970818e+00, 8.26808522e-07,
       8.14215811e-04]), array([ 1.08655817, -0.00330813, -0.0623904 , -0.07490994])]
covariance of pars after optimisation with BFGS :[[ 1.20550936e-02  1.47157305e-03 -9.79460542e-03 -2.15033035e+02
  -8.84926367e+01 -6.68465746e-03 -4.95039224e-03  1.48399578e-03
  -5.22869968e-02]
 [ 1.47157305e-03  1.60834637e-02  1.46938337e-02  8.36649231e+01
   3.44305983e+01  1.07301505e-03  1.18856426e-03 -7.45925953e-04
   2.73339758e-02]
 [-9.79460542e-03  1.46938337e-02  8.43171420e-02  5.65926080e+02
   2.32895561e+02  6.58369217e-03  1.93507979e-02  3.00388470e-03
   2.04712220e-01]
 [-2.15033035e+02  8.36649231e+01  5.65926080e+02  6.58273220e+08
   2.70898671e+08 -8.82558232e+00 -1.11728111e+02 -8.81290485e+01
  -1.51402341e+03]
 [-8.84926367e+01  3.44305983e+01  2.32895561e+02  2.70898671e+08
   1.11482723e+08 -3.63186736e+00 -4.59832253e+01 -3.62687317e+01
  -6.23098251e+02]
 [-6.68465746e-03  1.07301505e-03  6.58369217e-03 -8.82558232e+00
  -3.63186736e+00  1.31077453e-02  1.02692222e-03 -6.30033801e-03
   7.65633022e-03]
 [-4.95039224e-03  1.18856426e-03  1.93507979e-02 -1.11728111e+02
  -4.59832253e+01  1.02692222e-03  1.84245497e-01 -9.15604934e-03
   1.19130440e+00]
 [ 1.48399578e-03 -7.45925953e-04  3.00388470e-03 -8.81290485e+01
  -3.62687317e+01 -6.30033801e-03 -9.15604934e-03  2.02871321e-01
   3.65018538e-01]
 [-5.22869968e-02  2.73339758e-02  2.04712220e-01 -1.51402341e+03
  -6.23098251e+02  7.65633022e-03  1.19130440e+00  3.65018538e-01
   1.00607725e+01]]
Optim status withPrior is False & gpdtsMo is True with BFGS :True
Optim message withPrior with BFGS :Optimization terminated successfully.
minus_log_like for 1 rounds is [933.227293907447]
log_cov_parameters plus model bias after optimisation  is :[ 4.67825492e+00 -1.77690451e+00  1.17548326e+00 -1.40056927e+01
 -7.11328510e+00  1.08655817e+00 -3.30813448e-03 -6.23903972e-02
 -7.49099398e-02]
parameters after optimisation withPrior is  :[array([1.07582169e+02, 1.69160974e-01, 3.23970818e+00, 8.26808522e-07,
       8.14215811e-04]), array([ 1.08655817, -0.00330813, -0.0623904 , -0.07490994])]
Optim status  :True
Optim message :Optimization terminated successfully.
running time for optimisation using simData is True :698.8517680350001 seconds
output_folder in gpGaussLikeFuns is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_50/seed206/
predictons of biggest three are [3.39700574 4.00487903 3.69464658]
Out-of-sample RMSE for seed206 is :9.707825075331414
Out of sample average width of the prediction variance for seed 206 is 9.034991230149267
Out of sample prediction accuracy is 94.4%
In-sample RMSE for seed206 is :1.1800962144120646
In-sample average width of the prediction variance for seed 206 is 4.425422508149714
In-sample prediction accuracy is 100.0%

the variance of sigma2 and phi2 is very very large (sigma2:8.26808522e-07, phi2:8.14215811e-04)

After adding errors to Ztilde
Still the variance of sigma2 and phi2 is very very large (sigma2:4.09990465e-05, phi2:4.79357298e-03)

***but it seems to suggest that, both phi2 is very small,
 in contrast to 8.00494623e+03 in previous experiments where the gamma transformd data still looks like Gaussian with No error added to Ztilde, 
 1.24362603e+02 for seed 200 where Gammatranformed data with error added to Ztilde,
,the GP deltas captures the structure of the noise.


Kriging:
minus_log_like for repeat 0 with LBFGSB is [array(764.34969113)]
parameters after optimisation withPrior is False with LBFGSB :[5.84361220e+01 6.76193010e-03 7.98492977e+00]
covariance of pars after optimisation withPrior is False with LBFGSB :[[ 0.58935028 -0.02652124 -0.26513732]
 [-0.02652124  0.59526959  0.01204947]
 [-0.26513732  0.01204947  0.12840973]]
Optim status withPrior is False with LBFGSB :True
Optim message withPrior is False with LBFGSB :b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
Optim nit withPrior is False with LBFGSB :8
running time for optimisation is 0.17322567899999997 seconds
output_folder in predic_gpRegression is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/kriging/numObs_200/seed206/
Out-of-sample RMSE for seed206 is :10.722152076793995
Out of sample width of the prediction variance for seed 206 is 7.643663254255743
Out of sample prediction accuracy is 87.0%
len of mu_star is 1000000
In-sample RMSE for seed206 is :5.767864615848176
In sample average width of the prediction variance for seed 206 is 9.708240930018855
In sample prediction accuracy is 100.0%

SEED 208:
Output: /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_50/seed208/
(200, 2)
starting optimising when withPrior is False & gpdtsMo is True& useGradsFlag is True
Starting the 0 round of optimisation in optim_RndStart
initial theta in optim_RndStart :[ 1.22352666  1.17164327  0.21739859 -0.49389957 -1.37874848  0.
  0.          0.          0.        ]
The 0 repeat of optimisation in optim_RndStart
theta from the 0 repeat of optimisation with LBFGSB is [ 4.6        -1.82290283  0.85091513 -2.3        -2.3         1.03303059
 -0.18372254 -0.20492289 -1.33573418]
grads at theta from the 0 repeat of optimisation with LBFGSB is [ 1.05662110e+01  6.09643607e-04 -1.77792667e-03 -9.76375990e-03
 -4.89880202e-03  5.56091366e-03 -7.95738502e-04  1.53291948e-03
  2.78595895e-04]
initial theta from LBFGSB optimisation for BFGS is [ 4.6        -1.82290283  0.85091513 -2.3        -2.3         1.03303059
 -0.18372254 -0.20492289 -1.33573418]
theta from the 0 repeat of optimisation with BFGS is [  4.76514094  -1.8244531    0.33905046 -12.24712336  -6.42148033
   0.95732529  -0.2240752   -0.24732513  -2.03777703]
grads at theta from the 0 repeat of optimisation with BFGS is [ 9.65592888e-06 -6.99892448e-06  9.04100831e-06 -1.47054372e-08
 -2.10105895e-13 -1.46480410e-06  2.64591044e-06  1.24142331e-07
  4.01407562e-07]
np.max(np.abs(variance_log_covPars)) in optim_RndStart -firstPart is 22286641.687199317
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 926.0405148868541
parameters after optimisation with BFGS :[array([1.17347656e+02, 1.61305838e-01, 1.40361417e+00, 4.79890226e-06,
       1.62624708e-03]), array([ 0.95732529, -0.2240752 , -0.24732513, -2.03777703])]
covariance of pars after optimisation with BFGS :[[ 1.42251002e-02  1.45863984e-03 -3.89936822e-02 -3.15258598e+01
  -1.31547405e+01 -5.81442135e-03 -8.76836433e-04 -3.83257548e-03
  -3.50912460e-02]
 [ 1.45863984e-03  1.74060125e-02  4.93688432e-02  5.29435927e+01
   2.20922206e+01 -6.76460689e-04  2.24770253e-03 -7.18833585e-03
   4.48717750e-02]
 [-3.89936822e-02  4.93688432e-02  6.18074632e-01  7.49712733e+02
   3.12841632e+02  2.56488124e-02  3.34905711e-02 -1.71836156e-02
   5.93179262e-01]
 [-3.15258598e+01  5.29435927e+01  7.49712733e+02  2.22866417e+07
   9.30049008e+06  1.96691659e+01  1.40406144e+02 -1.84077647e+01
   1.47816902e+03]
 [-1.31547405e+01  2.20922206e+01  3.12841632e+02  9.30049008e+06
   3.88121025e+06  8.20775128e+00  5.85673667e+01 -7.68524941e+00
   6.16623055e+02]
 [-5.81442135e-03 -6.76460689e-04  2.56488124e-02  1.96691659e+01
   8.20775128e+00  9.21538615e-03 -1.70049733e-03  9.33318474e-03
   1.00308572e-02]
 [-8.76836433e-04  2.24770253e-03  3.34905711e-02  1.40406144e+02
   5.85673667e+01 -1.70049733e-03  1.20609692e-01 -9.49200984e-03
   9.10707205e-01]
 [-3.83257548e-03 -7.18833585e-03 -1.71836156e-02 -1.84077647e+01
  -7.68524941e+00  9.33318474e-03 -9.49200984e-03  1.51510205e-01
   1.31115060e-01]
 [-3.50912460e-02  4.48717750e-02  5.93179262e-01  1.47816902e+03
   6.16623055e+02  1.00308572e-02  9.10707205e-01  1.31115060e-01
   8.16913158e+00]]
Optim status withPrior is False & gpdtsMo is True with BFGS :True
Optim message withPrior with BFGS :Optimization terminated successfully.
minus_log_like for 1 rounds is [926.0405148868541]
log_cov_parameters plus model bias after optimisation  is :[  4.76514094  -1.8244531    0.33905046 -12.24712336  -6.42148033
   0.95732529  -0.2240752   -0.24732513  -2.03777703]
parameters after optimisation withPrior is  :[array([1.17347656e+02, 1.61305838e-01, 1.40361417e+00, 4.79890226e-06,
       1.62624708e-03]), array([ 0.95732529, -0.2240752 , -0.24732513, -2.03777703])]
Optim status  :True
Optim message :Optimization terminated successfully.
running time for optimisation using simData is True :917.2944580230001 seconds
output_folder in gpGaussLikeFuns is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_50/seed208/
Out-of-sample RMSE for seed208 is :10.079739248484538
Out of sample average width of the prediction variance for seed 208 is 9.413598563639669
Out of sample prediction accuracy is 94.3%
In-sample RMSE for seed208 is :0.2760065569723059
In-sample average width of the prediction variance for seed 208 is 1.9652119960547976
In-sample prediction accuracy is 100.0%

Output: /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/kriging/numObs_200/seed208/
starting optimising when withPrior is False& useGradsFlag is True
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[1.5802016  1.17164327 0.21739859]
The 0 round of optimisation
theta from the 0 round of optimisation with LBFGSB is [ 3.87962958 -0.59584581  2.08821432]
grads at theta from the 0 round of optimisation with LBFGSB is [-7.67590532e-07 -5.44283232e-07 -2.38371419e-06]
LBFGSB optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with LBFGSB is [array(743.40484882)]
parameters after optimisation withPrior is False with LBFGSB :[48.4062812   0.55109624  8.07049094]
covariance of pars after optimisation withPrior is False with LBFGSB :[[ 0.76008026  0.35626519 -0.12326984]
 [ 0.35626519  0.31622923 -0.02825416]
 [-0.12326984 -0.02825416  0.02885995]]
Optim status withPrior is False with LBFGSB :True
Optim message withPrior is False with LBFGSB :b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
Optim nit withPrior is False with LBFGSB :20
running time for optimisation is 0.18024009499999993 seconds
output_folder in predic_gpRegression is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/kriging/numObs_200/seed208/
Out-of-sample RMSE for seed208 is :11.021426334747641
Out of sample width of the prediction variance for seed 208 is 5.049724067511364
Out of sample prediction accuracy is 66.3%
len of mu_star is 1000000
In-sample RMSE for seed208 is :6.861441987165249
In sample average width of the prediction variance for seed 208 is 9.11824786476756
In sample prediction accuracy is 98.0%

The following are results from data whith error terms added to Ztilde
SEED 206 

starting optimising when withPrior is False & gpdtsMo is True& useGradsFlag is True
Starting the 0 round of optimisation in optim_RndStart
initial theta in optim_RndStart :[ 0.43532072 -0.13745337 -1.15993822  1.5304582  -1.22343392  0.
  0.          0.          0.        ]
The 0 repeat of optimisation in optim_RndStart
theta from the 0 repeat of optimisation with LBFGSB is [ 4.60000000e+00 -1.76519944e+00  1.28563121e+00 -2.30000000e+00
 -2.30000000e+00  1.12293216e+00  2.89873362e-03 -1.94829600e-02
  1.74914901e-01]
grads at theta from the 0 repeat of optimisation with LBFGSB is [ 4.28766240e+00  5.13477831e-04  2.81699829e-05 -5.38964684e-03
 -2.15846031e-03 -1.69506336e-05 -8.58066441e-04 -3.78022075e-04
  1.03607602e-04]
initial theta from LBFGSB optimisation for BFGS is [ 4.60000000e+00 -1.76519944e+00  1.28563121e+00 -2.30000000e+00
 -2.30000000e+00  1.12293216e+00  2.89873362e-03 -1.94829600e-02
  1.74914901e-01]
theta from the 0 repeat of optimisation with BFGS is [ 4.67580692e+00 -1.76906770e+00  1.18558291e+00 -1.01019617e+01
 -5.34047922e+00  1.09374603e+00 -1.97298855e-02 -8.89652473e-03
 -1.29680172e-02]
grads at theta from the 0 repeat of optimisation with BFGS is [-1.24869355e-06  3.18147073e-07  3.14361596e-07 -4.44286349e-08
 -7.30237634e-08 -1.04234740e-06  2.65859634e-07  4.22208212e-08
 -4.51358178e-08]
np.max(np.abs(variance_log_covPars)) in optim_RndStart -firstPart is 2514018.4100109367
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 933.2882873941644
parameters after optimisation with BFGS :[array([1.07319130e+02, 1.70491865e-01, 3.27259388e+00, 4.09990465e-05,
       4.79357298e-03]), array([ 1.09374603, -0.01972989, -0.00889652, -0.01296802])]
covariance of pars after optimisation with BFGS :[[ 1.27873406e-02  1.50536726e-03 -1.02304255e-02 -2.33472140e+01
  -9.11188893e+00 -6.67358597e-03 -3.04006608e-03  2.55602937e-03
  -3.72016247e-02]
 [ 1.50536726e-03  1.59085321e-02  1.40275721e-02  1.09375587e+01
   4.26862896e+00  1.06669203e-03  2.68142539e-04 -1.07056957e-03
   2.10526973e-02]
 [-1.02304255e-02  1.40275721e-02  8.11169826e-02  6.75832107e+01
   2.63759335e+01  6.35259077e-03  1.29107299e-02  1.17677954e-04
   1.56066007e-01]
 [-2.33472140e+01  1.09375587e+01  6.75832107e+01  2.51401841e+06
   9.81137565e+05 -1.13837144e+00 -3.02781486e+00 -5.77759475e+00
  -1.11840676e+02]
 [-9.11188893e+00  4.26862896e+00  2.63759335e+01  9.81137565e+05
   3.82906433e+05 -4.44172849e-01 -1.18738635e+00 -2.25643272e+00
  -4.36975140e+01]
 [-6.67358597e-03  1.06669203e-03  6.35259077e-03 -1.13837144e+00
  -4.44172849e-01  1.31374184e-02  8.11449756e-04 -6.39883308e-03
   5.51164855e-03]
 [-3.04006608e-03  2.68142539e-04  1.29107299e-02 -3.02781486e+00
  -1.18738635e+00  8.11449756e-04  1.81739358e-01 -1.02550812e-02
   1.17028156e+00]
 [ 2.55602937e-03 -1.07056957e-03  1.17677954e-04 -5.77759475e+00
  -2.25643272e+00 -6.39883308e-03 -1.02550812e-02  2.04656331e-01
   3.60084842e-01]
 [-3.72016247e-02  2.10526973e-02  1.56066007e-01 -1.11840676e+02
  -4.36975140e+01  5.51164855e-03  1.17028156e+00  3.60084842e-01
   9.89011355e+00]]
Optim status withPrior is False & gpdtsMo is True with BFGS :True
Optim message withPrior with BFGS :Optimization terminated successfully.
minus_log_like for 1 rounds is [933.2882873941644]
log_cov_parameters plus model bias after optimisation  is :[ 4.67580692e+00 -1.76906770e+00  1.18558291e+00 -1.01019617e+01
 -5.34047922e+00  1.09374603e+00 -1.97298855e-02 -8.89652473e-03
 -1.29680172e-02]
parameters after optimisation withPrior is  :[array([1.07319130e+02, 1.70491865e-01, 3.27259388e+00, 4.09990465e-05,
       4.79357298e-03]), array([ 1.09374603, -0.01972989, -0.00889652, -0.01296802])]
Optim status  :True
Optim message :Optimization terminated successfully.
running time for optimisation using simData is True :634.348529198 seconds
output_folder in gpGaussLikeFuns is /Users/xx249/Documents/trialProject/bayesMelding/dataSimulated/numObs_200_numMo_50/seed206/
Out-of-sample RMSE for seed206 is :9.709527258407487
Out of sample average width of the prediction variance for seed 206 is 9.008391911350433
Out of sample prediction accuracy is 94.3%
In-sample RMSE for seed206 is :1.2052411273740604
In-sample average width of the prediction variance for seed 206 is 4.466765258268247
In-sample prediction accuracy is 100.0%

5/12/2018:
1) SEEDS that have large RMSE in the simulation
[242 295 241 281 254 251 274 234 291 221
 260 245 255 248 209 228 215 259 286 225 275 207 268 246 257 236 200 265
 299 292 203 287 231 239 230 278 227 202 233 219 289 218]

 Normal estimates:
 [array([2.94262407e+01, 5.61608431e-01, 4.87467398e-01, 4.76486806e+00, 1.40044457e-02]), array([ 1.01954253, -0.02605225,  0.00449009, -0.22312738])]
 [array([3.81944926e+01, 5.61959456e-01, 4.50432046e-01, 3.27680967e+01, 1.48992852e-03]), array([ 0.98692284, -0.02385339, -0.02763425, -0.09186235])]

 Optimal parameters are abnormal:
 [array([1.93318618e+01, 1.97869756e-01, 8.69570686e-01, 8.13321296e-09, 2.02597194e-01]), array([ 1.13713522,  0.09103489, -0.01038628,  0.73358252])]
 [array([2.29401565e+01, 1.80136109e-01, 1.41306805e+00, 6.72154123e-08, 1.10546237e+02]), array([1.0409226 , 0.01788827, 0.06974338, 0.54395199])]

2) SEEDS that have large RMSE in the application

By checking the results, it seems that optimal results for all 100 seeds in the application are reasonable, so use all of them.

 50 max values of rmse are [3.65030991 3.66869148 3.67504023 3.70047919 3.78011612 3.80976888
 3.90087148 3.91061631 3.91609185 3.92393232 4.12403251  4.14634242
 4.15233752 4.18295994 4.19162945 4.19827374 4.21831719 4.25487074
 4.29669223 4.29682611 4.29945419 4.37201699 4.37240146 4.37594299
 4.42799959 4.43010932 4.4600751  4.52484843 4.55187423 4.55813797
 4.56528895 4.5994342  4.67316054 4.67486649 4.74154801 4.78638303
 4.89934605 4.95656994 4.96179205 4.97657832 4.97668099 5.00202134
 5.00790737 5.0570417  5.07364271 5.10357971 5.14602031 5.17161481
 5.22184284 5.24080466 5.28921548 5.33044098 5.34572325 5.37173641
 5.39157953 5.41686299 5.45749949 5.51156719 5.62663467 5.64494848]
[167 169 183 124 156 155 201 218 207 210 157 190 165 202 186 120 185 138
 132 204 121 137 173 161 206 212 178 130 135 123 153 174 151 198 184 208
 129 170 180 166 213 214 194 181 26 141 164 127 193 188 215 149 160 128
 133 189 182 200 168 203]

  Normal estimates:
  167 [array([37.12620562,  8.32706039,  4.44500904,  9.67945501,  0.14045742]), array([ 0.04364288, -0.68821898,  1.47508929, -2.1078845 ])]
  169 array([55.6078569 , 10.22937235,  4.41492322, 8.85479839,  0.13909211]), array([-0.07064871, -0.7858722 ,  1.54475719, -2.66416132])]
  183[array([30.74007072,  7.10758789,  4.42594583,  8.61023249,  0.11908599]), array([-0.15908029, -0.67200384,  1.63219521, -1.52225087])]
  210[array([14.54135017,  0.68922341,  3.60083162,  3.21129463,  0.09084219]), array([ 0.87526406, -0.51368898,  0.3275311 , -2.72470168])]
  157[array([14.00976447,  0.72805813,  3.45176376,  3.90285836,  0.11773103]), array([ 0.79721335, -0.51898647,  0.43260509, -2.87042984])]
  190[array([15.36465505,  0.6878561 ,  3.48747198,  3.54452513,  0.09562743]), array([ 0.89745005, -0.56645039,  0.33571974, -3.13154589])]
  202[array([16.14925695,  0.72041871,  3.57717703,  4.34326534,  0.0830863 ]), array([ 0.97757779, -0.54062192,  0.14436814, -3.35308044])]
  186[array([16.78974341,  0.76383075,  3.25148858,  4.3681113 ,  0.08988655]), array([ 0.93358173, -0.59967166,  0.15539064, -3.56503745])]


  ????Optimal parameters are abnormal:
  203[array([16.03321319,  0.76629197,  2.61243804,  3.86729258,  0.08921656]), array([ 0.92809247, -0.42665268,  0.025636  , -2.31247678])]82.1%
  168[array([18.63292218,  0.75869182,  2.85615578,  4.11385584,  0.08361925]), array([ 0.80948642, -0.50906183,  0.39938713, -2.39476625])]82.1%
  200[array([16.85990632,  0.63931125,  2.75805628,  3.28177046,  0.08802036]), array([ 0.88393852, -0.45321095,  0.34558804, -2.33059991])]82.1%
  133/128[array([113.33105366,  14.08305198,3.85846002,   8.75695165, 0.14571603]), array([-0.01948238, -0.68565853,  1.49307711, -2.03546161])] 89.3%
  160/149[array([4.15044888e-11, 5.99836189e-01, 5.29917158e+00, 3.11139218e+00, 8.32624353e-02]), array([-3.98959048e+05, -1.11736923e+00,  1.74509742e+00, -4.84956282e+00])] 100%


 ?182 189 133 
  ?165[array([32.26215753,  1.46890194,  3.3917116 ,  4.22724011,  0.20102224]), array([ 1.06194615, -0.57227857,  0.06437088, -4.23484732])]/50/ 89.3%
  ?138[array([15.94966038,  0.59295485,  3.38667465,  3.21253242,  0.09790144]), array([ 0.8194009 , -0.54758128,  0.5673435 , -2.47805969])] 89.3%
  ?130[array([21.24869849,  0.73161384,  3.0124553 ,  3.97295012,  0.08463072]), array([ 0.79807642, -0.40026495,  0.38717084, -1.84412741])] 96.4%



18/10/2018:
numObs 200, numMo 50 - 300

fount that whem numMo >=200, performance of BM is worse than Kriging - this is reasonable, as model outputs start to dominate the entire sample
see bmVsKrigResOfSimDataMoUpTo300/RMSE_outSample.png for reference

16/10/2018:
1.found that when using simulated data, if model outputs are just average of the simulated observatins, \
the resulting parameters for delats is NOT identifiable.
e.g. SEED209 numObs 200 numMo 50
parameters after optimisation is [array([2.91244637e+01, 7.72950972e-01, 1.00681797e-01, 2.80751592e-10,
       3.04652915e+02]), array([ 0.99933759,  0.00643555,  0.02149056, -0.0164501 ])]

2. So try to generate GP deltas exactly (with the specified variance and length-scale for deltas):
Found the results of BM and kriging are almost the same.

25/10/2018. In this case, the deltas has length-scale 0.1 and variance 0.5 (std 0.7), so average of the 400 blocks of model outputs that \
have more variance than the simulated truth will not help. In comparison, usnkng average of the simuated truth will give more information.

numObs 200, numMo 50 seed 200
Out-of-sample RMSE for seed200 is :0.1166138150746855
Out of sample average width of the prediction variance for seed 200 is 0.11320069762378861
Out of sample prediction accuracy is 94.0%

Kriging:
Out-of-sample RMSE for seed200 is :0.11460497769366891
Out of sample average width of the prediction variance for seed 200 is 0.10677070879538172
Out of sample prediction accuracy is 92.0%

numObs 300, numMo 50 seed 200
Out-of-sample RMSE for seed200 is :0.10984348493290538
Out of sample average width of the prediction variance for seed 200 is 0.10251541290966007
Out of sample prediction accuracy is 92.0%
[array([42.30540126,  0.78626823,  0.09856638,  0.27543994,  0.1983896 ]), array([ 0.42593292,  3.66470091,  2.23561488, 14.54431201])]

Kriging:
Out-of-sample RMSE for seed200 is :0.10984827920145201
OOut of sample average width of the prediction variance for seed 200 is 0.10253566657997089
Out of sample prediction accuracy is 92.0%

numObs 360, numMo 50 seed 200
Out-of-sample RMSE for seed200 is :0.09982558966369087
Out of sample average width of the prediction variance for seed 200 is 0.10319981832663316
Out of sample prediction accuracy is 94.0%
parameters after optimisation is  :
[array([38.3508918 ,  0.78099787,  0.10066872,  0.25795157,  0.19206394]), array([ 0.4431824 ,  3.57547096,  2.07328189, 14.62614526])]

Kriging:
Out-of-sample RMSE for seed200 is :0.09978547678384676
Out of sample average width of the prediction variance for seed 200 is 0.10323363619929243
Out of sample prediction accuracy is 94.0%

15/10/2018:
simdata:
SEED 120:

numObs = 200, numMo = 50

Out-of-sample RMSE for seed200 is :0.10557678952850573
Out of sample average width of the prediction variance for seed 200 is 0.09911049450323191
Out of sample prediction accuracy is 94.0%


Kriging:
Out-of-sample RMSE for seed200 is :0.11460497769366891
Out of sample average width of the prediction variance for seed 200 is 0.10677070879538172
Out of sample prediction accuracy is 92.0%

numObs = 300, numMo = 50
Out-of-sample RMSE for seed200 is :0.1089504558529765
Out of sample average width of the prediction variance for seed 200 is 0.1003739907831207
Out of sample prediction accuracy is 90.0%
parameters after optimisation withPrior is  :
[array([3.40778426e+01, 8.16797309e-01, 1.00362457e-01, 2.17033159e-11,
       7.05312846e+02]), array([ 1.00311663, -0.02028103, -0.00174026,  0.00944114])]

kriging:
Out-of-sample RMSE for seed200 is :0.10984827920145201
Out of sample average width of the prediction variance for seed 200 is 0.10253566657997089
Out of sample prediction accuracy is 92.0%
parameters after optimisation withPrior is False with BFGS :[42.77537333  0.78860606  0.09858084]

**24/08/2018:
1. Remove the cluster of obs between -4 and -2.5, leaving 213 obs
2. Found that for the same seed 120, the results from LBFGSB and BFGS are differnt. Will use LBFGSB to get the intial theta, then 
use BFGS to do global optimisation. However, for seed123, LBFGSB + BFGS and only LBFGSB got the same results.
For SEED124/122: LBFGSB + BFGS resulted in very small parameters 3.44351041e-07 (the corresponding variance is very large).
1/09/2018: For SEED129, results from LBFGSB + BFGS and LBFGSB are differnt,  LBFGSB has larger ML, and gives samller RMSE and avg_predic_width_of_var.
So still choose 3 repeats for LBFGSB and if all three not coverge, the last one use BFGS.
31/08/2018:
num_train = 278
num_test = 50

numMo = 100
seed120:
accuracy = 96.0%
RMSE for seed120 is :5.08329679159973
Average width of the prediction accuracy for seed 120 is 5.204351457460884

122: too large variance

121:
1 LBFGSB + 1 BFGS:
predic_accuracy for seed 121 is 94.0%
RMSE for seed121 is :5.260527330443121
Average width of the prediction accuracy for seed 121 is 5.187400079484956

seed129:
Res from LBFGSB + BFGS
accuracy = 94.0%
RMSE for seed129 is :6.055178072909078
Average width of the prediction accuracy for seed 120 is  5.809905530321202

Res from LBFGSB 
accuracy = 94.0%
RMSE for seed129 is :5.847792984510809
Average width of the prediction accuracy for seed 120 is  5.942831134981902

numMo = 150
120:
accuracy = 96%
RMSE for seed120 is : 5.010377649292999
Average width of the prediction accuracy for seed 120 is 5.178817270938059

seed129:
Res from LBFGSB + BFGS
accuracy = 94.0%
RMSE for seed129 is :6.317882473687837
Average width of the prediction accuracy for seed 129 is  6.061452663010348

Res from LBFGSB 
accuracy = 94.0%
RMSE for seed129 is : :6.3178830532544765
Average width of the prediction accuracy for seed 129 is   6.061458187873125

numMo = 200
120:
accuracy = 96%
RMSE for seed120 is : 4.830116825296874
Average width of the prediction accuracy for seed 120 is 5.128064081125524

129:
Res from LBFGSB + BFGS
accuracy = 96%
RMSE for seed129 is : 6.568241730411553
Average width of the prediction accuracy for seed 129 is 6.105029565486334

Res from LBFGSB 
accuracy = 96%
RMSE for seed129 is : 6.568210665679531
Average width of the prediction accuracy for seed 129 is 6.1050339150298525

numMo = 250
120:
accuracy = 96.0%
RMSE for seed120 is : 4.773184667788076
Average width of the prediction accuracy for seed 120 is  5.090505356867616

using 250 model output that are closest to the observations 2 LBFGSB + 1 BFGS
grads rounded to 2 decimal Point
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 with BFGS is 1632.32254385
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([15.61904589,  0.49889154,  5.10659063,  4.04175314,  0.05618916]), \
array([ 0.86856931, -0.66976673,  0.82413973, 21.03644957])]

RMSE for seed120 is :5.5078117562707485
Average width of the prediction accuracy for seed 120 is 5.463020080061772
prediction accuracy is 98.0%

using 250 model output that are closest to the observations 8 LBFGSB + 1 BFGS
grads rounded to 2 decimal Point
BFGS optmisation converged successfully at the 8 round of optimisation.
minus_log_like for repeat 8 with BFGS is 1681.10346388
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS : too small
[array([1.51717181e+01, 1.39920578e-01, 5.26808469e+00, 3.96765108e-10,
       1.07421888e+02]), array([ 1.02787635, -0.80294327,  0.99867868, 20.42559878])]
variance_log_covPars is [3.18087741e-02 2.64357286e-03 1.84142000e-03 8.71112233e+07
1.31303763e+03] - too large

using 250 model output that are closest to the observations 4 LBFGSB 
grads rounded to 1 decimal Point
LBFGSB optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 3 with LBFGSB is [array(1637.06714812) array(1637.06714722) array(1637.06714744)
 array(1657.46517127)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([19.35196029,  0.1116856 ,  4.97252007,  4.21172308,  0.58253498]), 
array([ 0.68631728, -0.81646736,  1.03485937, 20.77297314])]
prediction accuracy is 100.0%
RMSE for seed120 is :6.179218066757723
Average width of the prediction accuracy for seed 120 is 6.26735315534778

129: LBFGSB
accuracy = 96.0%
RMSE for seed129 is : 5.2919400006290935
Average width of the prediction accuracy for seed 129 is   5.211266356761487

numMo = 300
120:
accuracy = 98%
RMSE for seed120 is :5.127122193139592
Average width of the prediction accuracy for seed 120 is 5.34306264899286

121:
Res from LBFGSB + BFGS
predic_accuracy for seed 121 is 94.0%
RMSE for seed121 is :6.459823268331308
Average width of the prediction accuracy for seed 121 is 6.294742428375992

parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([11.79728061,  0.14719973,  5.5503397 ,  8.65392573,  1.00374793]), arr
ay([ 0.79576361, -1.08817281,  1.24907965, 19.80755088])]

The first parameter is not around 30, may not be ML.

122:
Res from 1 LBFGSB +  1 BFGS
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([2.35968098e+01, 1.58490161e-01, 4.41704495e+00, 1.01374384e-10,
       1.06340268e+02]), array([ 0.88342586, -1.1351499 ,  1.25027176, 19.1996
5826])] - too small
variance_log_covPars is [2.05109082e-02 2.91138645e-03 4.04060242e-03 4.37308359e+08
 4.33294644e+03] - too large

 2 LBFGSB + BFGS correct:
 RMSE for seed122 is :5.253496767109494
 Average width of the prediction accuracy for seed 122 is 5.120813971129835
 predic_accuracy for seed 122 is 94.0%


123:
Res from LBFGSB + BFGS
predic_accuracy for seed 123 is 94.0%
RMSE for seed123 is :5.235479838057615
Average width of the prediction accuracy for seed 123 is 5.0593495404795235

124:
Res from LBFGSB + BFGS
[array([1.63045488e+01, 2.35581362e-01, 5.20174709e+00, 1.95500685e-09,
       1.01004568e+02]), array([ 1.06166344, -1.06698245,  1.06903565, 19.25869797])]
too small
variance_log_covPars is [3.77697314e-02 1.96213906e-03 2.47321317e-03 2.57716112e+07
 3.10951542e+02] - too large

 2 LBFGSB + 1 BFGS:
 abnormal parameters - although converged

125:
Res from LBFGSB + BFGS
predic_accuracy for seed 125 is 96.0%
RMSE for seed125 is :5.102164403792912
Average width of the prediction accuracy for seed 125 is 5.136989615511531
RMSE for seed124 is :6.800660194406422
Average width of the prediction accuracy for seed 124 is 6.70140835920034
predic_accuracy for seed 124 is 96.0%


126:
Res from LBFGSB + BFGS
accuracy = 96%
RMSE for seed126 is : 6.316078497378258
Average width of the prediction accuracy for seed 126 is 6.701408380755561

parameters after optimisation withPrior is False & gpdtsMo is True with BFGS : abnormal
[array([2.50699499e-15, 1.02056012e+00, 6.70140831e+00, 5.17634684e+00,
       1.42475713e-01]), array([-7.03719149e+07, -1.15162927e+00,  1.25256581e+00,  1.94643008e+01])]

2 LBFGSB:
RMSE for seed126 is :4.475465476241665
Average width of the prediction accuracy for seed 126 is 5.169493300984073
prediction accuracy is 96.0%

127:
Res from 1 LBFGSB + 1 BFGS
accuracy = 92%
RMSE for seed127 is : 6.973790072492643
Average width of the prediction accuracy for seed 127 is  6.701409041087535

parameters after optimisation withPrior is False & gpdtsMo is True with BFGS : abnormal
[array([9.24981504e-12, 1.25130980e+00, 6.70140897e+00, 9.75372870e+00,
       1.67262886e-01]), array([-8.70629255e+05, -1.00538449e+00,  1.31292184e+00,  2.11455640e+01])]

2 LBFGSB + 1 BFGS
RMSE for seed127 is :4.367949879637815
Average width of the prediction accuracy for seed 127 is 5.118208677674613
predic_accuracy for seed 127 is 98.0%

128:
Res from LBFGSB + BFGS
accuracy = 96%
RMSE for seed128 is :4.82650016386838
Average width of the prediction accuracy for seed 128 is  5.184932324368166

129:
Res from LBFGSB + BFGS
accuracy = 94%
RMSE for seed129 is :6.670808278913135
Average width of the prediction accuracy for seed 129 is  6.310826935147701

Res from LBFGSB 
accuracy = 94%
RMSE for seed129 is :6.670807753480858
Average width of the prediction accuracy for seed 129 is  6.310843231078585

numMo = 350
120:
Res from LBFGSB 
predic_accuracy for seed 120 is 96.0%
RMSE for seed120 is :4.72863480125839
Average width of the prediction accuracy for seed 120 is 5.080190110105898

129:
Res from LBFGSB 
accuracy = 96%
RMSE for seed129 is :5.2730250037956905
Average width of the prediction accuracy for seed 129 is  5.198238476371481

numMo = 400
120:
Res from LBFGSB 
predic_accuracy for seed 120 is 94.0%
RMSE for seed120 is :4.761377801407543
Average width of the prediction accuracy for seed 120 is 5.071661807885753

129:
Res from LBFGSB 
accuracy = 96%
RMSE for seed129 is :5.226817550991617
Average width of the prediction accuracy for seed 129 is  5.183054568423857

numMo = 450
120:
Res from LBFGSB 
predic_accuracy for seed 120 is 94.0%
RMSE for seed120 is :4.773263495211603
Average width of the prediction accuracy for seed 120 is 5.054576074767008

129:
Res from LBFGSB 
accuracy = 92%
RMSE for seed129 is :6.982923196990028
Average width of the prediction accuracy for seed 129 is 6.456301499120326

LBFGSB optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with LBFGSB is [array(2145.93651657)]

parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([12.53812094,  0.12301151,  5.63686538, 10.23534421,  1.03605386]), \
array([ 0.69802291, -1.1039007 ,  1.30213874, 20.07560684])]


BFGS optmisation converged successfully at the 0 round of optimisation. 1 LBFGSB + 1 BFGS:
minus_log_like for repeat 0 with LBFGSB is [array(2145.93651536)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([12.54404108,  0.12300773,  5.6364263 , 10.23646032,  1.03607523]), 
array([ 0.69785761, -1.10397858,  1.3020664 , 20.07486303])]

RMSE for seed129 is :6.982937910934044
Average width of the prediction accuracy for seed 129 is 6.456262523430862
predic_accuracy for seed 129 is 92.0%

numMo = 500
120:
predic_accuracy for seed 120 is 96.0%
RMSE for seed120 is :4.756648063848882
Average width of the prediction accuracy for seed 120 is 5.065176761851449

128:
LBFGSB optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 with LBFGSB is [array(2248.04962437) array(2249.55542704)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([4.64394685, 1.89069576, 6.43476746, 6.60119899, 0.15378132]), 
array([-4.63724668, -3.34515142,  3.69792263,  9.45503248])]

RMSE for seed128 is :5.8520343357631015
Average width of the prediction accuracy for seed 128 is 6.438769436069318

MO450:
BFGS optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 2 with BFGS is 2055.47519521
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([35.55707092,  1.26008028,  4.96078504,  6.59452084,  0.15478818]), 
array([ 0.72766439, -0.49211886,  0.515009  , 22.2425101 ])]
RMSE for seed128 is :4.84379151062471
Average width of the prediction accuracy for seed 128 is 5.12507615991953

129:
Res from LBFGSB 
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([29.22068181,  1.12987558,  5.0198971 ,  5.95288411,  0.11350961]), \
array([ 0.75046499, -0.55636052,  0.75016139, 22.08222163])]

predic_accuracy for seed 129 is 96.0%
RMSE for seed129 is :5.0631043584112945
Average width of the prediction accuracy for seed 129 is 5.149693231839934

Kriging:
120:
RMSE for seed120 NoMO_is :5.154652629651371
Average width of the prediction accuracy for seed 120 NoMo_is 5.369095553545448
prediction accuracy is 96.0%

121:
RMSE for seed121 NoMO_is :5.298991023942831
Average width of the prediction accuracy for seed 121 NoMo_is 5.310593523909408
predic_accuracy for seed 121 is 94.0%

122:
RMSE for seed122 NoMO_is :5.2920942753900775
Average width of the prediction accuracy for seed 122 NoMo_is 5.3116843429399445
prediction accuracy is 94.0%

123:
RMSE for seed123 NoMO_is :5.835717875976812
Average width of the prediction accuracy for seed 123 NoMo_is 5.315552509213941
prediction accuracy is 94.0%

124:
RMSE for seed124 NoMO_is :4.5301963660966775
Average width of the prediction accuracy for seed 124 NoMo_is 5.322037754937184
prediction accuracy is 100.0%

125:
RMSE for seed125 NoMO_is :5.296032731847557
Average width of the prediction accuracy for seed 125 NoMo_is 5.341157898673979
prediction accuracy is 96.0%

126:
RMSE for seed126 NoMO_is :4.1888666391872
Average width of the prediction accuracy for seed 126 NoMo_is 5.29954603699613
prediction accuracy is 98.0%

127:
RMSE for seed127 NoMO_is :4.866632713458387
Average width of the prediction accuracy for seed 127 NoMo_is 5.32645056031402
prediction accuracy is 98.0%

128:
RMSE for seed128 NoMO_is :4.941279058820454
Average width of the prediction accuracy for seed 128 NoMo_is 5.397110323065192
prediction accuracy is 96.0%

129:
RMSE for seed129 NoMO_is :5.542044532506811
Average width of the prediction accuracy for seed 129 NoMo_is 5.392269724642892
prediction accuracy is 96.0%

seed 125: - rounded two
grads at theta from the 1 round of optimisation with LBFGSB is 
[-0.00053642  0.00029719  0.00406496 -0.00091694  0.00108393 -0.002769 0.00445911  0.00064581 -0.00027871]
LBFGSB optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 with LBFGSB is [array(1290.38886648) array(1290.3888658)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([30.31615896,  1.31719198,  3.97098501,  5.02599627,  0.14930688]), array([ 0.79714119, -0.34174997,  0.5199241 , 23.87090849])]
RMSE  3.8446976309081133
Average width of the prediction accuracy for seed 125 is 4.09318537588566


SEED124: LBFGSB + BFGS resulted in very small parameters  3.44351041e-07 (the corresponding variance is very large 1.30839053e+05)
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 1433.9910738
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([2.36638296e+01, 2.11690796e-01, 3.88266770e+00, 3.44351041e-07, 1.03173900e+02]), \
array([ 0.84335994, -0.98032514,  0.9101429 , 20.07685938])]
variance_log_covPars is [3.55755462e-02 2.34455691e-03 7.06663251e-03 1.30839053e+05 2.02811281e+00]


Output: Output/cntry_FR_numObs_213_numMo_300/SEED_125_withPrior_False_poly_deg_2_repeat1/
starting optimising when withPrior is False & gpdtsMo is True& useGradsFlag is True
initial theta when withPrior is False & gpdtsMo is True& useGradsFlag is True :[ 0.24572071 -1.95051852  0.37911415  1.14632892 -0.14760647  0.
  0.          0.          0.        ]
The 0 round of optimisation
theta from the 0 round of optimisation with LBFGSB is [ 3.41156391  0.27549379  1.37902222  1.61464123 -1.90169791  0.79716229
 -0.34160636  0.51996507 23.87194768]
grads at theta from the 0 round of optimisation with LBFGSB is [ 0.00077001  0.00062458  0.00043854  0.00099853 -0.00829316 -0.00239649
 -0.01500417 -0.0068424   0.00113874]
initial theta from LBFGSB optimisation for BFGS is [ 3.41156391  0.27549379  1.37902222  1.61464123 -1.90169791  0.79716229
 -0.34160636  0.51996507 23.87194768]
theta from the 0 round of optimisation with BFGS is [ 3.41165023  0.27549764  1.37902472  1.61461676 -1.90174897  0.7971406
 -0.34171735  0.51992854 23.87114205]
grads at theta from the 0 round of optimisation with BFGS is [ 3.15025326e-06  1.46738742e-07  2.41854841e-06  4.25267189e-07
 -3.43260709e-07  1.35925055e-06 -1.91464069e-07  1.10822541e-06
  1.77623577e-07]
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 1290.38886572
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :[array([30.31523014,  1.31718599,  3.97102689,  5.02596139,  0.14930726]), array([ 0.7971406 , -0.34171735,  0.51992854, 23.87114205])]

SEED123 -rounded two
theta from the 3 round of optimisation with LBFGSB is [ 3.48299926  0.37561325  1.38016566  1.85845559 -1.99064623  0.72598889
 -0.37485542  0.46182649 23.72705377]
grads at theta from the 3 round of optimisation with LBFGSB is [ 1.45378314e-04 -7.11451540e-04 -4.77806334e-03 -1.52825886e-03
  1.26498964e-03 -1.12506238e-04 -2.01024397e-03 -8.03620377e-05
  5.04818688e-04]
LBFGSB optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 3 with LBFGSB is [array(1330.87013517) array(1330.87013493) array(1330.87013522)
 array(1330.87013493)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([32.55722337,  1.45588397,  3.97556016,  6.41382353,  0.13660712]), array([ 0.72598889, -0.37485542,  0.46182649, 23.72705377])]
RMSE for seed123 is :3.841413685438401
Average width of the prediction accuracy for seed 123 is 4.104936972325392

SEED123: LBFGSB + BFGS
initial theta from LBFGSB optimisation for BFGS is [ 3.482974    0.37557717  1.3801678   1.85841453 -1.99064399  0.72596333
 -0.37483323  0.46181341 23.72730928]
theta from the 0 round of optimisation with BFGS is [ 3.48300116  0.37560291  1.38015196  1.85844536 -1.99064308  0.72598732
 -0.37482927  0.4618359  23.7273109 ]
grads at theta from the 0 round of optimisation with BFGS is [-1.21872290e-10  8.58335625e-11  1.80875759e-10  3.40492079e-10
  1.45604417e-10  6.33207264e-10  3.83750920e-10 -1.64555924e-10
  2.48677345e-09]
BFGS optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with BFGS is 1330.87013484
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([32.55728528,  1.4558689 ,  3.97550568,  6.41375791,  0.13660755]), array([ 0.72598732, -0.37482927,  0.4618359 , 23.72
73109 ])]
RMSE for seed123 is :3.8414033695996364
Average width of the prediction accuracy for seed 123 is 4.104884270775154

Output: Output/cntry_FR_numObs_213/SEED_124_withPrior_False_repeat9/   seed 122 similar
initial theta when withPrior is False& useGradsFlag is True :[ 1.76123258 -0.48949742 -0.1759926 ]
The 0 round of optimisation
theta from the 0 round of optimisation with LBFGSB is [ 3.54108903 -3.87120828 -0.29348163]
True
grads at theta from the 0 round of optimisation with LBFGSB is [-1.78461803e-05 -6.65298927e-06 -6.30370360e-06]
LBFGSB optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with LBFGSB is [array(677.9500916)]
parameters after optimisation withPrior is False with LBFGSB :[3.45044750e+01 2.08331818e-02 7.45662918e-01]
RMSE for seed124 NoMO_is :0.10845284769769711
Average width of the prediction accuracy for seed 124 NoMo_is 1.048850468878812

Bayesian Melding results:

BFGS method for SEED 120:
initial theta from LBFGSB optimisation for BFGS is [ 3.4817621   0.2178648   1.35264164  2.03299331 -2.05694791  0.62774556
 -0.5864336   0.65404961 22.18341285]
theta from the 2 round of optimisation with BFGS is [ 3.48179162  0.21786809  1.35263885  2.03298814 -2.05695774  0.62774633
 -0.58642035  0.65403743 22.1835108 ]
grads at theta from the 2 round of optimisation with BFGS is [-7.08173939e-07  2.35633458e-07  1.34530183e-07 -4.73116387e-07
  1.95414700e-06  3.05092172e-06 -1.13659927e-06 -1.60897162e-06
  1.76423134e-07]
BFGS optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 2 with BFGS is 1357.40734792
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :[array([32.51792973,  1.24342303,  3.86761815,  \
	7.63687237,  0.12784231]), array([ 0.62774633, -0.58642035,  0.65403743, 22.1835108 ])]
RMSE for seed120 is :3.6776189549943963
Average width of the prediction accuracy for seed 120 is 4.048320579914315

LBFGSB method:
The 3 round of optimisation
theta from the 3 round of optimisation with LBFGSB is [ 2.8154011  -2.0683095   1.45337556  1.96970809 -0.31450916  0.65561411
 -1.03318281  1.2761146  20.43609478]
grads at theta from the 3 round of optimisation with LBFGSB is [ 2.26931007e-03  6.08426397e-05  7.46297112e-05  6.87672592e-06
  2.13744074e-04  3.52737197e-03  2.59713415e-04  2.61317457e-03
 -1.33999064e-04]
LBFGSB optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 3 with LBFGSB is [array(1357.40734838) array(1357.40734805) array(1357.40734892)
 array(1424.34678403)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([16.69987269,  0.12639928,  4.27752925,  7.16858359,  0.73014717]), array([ 0.65561411, -1.03318281, \
 1.2761146 , 20.43609478])]
RMSE for seed120 is :3.2812825226696765
Average width of the prediction accuracy for seed 120 is 5.080358519160797

Seed121
LBFGSB optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 2 with LBFGSB is [array(1408.2589097) array(1324.8478507) array(1324.84784945)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :
[array([26.84852017,  1.13711378,  3.86454054,  6.41573246,  0.15569251]), array([ 0.77058273, -0.39478015,  0.6069839 , 23.
79629981])]
RMSE for seed121 is :3.688247741904384
Average width of the prediction accuracy for seed 121 is 4.032904351665013

Seed122
grads at theta from the 4 round of optimisation with LBFGSB is [ 1.48193106e-04  1.54620035e-03 -2.01055554e-03 -1.56625142e-03
 -1.22383009e-04  1.60936333e-03  2.60643058e-03 -6.24607309e-04
 -8.04392556e-05]
LBFGSB optmisation converged successfully at the 4 round of optimisation.
minus_log_like for repeat 4 with LBFGSB is [array(1430.12445244) array(1386.11144537) array(1393.89116386)
 array(1317.97008905) array(1317.97008843)]
parameters after optimisation withPrior is False & gpdtsMo is True with LBFGSB :[array([33.93958079,  1.25460622,  3.83447962,  5.51054076,  0.11786861]), array([ 0.72020754, -0.47902089,  0.59439078, 22.
8162229 ])]
RMSE for seed122 is :3.6855252124410462
Average width of the prediction accuracy for seed 122 is 3.9775787679226


kriging results: - lower RMSE and avg_predic_width_of_var than 328 obs with a cluster between -4 and -2.5

initial theta when withPrior is False& useGradsFlag is True :[ 1.62068405 -0.43317616  2.0034218 ]
The 0 round of optimisation
theta from the 0 round of optimisation with LBFGSB is [3.36206913 0.3423231  1.3786139 ]
True
grads at theta from the 0 round of optimisation with LBFGSB is [-0.0002435   0.00070284 -0.00083815]
LBFGSB optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with LBFGSB is [array(628.16038455)]
parameters after optimisation withPrior is False with LBFGSB :[28.84882127  1.40821521  3.96939581]
RMSE for seed120 NoMO_is :3.7082835579275812
Average width of the prediction accuracy for seed 120 NoMo_is 4.213025805566944

250 obs:
kriging results:
Output: Output/cntry_FR_numObs_328/SEED_120_withPrior_False_repeat9/
starting optimising when withPrior is False& useGradsFlag is True
bouds is ((-5, 5), (-5, 5), (-5, 5))
initial theta when withPrior is False& useGradsFlag is True :[ 1.62068405 -0.43317616  2.0034218 ]
The 0 round of optimisation
theta from the 0 round of optimisation with LBFGSB is [3.11278451 0.10794468 1.64871391]
True
grads at theta from the 0 round of optimisation with LBFGSB is [-0.00030765  0.00017675  0.00056286]
LBFGSB optmisation converged successfully at the 0 round of optimisation.
minus_log_like for repeat 0 with LBFGSB is [array(796.91506534)]
parameters after optimisation withPrior is False with LBFGSB :[22.4835631   1.11398612  5.20028749]
rounded lower_interval is [1.8 0.4 4.4]
RMSE for seed120 NoMO_is :4.867557918595224

Average width of the prediction accuracy for seed 120 NoMo_is 5.5106496521555215

number of estimated parameters within the 95 percent confidence interval with rounding is 239
prediction accuracy is 95.6%

250 obs + 250 closes model output + 50 random model output
Bayesian Melding results:
output: Output/cntry_FR_numObs_328_numMo_300/SEED_120_withPrior_False_poly_deg_2_repeat9/
initial theta from LBFGSB optimisation for BFGS is [ 2.73520918 -2.3         1.68617592  1.74243706 -0.53131599  0.69369724
 -0.91225016  1.21282775 20.55652864]
theta from the 8 round of optimisation with BFGS is [ 2.69379432 -2.36923648  1.69321522  1.80072765 -0.56350002  0.68309639
 -0.91909742  1.22522196 20.53521831]
grads at theta from the 8 round of optimisation with BFGS is [ 1.15306959e-06 -2.07953352e-06  8.51227242e-07  8.38150733e-07
  7.52587908e-07  2.28073577e-06 -3.80802897e-07  1.72664731e-07
 -5.57938561e-07]
BFGS optmisation converged successfully at the 8 round of optimisation.
minus_log_like for repeat 8 with BFGS is 1535.90779305
parameters after optimisation withPrior is False & gpdtsMo is True with BFGS :
[array([14.78767874,  0.09355213,  5.43693355,  6.05405111,  0.56921331]), 
array([ 0.68309639, -0.91909742,  1.22522196, 20.53521831])]
RMSE for seedNone is :6.133576156558632

20/08/2018:
Angus: 112 + 150 
Seed 100
[array([39.95484092,  1.23361032,  3.02033554,  3.57509964,  0.18122187]), \
array([ 0.90508227, -0.23785494,  0.82040973, 24.78415041])]
RMSE for seed100 is :2.873598096182055
Average width of the prediction accuracy for seed 100 is 3.1600312810486844

101:
[array([5.03858541e+01, 1.19992651e+00, 3.04330079e+00, 2.75320075e+00,
       4.62800280e-02]), array([ 0.88633243, -0.22003565,  0.69690993, 24.67265513])]
BFGS optmisation converged successfully at the 8 round of optimisation.
RMSE for seed101 is :2.9181352945367256
Average width of the prediction accuracy for seed 101 is 3.163133108048332

Kriging:
100:
parameters after optimisation withPrior is False with LBFGSB :[41.85455832  1.17115519  2.98210776]
RMSE for seed100 NoMO_is :2.6224429193527192
Average width of the prediction accuracy for seed 100 NoMo_is 3.3011077050717454

101:
[41.85293473  1.17115308  2.98212705]
RMSE for seed101 NoMO_is :2.622453896529297
Average width of the prediction accuracy for seed 101 NoMo_is 3.301126193212792


19/08/2018:
Bayesian Melding:
optimisation grads rounded to ONE decimal point
120:
[array([36.01809711,  1.14682281,  4.89810041,  6.04338036,  0.13695239]), \
array([ 0.67685421, -0.58237566,  0.7435712 , 21.93326337])]
RMSE for seed120 is :4.736518255635389
Average width of the prediction accuracy for seed 120 is 5.053965421862369
predic_accuracy for seed 120 is 94.8%

121:
[array([11.80236066,  0.14717745,  5.55000778,  8.65075191,  1.00314012]), \
array([ 0.79551614, -1.08839333,  1.25034119, 19.80778157])]
RMSE for seed121 is :4.904184232512637
Average width of the prediction accuracy for seed 121 is 6.126577406330891
prediction accuracy is 97.9%

123:
[array([38.55206137,  1.20448212,  4.89540379,  5.50440224,  0.13720187]), array
([ 0.61851954, -0.50865822,  0.61182339, 22.4775828 ])]
RMSE for seed123 is :4.735009625261411
Average width of the prediction accuracy for seed 123 is 5.050226433624218 

124:
[array([22.62651063,  0.15870335,  4.54452325,  9.30247941,  1.1275087 ]), array
([ 0.59816708, -1.07350536,  1.26453475, 19.9826019 ])]
RMSE for seed124 is :3.5564723780750196
Average width of the prediction accuracy for seed 124 is 5.348397189171781
prediction accuracy is 98.8%

optimisation grads rounded to TWO decimal point
120:
[array([36.00344971,  1.14678314,  4.89799175,  6.0438418 ,  0.1369706 ]), \
array([ 0.6768997 , -0.582397  ,  0.7436122 , 21.93301935])]
RMSE for seed120 is :4.7365431691372315
Average width of the prediction accuracy for seed 120 is 5.053849627265954

121:

[array([29.93638562,  1.14533905,  4.99137489,  6.87561171,  0.13578221]), \
array([ 0.7520156 , -0.6755699 ,  0.68654461, 20.79950487])]
RMSE for seed121 is :4.837584173549779
Average width of the prediction accuracy for seed 121 is 5.14012432706439

122:
[array([29.96976277,  1.0904342 ,  4.96700468,  5.64037494,  0.03002058]), \
array([ 0.79587322, -0.68331833,  0.64604353, 20.79892966])]
RMSE for seed122 is :4.833226451914728
Average width of the prediction accuracy for seed 122 is 5.096854355206951


123:

[array([38.54977577,  1.20449081,  4.89541362,  5.50457805,  0.13719894]), \
array([ 0.61851021, -0.5086665 ,  0.61182848, 22.47740801])]
RMSE for seed123 is :4.73501379372112
Average width of the prediction accuracy for seed 123 is 5.050235726692669



124:
[array([22.62651063,  0.15870335,  4.54452325,  9.30247941,  1.1275087 ]), array
([ 0.59816708, -1.07350536,  1.26453475, 19.9826019 ])]
RMSE for seed124 is :3.5564723780750196
Average width of the prediction accuracy for seed 124 is 5.348397189171781
prediction accuracy is 98.8%

[array([27.04958448,  1.15112061,  5.02421157,  7.12345604,  0.12710025]), \
array([ 0.78623568, -0.50095933,  0.6285341 , 21.94607506])]
RMSE for seed124 is :4.881805939381086
Average width of the prediction accuracy for seed 124 is 5.162253084870744

125:
[array([28.72344344,  1.23901608,  4.99802164,  6.5052486 ,  0.10520593]), \
array([ 0.76011396, -0.47298619,  0.60052752, 22.53405716])]
RMSE for seed125 is :4.868444749808087
Average width of the prediction accuracy for seed 125 is 5.123938585454191

126: although conveged using  optimisation with no constrint, 5.45581066e-10 is too small, 
may using constrint optimisation using grads with one decimal point, as all 8 LBFGSB failed with two decimal points
[array([1.28289011e+01, 2.24044856e-01, 5.45344930e+00, 5.45581066e-10,
       1.05539063e+02]), array([ 1.0768204 , -1.08441293,  1.055659  , 19.24739414])]

rouded to one decimal
[array([29.42165133,  1.19081262,  5.0332706 ,  6.01515833,  0.1522037 ]), array
([ 0.7822635 , -0.5826653 ,  0.58781596, 21.37698689])]
grads at theta from the 1 round of optimisation with LBFGSB is 
[-0.0001966   0.00440894 -0.00688863 -0.0023826   0.005855   -0.00492525
 -0.0072051  -0.00868089  0.00157862]
RMSE for seed126 is :4.900754628426498
Average width of the prediction accuracy for seed 126 is 5.1620250060311275

127:
[array([37.00277077,  1.20839627,  4.9046284 ,  9.25293393,  0.16350212]), \
array([ 0.58437351, -0.55724646,  0.74175429, 22.31607511])]
RMSE for seed127 is :4.70872689777154
Average width of the prediction accuracy for seed 127 is 5.092267446058472

128
[array([5.43830407, 1.85350354, 6.37331879, 6.73745554, 0.1700704 ]), \
array([-3.447129  , -2.99025603,  3.21936072, 10.90439713])]
RMSE for seed128 is :6.364341736585207
Average width of the prediction accuracy for seed 128 is 6.382236032862211

grads at theta from the 1 round of optimisation with LBFGSB is [-2.94673079e-04  2.78472402e-04 -4.60936201e-03  2.07404018e-03
 -3.76698170e-03 -7.30348265e-05 -7.78453550e-04 -1.65035195e-04
  1.54061714e-04]

 It can be seen that even with two decimal point, the -4.60936201e-03  2.07404018e-03
 -3.76698170e-03 is not acturall close to zero. It this is case, can use optimisation with grads of higher precision (rounded to three decimal points or more)

21/08/2018: The below results show that using optimisation with grads of higher precision (rounded to three decimal points or more) gives better results.
grads at theta from the 8 round of optimisation with BFGS is [ 2.18815188e-10 -8.75061801e-10 -3.22302185e-10 -2.33819719e-09
 -7.49324158e-10  3.81191967e-10  4.31251923e-10 -6.95129954e-10 -1.05687459e-10]
BFGS optmisation converged successfully at the 8 round of optimisation.
[array([34.95755741,  1.26695965,  4.99380511,  6.49738985,  0.16810528]), array([
 0.71270328, -0.51884263,  0.48486043, 21.92520902])]
RMSE for seed128 is :4.845113660306848
Average width of the prediction accuracy for seed 128 is 5.137685735083754
predic_accuracy for seed 128 is 95.7%

129:
[array([29.0962621 ,  1.24911128,  5.09759288,  6.08434309,  0.13872906]), \
array([ 0.80040911, -0.53790037,  0.68808363, 22.17839962])]
RMSE for seed129 is :4.974431218405604
Average width of the prediction accuracy for seed 129 is 5.217537690566533

17/08/2017:
Kriging 

With mean:
120
(array([12.96363646,  0.87826724,  4.99675794]), array([-0.95937763,  1.00946758, 19.85939128]))
RMSE for seed120 NoMO_is :4.706354334937888
Average width of the prediction accuracy for seed 120 NoMo_is 5.269643677889688
prediction accuracy is 95.1%

121
:(array([12.96395507,  0.87827617,  4.99675583]), array([-0.95938188,  1.00946581, 19.85935588]))
RMSE for seed121 NoMO_is :4.706353272897486
Average width of the prediction accuracy for seed 121 NoMo_is 5.269641523944171
prediction accuracy is 95.1%

122
(array([12.96354036,  0.87828543,  4.99675643]), array([-0.95942189,  1.00944099, 19.85909546]))
RMSE for seed122 NoMO_is :4.706365334351724
Average width of the prediction accuracy for seed 122 NoMo_is 5.269636146138104

123 similar to above

124
(array([29.60635405,  0.05409173,  2.61739392]), array([-0.7497561 ,  0.84091833, 21.02728014]))
RMSE for seed124 NoMO_is :1.1806305262654044
Average width of the prediction accuracy for seed 124 NoMo_is 3.5077900884167375
prediction accuracy is 100.0%

NO mean:
120
[31.59855917  1.29408406  5.03192951]
RMSE for seed120 NoMO_is :4.77379192771778
Average width of the prediction accuracy for seed 120 NoMo_is 5.275607196366954
prediction accuracy is 95.4%

121
[31.59847615  1.29407774  5.03192864]
RMSE for seed121 NoMO_is :4.773789343951913
Average width of the prediction accuracy for seed 121 NoMo_is 5.275607446292347
prediction accuracy is 95.4%

122
[31.59746728  1.29410117  5.03197378]
RMSE for seed122 NoMO_is :4.773808727992981
Average width of the prediction accuracy for seed 122 NoMo_is 5.275646668873927

123 similar to above

124
[31.59844273  1.29408093  5.03193076]
RMSE for seed124 NoMO_is :4.773791160143488
Average width of the prediction accuracy for seed 124 NoMo_is 5.275608887779317
prediction accuracy is 95.4%

125:
RMSE for seed125 NoMO_is :4.773792689230933
Average width of the prediction accuracy for seed 125 NoMo_is 5.275613501662611
parameters after optimisation withPrior is False with LBFGSB :[31.59859317  1.29408517  5.03193584]

06/08/201:
Use LBFGS + BFGS for data of 328 :
Seed124:
fold0: 93.8% converge at the 1 round of optimisation
fold1: 93.8% converge at the first round of optimisation
fold2: 90.6% converge at the first round of optimisation
fold3: 93.8%  converge at the 1 round of optimisation
fold4: 96.9% converge at the first round of optimisation
fold5: 100% converge at the first round of optimisation
fold6: 96.9% converge at the first round of optimisation
fold7: 90.6% converge at the first round of optimisation
fold8: 93.8% converge at the first round of optimisation
fold9: 82.5% converge at the first round of optimisation
average:93.3%
Seed123:
fold0: 93.8%  converge at the 1 round of optimisation
fold1: 93.8% converge at the first round of optimisation
fold2: 100% converge at the first round of optimisation
fold3: 90.6%  converge at the 1 round of optimisation
fold4: 96.9% converge at the first round of optimisation
fold5: 93.8% converge at the first round of optimisation
fold6: 90.6% converge at the first round of optimisation
fold7: 93.8% converge at the first round of optimisation
fold8: 96.9% converge at the first round of optimisation
fold9: 92.5% converge at the first round of optimisation
average:94.3%
Seed122:
fold0: 93.8%  converge at the 1 round of optimisation
fold1: 93.8% converge at the second round of optimisation
fold2: 87.5% converge at the second round of optimisation
fold3: 93.8% converge at the second round of optimisation
fold4: 93.8% converge at the second round of optimisation
fold5: 90.6% converge at the second round of optimisation
fold6: 96.9% converge at the third round of optimisation
fold7: 93.8% converge at the second round of optimisation
fold8: 87.5% converge at the second round of optimisation
fold9: 92.5% converge at the second round of optimisation
average:92.4%
Seed121:
fold0: 93.8%  converge at the 1 round of optimisation
fold1: 100% converge at the second round of optimisation
fold2: 100% converge at the second round of optimisation
fold3: 96.9% converge at the second round of optimisation
fold4: 93.8% converge at the second round of optimisation
fold5: 90.6% converge at the second round of optimisation
fold6: 93.8% converge at the third round of optimisation
fold7: 93.8% converge at the second round of optimisation
fold8: 90.6% converge at the second round of optimisation
fold9: 90.0% converge at the second round of optimisation
average:94.3%
Seed120:
fold0: 90.6%  converge at the 1 round of optimisation
fold1: 96.9% converge at the second round of optimisation
fold2: 93.8% converge at the second round of optimisation
fold3: 90.6% converge at the second round of optimisation
fold4: 100% converge at the second round of optimisation
fold5: 96.9% converge at the second round of optimisation
fold6: 87.5% converge at the third round of optimisation
fold7: 90.6% converge at the second round of optimisation
fold8: 96.9% converge at the second round of optimisation
fold9: 97.5% converge at the second round of optimisation
average:94.1%
31/07/2018, 10/08/2018:
Use LBFGS + BFGS for data of 328 + 300:
seed124:
fold0: 93.8% converge at the first round of optimisation
fold1: 84.4% converge at the first round of optimisation
fold2: 90.6% converge at the first round of optimisation
fold3: 93.8% converge at the first round of optimisation
fold4: 96.9% converge at the first round of optimisation
fold5: 100% converge at the first round of optimisation
fold6: 93.8% converge at the first round of optimisation
fold7: 100% converge at the first round of optimisation
fold8: 96.9% converge at the first round of optimisation
fold9: 90.0% converge at the first round of optimisation
average:94.0%
Seed123:
fold0: 90.6% converge at the first round of optimisation
fold1: 100% converge at the first round of optimisation
fold2: 90.6% converge at the first round of optimisation
fold3: 96.9% converge at the first round of optimisation
fold4: 93.8% converge at the first round of optimisation
fold5: 96.9% converge at the first round of optimisation
fold6: 96.9% converge at the first round of optimisation
fold7: 96.9% converge at the first round of optimisation
fold8: 93.8% converge at the first round of optimisation
fold9: 92.5% converge at the first round of optimisation
average:94.9%
In-sample prediction accuracy:95.4%
Seed122:
fold0: 90.6%  converge at the 1 round of optimisation
([8.65814211, 1.95850797, 6.30067154, 6.53259799, 0.06961047]), array([-3.6233581 , -3.37573411,  3.97690693,  9.93915639])]
fold1: 96.9% converge at the second round of optimisation
[25.33174565,  1.01549441,  5.02863401,  5.43706474,  0.06026189]), array([ 0.84363873, -0.55784564,  0.77609813, 22.08104723])
fold2: 90.6% converge at the second round of optimisation
[24.13597631,  1.01702064,  4.90642173,  5.44816918,  0.06010555]), array([ 0.87043976, -0.58022979,  0.64003841, 21.75071683])]
fold3: 96.9% converge at the second round of optimisation
[23.15921303,  1.03404385,  5.1445295 ,  5.47698092,  0.06003192]), array([ 0.87719614, -0.59201593,  0.71825703, 21.74961645])]
fold4: 100% converge at the second round of optimisation
[4.25904421, 1.68580986, 6.59567592, 6.32739895, 0.06709192]), array([-3.68259692, -2.77755363,  3.09668182, 11.74141575])]
fold5: 96.9% converge at the second round of optimisation
[22.42267465,  1.00729002,  5.1115951 ,  5.360861  ,  0.05914433]), array([ 0.89940981, -0.54348708,  0.82244359, 22.37508329])]
# fold 6, 7, 8, 9 took around 25 hours (nit around 1500) to converge and the variance (sigma1) is nearly zero?also, b bias is negative??
fold6: 96.9% converge at the third round of optimisation 
[1.45832417e-14, 8.25771129e-01, 6.84584986e+00, 4.83880233e+00,5.10478159e-02]), array([-2.81659882e+07, -1.18791582e+00,  1.33894177e+00,  1.94320093e+01])]
fold7: 96.9% converge at the second round of optimisation
[2.80356463e-13, 8.25771099e-01, 6.75992754e+00, 4.83880232e+00,5.10478160e-02]), array([-6.42386908e+06, -1.18791598e+00,  1.33894195e+00,  1.94320084e+01])]
fold8: 100% converge at the second round of optimisation
[6.11059773e-14, 8.25771117e-01, 6.68430470e+00, 4.83880233e+00,5.10478157e-02]), array([-1.37597386e+07, -1.18791589e+00,  1.33894182e+00,  1.94320089e+01])]
fold9: 92.5% converge at the second round of optimisation
[3.36134089e-14, 8.25771118e-01, 6.63851471e+00, 4.83880232e+00, 5.10478156e-02]), array([-1.85522098e+07, -1.18791586e+00,  1.33894180e+00,  1.94320091e+01])]
average:95.8%
Seed121:
fold0: 90.6% converge at the second round of optimisation
fold1: 93.8% converge at the second round of optimisation
fold2: 100% converge at the second round of optimisation
fold3: 100% converge at the second round of optimisation
fold4: 96.9% converge at the second round of optimisation
fold5: 93.8% converge at the second round of optimisation
fold6: 87.5% converge at the third round of optimisation
fold7: 96.9% converge at the second round of optimisation
fold8: 87.5% converge at the second round of optimisation
fold9: 100.% converge at the second round of optimisation
average:94.7%
In-sample prediction accuracy:94.8%

Seed120:
fold0: 90.6% converge at the second round of optimisation
fold1: 96.9% converge at the second round of optimisation
fold2: 93.8% converge at the second round of optimisation
fold3: 96.9% converge at the second round of optimisation
fold4: 100% converge at the second round of optimisation
fold5: 100% converge at the second round of optimisation
fold6: 84.4% converge at the third round of optimisation
fold7: 90.6% converge at the second round of optimisation
fold8: 96.9% converge at the second round of optimisation
fold9: 95.% converge at the second round of optimisation
average:94.5%

28/07/2018:
1.adding constraint speed up optimisation, but none of the 12 folds converge, but most of them are high predic_accuracy.
seed121f0:18.8% f1:40.6%, f2:84.4%, f3:3.1%, f4:96.9%, f5/9:100%, f6:93.8%, f7:100%, f8:100%
seed123f0:100% f9:92.5%
2. when maxiter is 100 with bondary, even with 10 repeats, the 10th optimisation still not converge. sees SEED121_repeat10_fold5/9_constraint 
3. increase maximum iteration from 100 to 2000, and check the convergence status - still, one repeat optimisation terminates quickly, and does not converge, nor does the 10 repeats.
4. try to using boudary of the otpmisation package itself.


26/07/2018;
1. Just using HMC to get one accepted sample may not converge as well (see SEED121_repeat1_fold0/3/6/7_hmc); \
need a few samples (converge diagnosis needed, but it takes time) 
2. As below, the results suggested that using one sample of HMC to jump out of the area is NOT a good idea.
3. All the unconverged ones seem to be very exteme (0 or very large), so need to try constrain the parameters to do optimisation. For MCMC, \
it is also important to idendify a good prior to work properly.

Diagonal of hess_inv for fold7/6/3/0:
[2.76662501e-03 8.07014487e-02 5.94169490e-02 1.83443330e-02 1.01801686e-02 7.39527328e-04 1.23753610e-02 1.43325905e-02 9.89407837e-01]
[9.59558160e-03 9.72207264e-02 3.11530888e-02 6.24292754e-03 2.39692263e-05 1.11342177e-03 9.06549646e-03 1.07057471e-02 6.29209553e-01]
[5.44898296e-03 1.82632692e-01 1.26541404e-01 1.12716006e-02 4.01542823e-05 2.50077405e-03 1.65442105e-02 1.17406465e-02 1.10379343e+00]
[7.57060484e-03 3.17019897e-01 4.71831033e-02 2.26596652e-01 6.50883435e-03 8.05020932e-04 7.05745834e-01 1.90228718e+00 3.97518304e+00]
Diagonal of hess_inv for fold8/4 that never accepted one sample and the loop became endeless:
f8:[0.00461493 0.97527763 0.26852633 1.00476187 0.00191943 0.02712132 0.99999607 0.99999993 0.99999993]
f4:[4.18947625e-03 2.45443835e+00 1.23384269e-01 1.84281434e-03 1.80953036e-04 3.82801541e-04 1.34632784e-02 1.45221606e-02 9.66337661e-01]

It seems that the diagonal of hess_inv for fold8/4 is NOT abnormal. Then why starting from this point can never get an accepted sample \
(got stuck in this area)? - need to debug in the future (one reason would be that this point is at the boudary, so wherever it jumps,it \
got rejected.

Fold 8/4:
theta from the 1 round of optimisation is 
f8:[2.92238449e+00 -5.35100430e+01 -1.48618250e+01  1.14445079e+01 1.40313816e+01  3.20932311e+02 -7.00114629e-01 -1.17651633e-01 9.05987588e-02]
f4:[3.59524365 -38.30700665  -8.93684497   2.6777993   -1.09137196 0.20197448  -1.14690893   1.37896257  19.4239165 ]

f7:[3.54838893 -17.8384053  -15.43802459   2.42714738  -0.6641756 0.31606151  -1.21559167   1.36105746  18.54181512]
f6:[3.74090322 -15.84276666  -7.73620834   2.48912205  -1.52642935 0.14268445  -1.18122392   1.41149099  18.92724489]
f3:[3.60856564 -19.57816547 -12.74581765   2.59279517  -1.68821485 0.08483484  -1.0902914    1.38590504  19.91024136]
f0:[3.62600741 -18.64408783  -8.44106476   8.4994144    1.29523676 0.45259374   2.00677489 -19.89206899  27.91897954]

BFGS optmisation converged successfully at the 1 round of optimisation.
f1:[3.73903251  0.197634    1.59887872  1.85283309 -1.94951823  0.68816452 -0.44068524  0.5325475  22.47666516]
f2:[3.72935227  0.18535738  1.56893065  1.85820004 -1.94988585  0.6729309 -0.45317889  0.55495879 22.40808706]

23/07/2018;
1. When adding hmc to otpmisation, if keep sampling unitil choosing a point that hmc accepts, the loop can become endless, that is, \
can never propose a point that can be accepted - problems might come from the optimisation results, the variance is close to zero. - \
need to double check
2. Some accepted point also leads to unconvergence - but looking at the mean obtained from the initial optimisation results, `
the proposed point is not far away from the mean obtained from the initial optimisation - seed 123 fold 0, fold 9; seed 121 fold 4.
3. Only seed121 fold 0 with hmc works - b bias parameter changed a lot, the rest of parameters did not change much.

19/07/2018:
1. Tried to constrain log_phi_ZS > -4.6 (phi_Zs >0.01), still NOT converge. So still break the loop when any of the 10 repeats converge.
2. Or try to use sampling methods in future.

05/07/2018:
1. As shown below, most converged cases are NOT necessarily the maximum likelihood, does it mean we only got a local maximum?
This is the reason why it failed to converge when repeat is 3, break the loop when any of the 10 repeats converge,
thus the issue of convergence occured when optimising is SOLVED!
2. Need to consider using mcmc to jump out of the area where the optimisation got stuck (not converge) - ?


04/07/2018: - one repeat - cntry_FR_numObs_328_numMo_300
seed120
minus_log_like for repeat 1 is [array(865.81732802)]
Fold 0 - NOT converge
predic accuracy 34.4%

log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[ 3.88147546 -6.32084972 -9.02473048  4.13439283  0.60896547  0.30957785
 -0.72513094  0.63545816 10.44001508]
parameters after optimisation withPrior is False & gpdtsMo is True :[array([4.84
957160e+01, 1.79841471e-03, 1.20395249e-04, 6.24516607e+01,
       1.83852840e+00]), array([ 0.30957785, -0.72513094,  0.63545816, 10.44001508])]

minus_log_like for repeat 3 is [array(942.50856451) array(1616.74338088) array(1616.74338088)]
NOT converge
31.2%
log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[  3.35087992 -48.25071456 -11.33301851 -14.93965612   6.55064554
   0.74208048  -1.29780852   1.39484878  18.25380044]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([2.85278249e+01, 1.10912608e-21, 1.19710598e-05, 3.24929982e-07,
       6.99695707e+02]), array([ 0.74208048, -1.29780852,  1.39484878, 18.25380044])]

minus_log_like for repeat 10 is [array(1616.74338088)]
predic_accuracy for seed 120 fold 0 is 96.9%
**??Question again: The conveged case is NOT necessarily the maximum?
theta from the 1 round of optimisation is 
[ 3.61298041  0.02190202  1.48435448  1.76830816 -2.20566999  0.65986847 -0.63575591  0.61871919 21.08647758]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([37.07639105,  1.02214363,  4.41211639,  5.86092922,  0.11017668]), array([ 0.65986847, -0.63575591,  0.61871919, 21.08647758])]

Fold 1 - converge
[array([37.69335369,  0.99358416,  4.34389753,  5.75182034,  0.1084357 ]), array([ 0.64608876, -0.62055488,  0.64438432, 21.21055003])]
predic accuracy 46.9%
predic accuracy after adding the observation noise to the 95% confidence interval: 93.8%


Fold 2 - NOT converge
minus_log_like for repeat 1 is [array(897.3403511)]
predic accuracy 28.1%

log_cov_parameters plus model bias after optimisation withPrior is False & gpdts
Mo is True :[ 3.4484941  -4.42826045 -8.42833154  2.86481564 -0.30794496  0.35582694 -1.29950952  1.33824848 18.87236672]
parameters after optimisation withPrior is False & gpdtsMo is True :[array([3.14
529916e+01, 1.19352335e-02, 2.18585888e-04, 1.75458182e+01,7.34955770e-01]), array([ 0.35582694, -1.29950952,  1.33824848, 18.87236672])]

minus_log_like for repeat 3 is [array(1621.61141516) array(1620.59793244) array(1614.49167274)]
predic_accuracy for seed 120 fold 2 is 96.9%

log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[ 3.56874574  0.02540528  1.48172905  1.75644367 -2.21623226  0.68003037 -0.60973037  0.61882064 21.2583068 ]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([35.4720739 ,  1.02573074,  4.40054789,  5.79180313,  0.10901909]), array([ 0.68003037, -0.60973037,  0.61882064, 21.2583068 ])]

Fold 3 - NOT converge
minus_log_like for repeat 1 is [array(924.97167859)]
predic accuracy 43.8%

minus_log_like for repeat 3 is [array(949.3176184) array(1618.20073978) array(1618.20073978)]
NOT converge
predic_accuracy for seed 120 fold 3 is 43.8%

minus_log_like for repeat 4 is [array(1618.20073978) array(960.94731871) array(1618.20073978) array(859.52524194)]
NOT converge

minus_log_like for repeat 10 is [array(1618.20073978)]
predic_accuracy for seed 120 fold 3 is 93.8%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 4 - converge
[array([38.76445629,  1.01895466,  4.44314145,  5.70046452,  0.10772075]), array([ 0.65006305, -0.62284719,  0.65081792, 21.19688783])]
predic accuracy 59.4%
predic accuracy after adding the observation noise to the 95% confidence interval: 96.9%

Fold 5 - converge
minus_log_like for repeat 1 is [array(1703.77507347)]
[array([2.48772420e+01, 1.89950706e-01, 3.92078096e+00, 1.04830008e-10,2.20714949e+28]), array([ 0.81746473, -1.15428024,  1.00671477, 18.19355799])]
NO predic accuracy (84.4%)
predic accuracy after adding the observation noise to the 95% confidence interval: 100%

**??Question again: The conveged case is NOT necessarily the maximum?

minus_log_like for repeat 3 is [array(1616.32497179) array(1703.77507347) array(1428.60825013)]
NOT converge


Fold 6 - NOT converge
minus_log_like for repeat 1 is [array(857.33077324)]
[array([3.07551361e+01, 9.00356988e-04, 2.93083050e-19, 1.07758184e+01,6.31147401e-01]), array([ 0.35106938, -1.24002488,  1.39769655, 19.03620755])]
predic accuracy 40.6%

minus_log_like for repeat 3 is [array(1609.58712598) array(857.63003636) array(1609.58712598)]
NOT converge
predic_accuracy for seed 120 fold 6 is 40.6%

minus_log_like for repeat 4 is [array(896.09565203) array(866.07104482) array(1609.58712598) array(1616.63779873)]
NOT converge

BFGS optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 10 is [array(848.20025415) array(1062.5448149) array(1609.58712598)]
prediction accuracy is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?


minus_log_like for repeat 2 is [array(1609.58712598) array(1609.58712598)]
predic_accuracy for seed 120 fold 6 is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 7 - converge
[array([3.73096823e+01, 9.08592123e-01, 4.15188365e+00, 5.12147514e+00,1.34896107e-07]), array([ 0.65556638, -0.56261018,  0.67924985, 21.60846573])]
predic accuracy 43.8%
predic accuracy after adding the observation noise to the 95% confidence interval: 93.8%

Fold 8 - converge
[array([39.75199266,  1.09314288,  4.34096967,  5.88629643,  0.11006568]), array([ 0.65833243, -0.60125042,  0.62913625, 21.30532542])]
predic accuracy 50.0%
predic accuracy after adding the observation noise to the 95% confidence interval: 87.5%

Fold 9 - converge
[array([3.63515671, -0.0807748, 1.49716868,1.71603756, -2.24673582, 0.62497301,-0.6809207,4 0.66074515, 20.91009502])]
predic accuracy 57.5%
predic accuracy after adding the observation noise to the 95% confidence interval: 97.5%

avg_predic_accuracy_converge = 95.2%

seed121
minus_log_like for repeat 1 is [array(999.69980549)]
Fold 0 - NOT converge
predic accuracy 46.9%
log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[  3.55833479  -7.89334304  -9.45386436 -10.14888049  53.15352224
   0.68232716  -1.08891167   1.46849867  20.03434694]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([3.51046919e+01, 3.73219797e-04, 7.83860669e-05, 3.91198522e-05,
       1.21417533e+23]), array([ 0.68232716, -1.08891167,  1.46849867, 20.03434694])]

minus_log_like for repeat 3 is [array(1772.99731643) array(1740.41850064) array(914.3773862)]
NOT converge
log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[ 3.65389472 -3.32231814 -8.56055704  2.42292605 -1.16413637  0.22447513
 -1.05740413  1.42049366 20.11273713]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([3.86248063e+01, 3.60691214e-02, 1.91512583e-04, 1.12788134e+01,
       3.12192163e-01]), array([ 0.22447513, -1.05740413,  1.42049366, 20.11273713])]
@%%In this case, phi_Zs>0.01, but still NOT converge???

BFGS optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 10 is [array(1061.36958794) array(1632.66399496)]
prediction accuracy is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 1 - NOT converge
minus_log_like for repeat 1 is [array(1794.79658212)]
predic accuracy 100%

log_cov_parameters plus model bias after optimisation withPrior is False & gpdts
Mo is True :[  4.27158925 -16.47156251  -1.84500763   5.79516603  -1.27467853
   0.89627737  -1.47664558  -2.26134914  18.31449711]
parameters after optimisation withPrior is False & gpdtsMo is True :[array([7.16
353921e+01, 7.02249262e-08, 1.58024115e-01, 3.28706753e+02,
       2.79520812e-01]), array([ 0.89627737, -1.47664558, -2.26134914, 18.31449711])]
$$??Question: In this case, the variance of deltas seems to be large -3.28706753e+02, need to constraint this parameter as well?
Also, may consider adding priors and using mcmc methods to sample rather than doing optimisation (have to constrain the parameters \
	and run multple repeats of optimisation to ensure converge)? 

20/07/2018: - adding penalty works fine in  this case.
theta from the 1 round of optimisation is
 [ 3.73903251  0.19763402  1.59887872  1.85283309 -1.94951824  0.68816453 -0.44068524  0.53254746 22.47666522]
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 is [array(1642.86255048)]

minus_log_like for repeat 3 is [array(1733.23145765) array(1691.86896735) array(1642.86255048)]
predic_accuracy for seed 121 fold 1 is 100.0%
predic_accuracy for seed 121 fold 1 is 100.0%

Fold 2 - NOT converge
minus_log_like for repeat 1 is [array(974.71522654)]
predic accuracy 46.9%
19/07/2018:
log_cov_parameters plus model bias after optimisation withPrior is False & gpdts
Mo is True :[  3.75454947  -8.53116288 -10.99776413  -0.16739359  -1.22802799 -0.51730355  -1.00311004   1.44833606  20.65766103]
parameters after optimisation withPrior is False & gpdtsMo is True :[array([4.27149712e+01, 1.97225488e-04, 1.67390854e-05, 8.45866622e-01,\
	2.92869550e-01]), array([-0.51730355, -1.00311004,  1.44833606, 20.65766103])]
running time for optimisation with fixMb is False2175.83633494 seconds

***All parameters seem NOT to be extreme, it is difficult to constraint the parameters for optimisation; still have to use multiple repeats \
to obtian one that converges -  combine constraint of  the sigma parameters and length-scale parameters \
and multiple repeats would be a better choice.

20/07/2018: Adding penalty not work in this case and doubles the running time.
theta from the 1 round of optimisation is [  3.19773727  -4.59998597  -4.79001027   7.25474306  -3.00172399
29.87994253   7.12725011 -13.28490357  37.28699984]
grads at theta from the 1 round of optimisation is [-29.4726846    0.29027684 -92.6585476   -8.11642535   0.10001556
  -8.43657005  -2.45491419   0.94225501   0.20733462]
running time for optimisation with fixMb is False5135.08711505 seconds

minus_log_like for repeat 3 is [array(891.50188763) array(891.41062458) array(961.22739385)]
NOT converge

minus_log_like for repeat 3 is [array(927.37346646) array(890.22594788) array(932.63594552)]
NOT converge

minus_log_like for repeat 10 is [array(1635.22107092)]
predic_accuracy for seed 121 fold 2 is 96.9%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 3 - NOT converge
predic accuracy 43.8%

20/07/2018: adding penalty works in this case 
theta from the 1 round of optimisation is [ 3.72674861  0.12894185  1.5427438   1.82996175 -1.96568744  0.7000486
 -0.37216012  0.51501306 22.8698111 ]
grads at theta from the 1 round of optimisation is [ 6.44375575e-09 -1.63705383e-06 -3.68187864e-06  7.01699804e-07
  1.61378111e-07 -2.98234747e-06 -3.65581874e-06 -1.02996451e-06
  8.57894667e-07]
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 is [array(1631.07675153)]

minus_log_like for repeat 3 is [array(1631.07675153) array(958.49447734) array(1631.07675153)]
NOT converge

minus_log_like for repeat 3 is [array(2577.23895459) array(2667.0597701) array(1014.00103662)]
NOT converge

log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[  3.53708565 -88.22905003 -12.47642794   0.93137217   6.25010701 -0.70577089  -1.14259886   1.44005444  19.59670929]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([3.43666166e+01, 4.81515672e-39, 3.81554158e-06, 2.53798933e+00,
       5.18068263e+02]), array([-0.70577089, -1.14259886,  1.44005444, 19.59670929])]

BFGS optmisation converged successfully at the 5 round of optimisation.
minus_log_like for repeat 10 is [array(917.05850933) array(1303.14063513) array(1770.03950217)array(1250.44553578) array(1631.07675153)]
prediction accuracy is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?

 Fold 4 - NOT converge
minus_log_like for repeat 1 is [array(1568331.22364377)]
predic accuracy 3.1%
19/07/2018:
log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[ 1.11359072e+00 -4.29245185e-01 -3.66907765e-01 -9.29788851e-01 1.22015590e+00  8.55119044e-01  5.86453508e-05 -1.93262614e-05 -5.85557716e-06]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([3.0452735 , 0.65100029, 0.69287355, 0.39463703, 3.38771583]), array([ 8.55119044e-01,  5.86453508e-05, -1.93262614e-05, -5.85557716e-06])]

@%%All parameters seem NOT to be extreme (phi_Zs > 0.01), it is difficult to constraint the parameters for optimisation; \
still have to use multiple repeats \
to obtian one that converges -  combine constraint of  the sigma parameters (??) and length-scale parameters \
and multiple repeats would be a better choice.

20/07/2018:
&&&&&&& - Need to check adding HMC to opimtisation works or not


minus_log_like for repeat 3 is [array(986.7930685) array(1638.48671647) array(1692.96194449)]
NOT converge

minus_log_like for repeat 10 is [array(1638.48671647)]
predic_accuracy for seed 121 fold 4 is 100.0%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 5 - converge
minus_log_like for repeat 1 is [array(1636.48324377)]
predic accuracy 37.5%
predic accuracy after adding the observation noise to the 95% confidence interval: 96.9%
**??Question again: The conveged case is NOT necessarily the maximum?

minus_log_like for repeat 3 is [array(1014.53801302) array(988.66483773) array(973.10377035)]
NOT converge

BFGS optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 10 is [array(2658.99595696) array(1833.85947482) array(1636.48324377)]

Fold 6 - NOT converge
minus_log_like for repeat 1 is [array(1003.68306233)]
predic accuracy 43.8%

minus_log_like for repeat 3 is [array(897.61298697) array(1638.92619709) array(1637.85251281)]
NOT converge

minus_log_like for repeat 10 is [array(1637.85251281)]
predic_accuracy for seed 121 fold 6 is 96.9%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 7 - NOT converge
minus_log_like for repeat 1 is [array(1630.38509596)]
predic accuracy 53.1%

minus_log_like for repeat 3 is [array(959.5761014) array(871.97327806) array(1628.57161002)]
NOT converge

minus_log_like for repeat 10 is [array(1774.62308877)]
predic_accuracy for seed 121 fold 7 is 93.8%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 8 - NOT converge
minus_log_like for repeat 1 is [array(982.91902593)]
predic accuracy 40.6%

minus_log_like for repeat 3 is [array(892.66855088) array(1631.18097371) array(977.36194959)]
NOT converge

BFGS optmisation converged successfully at the 2 round of optimisation.
minus_log_like for repeat 10 is [array(1822.79423304) array(1631.18097371)]
predic_accuracy for seed 121 fold 8 is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 9 - converge
minus_log_like for repeat 1 is [array(1748.54131001)]
prediction accuracy is 92.5%
**??Question again: The conveged case is NOT necessarily the maximum?

minus_log_like for repeat 3 is [array(989.8650724) array(1708.64760399) array(1608.12833471)]
NOT converge

BFGS optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 10 is [array(977.89912527) array(925.78960685) array(1608.12833471)
prediction accuracy is 92.5%
**??Question again: The conveged case is NOT necessarily the maximum?

seed122
Fold 0 - converge
predic accuracy 59.4%

Fold 1 - converge
predic accuracy 59.4%

Fold 2 - NOT converge
minus_log_like for repeat 1 is [array(990.31680184)]
predic accuracy 53.1%

minus_log_like for repeat 3 is [array(992.2662546) array(1651.48634657) array(1657.44002275)]
NOT converge

minus_log_like for repeat 10 is [array(1657.44002274)]
predic_accuracy for seed 122 fold 2 is 90.6%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 3 - NOT converge
minus_log_like for repeat 1 is [array(1877.48040626)]
predic accuracy 31.2%

minus_log_like for repeat 3 is [array(1009.76238047) array(982.10306353) array(932.35980201)]
NOT converge

minus_log_like for repeat 10 is [array(1709.95121814) array(1651.83038772)]
predic_accuracy for seed 122 fold 3 is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 4 - NOT converge
minus_log_like for repeat 1 is [array(989.93426846)]
predic accuracy 50.%

minus_log_like for repeat 3 is [array(1648.45583095) array(913.05334305) array(1648.45583095)]
NOT converge

BFGS optmisation converged successfully at the 3 round of optimisation.
minus_log_like for repeat 10 is [array(1726.06380897) array(9366.05801501) array(1648.45583095)]
predic_accuracy for seed 122 fold 4 is 87.5%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 5 - converge
predic accuracy 50.%

Fold 6 - NOT converge
minus_log_like for repeat 1 is [array(1000.5618123)]
predic accuracy 40.6%

minus_log_like for repeat 3 is [array(1824.60717111) array(972.84937364) array(1664.19840031)]
NOT converge

Fold 7 - NOT converge
minus_log_like for repeat 1 is [array(1720.23997429)]
predic accuracy 0.%

minus_log_like for repeat 3 is [array(1657.4029652) array(1657.4029652) array(1657.4029652)]
predic_accuracy for seed 122 fold 7 is 93.8%

Fold 8 - NOT converge
minus_log_like for repeat 1 is [array(1579.76731708)]
predic accuracy 96.9%

minus_log_like for repeat 10 is [array(1646.94558758)]
predic_accuracy for seed 122 fold 8 is 84.4%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 9 - NOT converge
minus_log_like for repeat 1 is [array(943.11066409)]
predic accuracy 50.%
**??Question again: The conveged case is NOT necessarily the maximum?

minus_log_like for repeat 3 is [array(1635.48806874) array(1698.3891571) array(1682.9886828)]
predic_accuracy for seed 122 fold 9 is 97.5%


seed123
Fold 0 - NOT converge
minus_log_like for repeat 1 is [array(811.35134119)]
predic accuracy 34.4%

log_cov_parameters plus model bias after optimisation withPrior is False & gpdtsMo is True :
[ 3.77204825e+00 -1.63516486e+01 -6.89610069e+00  3.55760539e+00
 -1.91903775e+00 -1.65051207e-02 -3.33577718e-01  1.40095141e+00 2.61872001e+01]
parameters after optimisation withPrior is False & gpdtsMo is True :[array([4.34690093e+01, 7.91715695e-08, 1.01172277e-03, 3.50790957e+01,
       1.46748102e-01]), array([-1.65051207e-02, -3.33577718e-01,  1.40095141e+00,  2.61872001e+01])]

19/07/2018:
Constrain log_phi_ZS >-20 also NOT working and the resulting parameters are:
theta from the 1 round of optimisation is [  3.51486091  -8.83802595  -8.64297698   0.4625737  -11.69782681 0.68037156  \
-0.92361025   1.24693074  21.30845544]

Constrain log_phi_ZS >-4.6 also NOT working and the resulting parameters are:
[ 2.91268929 -0.75979741  1.46873899 -1.8859822   8.62371805  6.83748346-1.84206942  0.17804257  0.27107298]
grads at theta from the 1 round of optimisation is 
[ 1.17540864e+01 -1.19905081e+03  1.44195911e+02  5.38534653e-01 1.32245716e-07  2.66566978e+00 -1.16778119e+01 \
-8.55824422e+00 2.71701276e+00]

20/07/2018:
theta from the 1 round of optimisation is [ 3.45701714 -0.08278985  1.47941139  2.00459343 -1.85242137  0.60677386
 -0.67619085  0.81586148 21.56745481]
grads at theta from the 1 round of optimisation is [-1.87632985e-07  2.41762444e-06  4.02588796e-07  1.21623208e-06
 -1.32678760e-06 -1.58305612e-06 -5.54427583e-07 -1.21465739e-06
  4.89431669e-07]
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 is [array(1622.77841835)]


minus_log_like for repeat 3 is [array(771.25505055) array(1622.77841835) array(1746.78242085)]
NOT converge
minus_log_like for repeat 10 is [array(4981.20381578) array(735.87118565) array(4032.23820245) array(3845.36690646) \
array(2099.8310705) array(744.12802106) array(1622.77841835)]
BFGS optmisation converged successfully at the 7 round of optimisation.
**??Question again: The conveged case is NOT necessarily the maximum?
prediction accuracy is 90.6%

20/07/2018:
theta from the 1 round of optimisation is [ 3.60309275  0.06979422  1.52470448  2.03452821 -1.83411347  0.5997235
 -0.63853782  0.78046066 21.72175841]
grads at theta from the 1 round of optimisation is [ 3.21102813e-06  4.95310593e-05  8.06493751e-05  1.01563756e-04
 -1.39718315e-04  5.79268734e-05  1.97936045e-04  1.65232518e-04
 -4.29145598e-05]
BFGS optmisation converged successfully at the 1 round of optimisation.
minus_log_like for repeat 1 is [array(1606.87137015)]
parameters after optimisation withPrior is False & gpdtsMo is True :
[array([36.71159872,  1.0722875 ,  4.59378582,  7.64864272,  0.15975507]), array([ 0.5997235 , -0.63853782,  0.78046066, 21.72175841])]
running time for optimisation with fixMb is False3405.74405098 seconds
predic_accuracy for seed 123 fold 9 is 97.5%

Fold 1 - converge
predic accuracy 62.5%

Fold 2 - NOT converge
minus_log_like for repeat 1 is [array(4773.84016708)]
predic accuracy 100.%

minus_log_like for repeat 3 is [array(1618.79600427) array(1618.79600427) array(1618.79600427)]
predic_accuracy for seed 123 fold 2 is 84.4%

Fold 3 -  converge
predic accuracy 62.5%

Fold 4 - converge
predic accuracy 40.6%

Fold 5 - NOT converge
minus_log_like for repeat 1 is [array(3433.4765823)]
predic accuracy 90.6%

minus_log_like for repeat 3 is [array(1635.18173898) array(1759.49336909) array(834.95080483)]
NOT converge

minus_log_like for repeat 10 is [array(1635.18173898)]
predic_accuracy for seed 123 fold 5 is 100.0%
**??Question again: The conveged case is NOT necessarily the maximum?

Fold 6 - converge
predic accuracy 43.8%

Fold 7 - converge
predic accuracy 53.1%

Fold 8 - converge
predic accuracy 56.2%

Fold 9 - NOT converge
minus_log_like for repeat 1 is [array(840.87209238)]
predic accuracy 30%
19/07/2018:
log_cov_parameters plus model bias after optimisation withPrior is False & gpdts
Mo is True :[ 3.90279454 -5.70535425 -8.92673315  2.43683789 -3.39869274 -0.23887934
 -0.97156847  1.29586014 20.63961043]
parameters after optimisation withPrior is False & gpdtsMo is True :[array([4.95
406992e+01, 3.32809821e-03, 1.32791125e-04, 1.14368190e+01,
       3.34169259e-02]), array([-0.23887934, -0.97156847,  1.29586014, 20.63961043])]

***All parameters seem NOT to be extreme, log_phi_ZS is only -5.7 (constrain log_phi_ZS >-20 NOT working as well),\
 do we need to constrain phi_Zs be to greater than 0.01???

 Constrain log_phi_ZS >-4.6 (phi_Zs >0.01) also NOT working and the resulting parameters are:

theta from the 1 round of optimisation is [ 4.26959357e+00 -2.41585916e+00 -6.16234943e+00  5.75438746e-01 \
1.12057487e+00  1.62742180e+02 -4.58713494e-01 -8.87520685e-02 5.63929670e-02]
grads at theta from the 1 round of optimisation is 
[-1.59662396e+02 -2.92236476e+01 -8.73333576e+01 -2.50434290e-05 -7.49709115e-06 -1.83885487e+00  1.99885114e-02  
5.43315919e-02 -1.07700527e-02]

minus_log_like for repeat 3 is [array(774.30524293) array(3162.33753724) array(2328.66291362)]
NOT converge

BFGS optmisation converged successfully at the 7 round of optimisation.
minus_log_like for repeat 10 is [array(852.27208606) array(1067.14090717) array(1221.76555981)
 array(992.45779128) array(765.53720938) array(1674.98196073) array(1606.87137015)]
 prediction accuracy is 97.5%
 **??Question again: The conveged case is NOT necessarily the maximum?


seed124
Fold 0 - NOT converge
predic accuracy 87.5%
minus_log_like for repeat 1 is [array(2105.06748645)]

repeat3 - 56.2%
minus_log_like for repeat 3 is [array(1655.53517338) array(1907.60696297) array(1658.68460676)]
predic accuracy after adding the observation noise to the 95% confidence interval: 100%

Fold 1 - converge
[array([37.69335369,  0.99358416,  4.34389753,  5.75182034,  0.1084357 ]), array([ 0.64608876, -0.62055488,  0.64438432, 21.21055003])]
predic accuracy 56.2%
predic accuracy after adding the observation noise to the 95% confidence interval: 96.9%

Fold 2 - converge
predic accuracy 53.1%
predic accuracy after adding the observation noise to the 95% confidence interval: 93.8%

Fold 3 - converge
predic accuracy 43.8%
predic accuracy after adding the observation noise to the 95% confidence interval: 93.8%

Fold 4 - converge
[array([38.76445629,  1.01895466,  4.44314145,  5.70046452,  0.10772075]), array([ 0.65006305, -0.62284719,  0.65081792, 21.19688783])]
predic accuracy 87.5%
predic accuracy after adding the observation noise to the 95% confidence interval: 87.5%

Fold 5 - converge
[array([4.18939588e+01, 1.48353139e-02, 1.11638943e-01, 2.00457086e-07, 5.21392988e+14]), array([-0.61244081, -0.98397134,  1.35384153, 20.66346819])]
predic accuracy 90.6%
predic accuracy after adding the observation noise to the 95% confidence interval: 90.6%

Fold 6 - NOT converge
minus_log_like for repeat 1 is [array(1640.87766484)]
[array([3.07551361e+01, 9.00356988e-04, 2.93083050e-19, 1.07758184e+01,6.31147401e-01]), array([ 0.35106938, -1.24002488,  1.39769655, 19.03620755])]
predic accuracy 25.%

minus_log_like for repeat 3 is [array(1637.99745821) array(1721.59296881) array(1341.98188187)]
predic accuracy 93.8%

Fold 7 - converge
[array([32.17379279,  1.06763943,  4.70559787,  6.49387318,  0.09567221]), array([ 0.72955735, -0.51672845,  0.65900117, 21.81316826])]
predic accuracy 59.4%
predic accuracy after adding the observation noise to the 95% confidence interval: 90.6%

Fold 8 - converge
[array([3.03280421e+01, 1.03353339e+00, 4.73032991e+00, 5.93397668e+00, 5.54167065e-03]), array([ 0.73615605, -0.52766232,  0.67837783, 21.70900112])]
predic accuracy 34.4%
predic accuracy after adding the observation noise to the 95% confidence interval: 100%

Fold 9 - NOT converge
minus_log_like for repeat 1 is [array(2740.15752081)]
[array([37.90779312,  0.92240139,  4.4690179 ,  5.56244389,  0.10574383]), array([ 0.62497301, -0.68092074,  0.66074515, 20.91009502])]
predic accuracy 82.5%

minus_log_like for repeat 3 is [array(1437.42163736) array(1611.71247499) array(1611.15417582)]
predic accuracy 97.5%
predic accuracy after adding the observation noise to the 95% confidence interval: 97.5%


28/06/2018:
#****Use results from linear regression for optimisation - found this doed not speed up optimisation. 
#****The computational cost hinges on the cholesky decompostion.
cntry_FR_numObs_200_numMo_150
SEED108 
2 out of 3 give more reasonable results - 2131s - converge success
model bias from linear regression is :[ 0.4290033  -0.76380239  0.84255376 10.85012163]
[array([4.79287700e+01, 3.43872808e-02, 1.15437416e-01, 1.18733679e+01,1.04155494e+00]), array([ 0.35442668, -1.17811248,  1.29122383, 19.49767567])]
rounded upper_interval is [60.5  0.1  0.1 20.9  1.4  0.4 -0.6  1.9 23.8]
rounded lower_interval is [38.   0.   0.1  6.7  0.8  0.3 -1.7  0.7 15.2]

SEED100 - NOT converge - 2606s
[array([3.48115807e+01, 4.89597034e-05, 1.07422207e-04, 6.78041619e+00,1.21924169e+00]), array([ 0.47740582, -1.26054497,  1.33078869, 18.78128498])]

SEED101 - NOT converge - 2909s
[array([2.93906999e+01, 2.43080477e-08, 6.86041609e-05, 1.11968698e+01,9.23902097e-01]), array([ 0.45634516, -1.18342719,  1.39656689, 19.20553643])]

SEED102 - NOT converge - 450s - 1repeat
[array([3.85449266e+01, 1.70335282e-04, 2.81845658e-05, 1.05003088e+01,1.12080437e+00]), array([-0.35845745, -1.08618494,  0.90541661, 19.61453554])]
rounded upper_interval is [47.1  0.   0.  19.8  1.5 -0.3 -0.5  1.6 25.1]
rounded lower_interval is [31.6  0.   0.   5.6  0.8 -0.4 -1.7  0.2 14.1]
3 repeats- NOT converge - 3649s
[array([3.80563373e+01, 7.84569481e-11, 1.88290742e-05, 1.30231127e+01,4.78330463e-01]), array([-0.19346625, -1.15011108,  1.16658481, 19.44219457])]

SEED103 - NOT converge - 2352s
[array([3.78355300e+01, 4.66733492e-04, 3.46808410e-05, 1.56296697e+01,2.53906010e-01]), array([ 4.78338268e-03, -1.44963673e+00,  1.11544603e+00,  1.71599091e+01])]

SEED104 - NOT converge - 3104s
[array([3.40351435e+01, 2.90812515e-10, 2.13295150e-04, 5.58418659e+01,1.90954341e+00]), array([ 0.42541977, -1.44997684, -0.45063593, 17.79583297])]

SEED105 - NOT converge - 3219s
[array([3.48244912e+01, 7.94175925e-03, 4.26290423e-05, 1.39676457e+01,1.31609358e+00]), array([ 0.38735349, -1.09783773,  1.20072834, 20.15797196])]

SEED106 - NOT converge - 2376s
[array([2.71805686e+01, 4.68631024e-02, 4.81981651e-05, 2.05972959e+01,1.48767919e+00]), array([-0.47614531, -0.79774357,  1.34233839, 23.11163827])]

SEED107 - NOT converge - 2674s
[array([3.37524899e+001, 1.85458068e-117, 8.24520568e-019, 8.66309132e+000,7.44201584e-001]), array([-0.46548602, -1.29439991,  1.42191717, 18.50996142])]

SEED109 - NOT converge - 3192s
[array([3.19009696e+01, 7.92772010e-03, 5.00972698e-05, 1.40068362e+01,1.05475862e+00]), array([ 0.42969899, -1.07212554,  1.39272648, 20.31919138])]

cntry_FR_numObs_200_numMo_150
SEED110
3 repeats give more reasonable results of loglikelihood - 25636s - converge success
2.4h per repeat
model bias from linear regression is :[ 0.24692038 -0.9430733   1.00886398 14.3791568 ]
#fixMbFalse-[array([35.03549185,  1.17956234,  4.78677971,  7.25831802,  0.1399206 ]), array([ 0.69824594, -0.58053498,  0.63784201, 21.86078646])]
[array([4.08873780e+01, 2.13301503e-04, 1.02499255e-01, 1.18340220e+01,6.46489696e-01]), array([-0.28370863, -1.14591679,  1.32918537, 19.44761009])]
rounded upper_interval is [49.9  0.   0.1 16.2  0.8 -0.2 -0.8  1.7 22.5]
rounded lower_interval is [33.5  0.   0.1  8.6  0.5 -0.3 -1.5  0.9 16.4]

SEED117
3 repeats give more reasonable results of loglikelihood - 43423s - converge success
4h per repeat
rounded upper_interval is [59.8  0.1  0.1 14.3  0.8 -0.2 -0.9  1.6 21.8]
rounded lower_interval is [40.6  0.   0.1  7.8  0.5 -0.3 -1.5  1.  16.6]

SEED118
3 repeats give more reasonable results of loglikelihood - 37996s - converge success
3.5h per repeat
[array([41.64041964,  0.08007025,  0.08686883, 10.8366624 ,  1.02909697]), array([ 0.35977835, -1.02397683,  1.22423551, 20.42940003])]
rounded upper_interval is [59.8  0.1  0.1 14.3  0.8 -0.2 -0.9  1.6 21.8]
rounded lower_interval is [40.6  0.   0.1  7.8  0.5 -0.3 -1.5  1.  16.6]

27/06/2018:
# remove the mean of obserations: in this case, it seems the variance of epsilon1 is reduced a lot and makes more sense.
cntry_FR_numObs_200_numMo_150
SEED102
2 out of 3 give more reasonable results - 2934s - converge success
[array([44.66525341,  1.21594013,  4.84130967,  5.51201257,  0.16689223]), array([ 0.62227295, -0.49502318,  0.7776202 , 22.18810228])]
rounded upper_interval is [83.9  1.6  5.4  7.6  0.2  0.8 -0.2  1.1 24.2]
rounded lower_interval is [23.8  0.9  4.4  4.   0.1  0.5 -0.8  0.5 20.2]
SEED108
3 repeats give the same results - 4460s- converge success
[array([57.80347138,  1.26618548,  4.66561196,  7.66483637,  0.23428858]), array([ 0.54738545, -0.44705078,  0.77615932, 23.44881084])]
rounded upper_interval is [ 1.18e+02  1.60e+00  5.20e+00  1.05e+01  3.00e-01  7.00e-01 -1.00e-01 1.10e+00  2.57e+01]
rounded lower_interval is [28.3  1.   4.2  5.6  0.2  0.4 -0.8  0.4 21.2]

#The rest is NOT converged - for most of the cases here, three repeats give different minus loglikelihood - need more random initialisations
#or try use linear regression to give the initial values for optimisation.
[array([3.54392567e+01, 1.62457147e-37, 1.89114783e-04, 4.49732210e-01,1.07015735e+00]), array([ 0.63086377, -1.39782281,  1.4525525 , 17.69437632])]#seed109
[array([3.38528482e+01, 1.76141136e-34, 8.44311811e-05, 1.11585228e+01, 2.73445813e-01]), array([-0.37362338,-1.39466161,  1.63946844, 18.02521746])]#seed107
[array([2.69528591e+01, 3.50108161e-03, 2.49785799e-11, 1.80632202e+01,1.43180485e+00]), array([ 0.48034451, -0.98517537,  1.32975196, 21.62642921])]#seed106
[array([3.29080852e+01, 7.52723645e-02, 2.30626261e-07, 1.30570319e+01,1.27770151e+00]), array([ 0.41074579, -1.12281933,  1.1783113 , 20.04734531])]#seed105
[array([3.43217224e+01, 1.10479008e-09, 2.46517638e-05, 1.56672585e+01,1.41317910e+00]), array([ 0.47911104, -1.20400578,  1.28838193, 19.45794966])]#seed100
[array([3.39949741e+01, 3.74421735e-25, 4.02452195e-06, 9.96856859e+00,3.92419036e-03]), array([-0.4079806 , -1.25224652,  1.56928036, 18.75665805])]#seed101
[array([4.13106284e+01, 6.39370665e-05, 1.01948560e-07, 1.43236567e+02,2.58381108e+00]), array([-0.52656543, -1.70650674,  2.61916782,  3.60485402])]#seed103
[array([4.26506189e+01, 7.67465123e-25, 3.72440053e-06, 1.87586756e+00,.53102813e-02]), array([-0.59127261, -1.03964807,  1.27310848, 20.38079052])]#seed104

cntry_FR_numObs_328_numMo_500
SEED110
3 repeats give different results of loglikelihood - 64505s - converge success
6h per repeat
#res without removing the mean-[array([2.99999705e+02, 1.47167585e+00, 4.60307958e+00, 5.62952941e+00, 1.43983120e-01]), array([ 0.62273737, -0.64436076,  0.7511759 ,  6.59485476])]
[array([29.31656422,  0.84633653,  4.48123686,  4.78159359,  0.13306721]), array([ 0.71798773, -0.65766834,  0.72148804, 21.30597663])]
rounded upper_interval is [47.6  1.   4.9  5.8  0.2  0.8 -0.5  0.9 22.6]
rounded lower_interval is [18.1  0.7  4.1  4.   0.1  0.6 -0.9  0.5 20. ]

SEED111
1 repeat  - 11547s - converge success
3h per repeat
#res without removing the mean-[array([3.33777037e+02, 1.78394320e+00, 4.81612525e+00, 7.54948178e+00, 1.42665809e-01]), array([ 0.64703781, -0.55572811,  0.61478058,  6.28554463])]
[array([35.03549185,  1.17956234,  4.78677971,  7.25831802,  0.1399206 ]), array([ 0.69824594, -0.58053498,  0.63784201, 21.86078646])]
rounded upper_interval is [61.9  1.4  5.2  8.5  0.2  0.9 -0.3  0.9 23.4]
rounded lower_interval is [19.8  1.   4.4  6.2  0.1  0.5 -0.8  0.4 20.3]

SEED116
3 repeats give the same results of loglikelihood - 13397s - converge success
1.2h per repeat
#res without removing the mean-NOT converge [array([3.89229001e+02, 2.59156701e-04, 1.25885381e-19, 1.69599544e-02, 2.09641552e+01]), array([ 0.20142198, -1.20647823,  1.35738602, 19.03663243])]
[array([28.73282761,  0.88424882,  4.5200007 ,  5.37790408,  0.15830029]), array([ 0.7154759 , -0.622429  ,  0.70821174, 21.29392815])]
rounded upper_interval is [46.8  1.1  4.9  6.7  0.2  0.9 -0.4  0.9 22.8]
rounded lower_interval is [17.6  0.7  4.1  4.3  0.1  0.6 -0.8  0.5 19.8]

SEED119
3 repeats give different results of loglikelihood - 19666s - converge success
1.8h per repeat
#res without removing the mean-[array([3.51861996e+02, 1.76974109e+00, 4.49911163e+00, 7.04539521e+00,1.41569305e-01]), array([ 0.63582871, -0.64688165,  0.79516124,  5.93787611])]
[array([26.85729823,  0.75209948,  4.46950498,  4.94370272,  0.11733326]), array([ 0.74726538, -0.64081747,  0.84884139, 21.4494333 ])]
rounded upper_interval is [41.2  0.9  4.9  6.3  0.1  0.9 -0.5  1.1 22.7]
rounded lower_interval is [17.5  0.6  4.1  3.9  0.1  0.6 -0.8  0.6 20.2]

25/06/2018:

avg time per optimisation: 19min
cntry_FR_numObs_200_numMo_150
[array([3.22800661e+02, 2.54120345e+00, 4.61385585e+00, 9.47127065e+00, 7.64739233e-02]), array([ 0.54407861, -0.68330605,  0.69709006,  8.3389994 ])]
[array([3.57431561e+02, 2.33156016e+00, 4.53771902e+00, 9.80979891e+00, 2.02848217e-01]), array([ 0.50958769, -0.54728553,  0.8415862 ,  9.79865064])]
[array([4.73691817e+02, 8.61545387e-58, 4.20710652e-07, 7.23817024e-20,9.92861865e+55]), array([-0.18060364, -1.53269715,  1.13921664, 16.50587151])]
[array([3.65251986e+02, 2.01167294e+00, 4.82924322e+00, 8.25444537e+00, 1.77722259e-01]), array([ 0.50050571, -0.605062,  0.73408066,  9.58670849])]
[array([3.63835390e+02, 1.72361448e+00, 3.81552499e+00, 6.26418285e+00, 7.01194210e-02]), array([ 0.52127751, -0.74253944,  0.9498881 ,  8.62597908])]
[array([4.34973119e+02, 2.12622240e+00, 4.72608682e+00, 9.01221696e+00, 2.58946063e-01]), array([ 0.43620853, -0.4754233,  0.80782628, 12.97356217])]
cntry_FR_numObs_328_numMo_500
avg time per optimisation: 8hour
[array([2.99999705e+02, 1.47167585e+00, 4.60307958e+00, 5.62952941e+00, 1.43983120e-01]), array([ 0.62273737, -0.64436076,  0.7511759 ,  6.59485476])]
[array([3.33777037e+02, 1.78394320e+00, 4.81612525e+00, 7.54948178e+00, 1.42665809e-01]), array([ 0.64703781, -0.55572811,  0.61478058,  6.28554463])]
[array([3.36168686e+02, 1.62257744e+00, 4.88172021e+00, 6.16126132e+00, 1.13201430e-01]), array([ 0.62122213, -0.88060062,  0.76168632,  4.59096707])]
[array([3.42770151e+02, 1.94880911e+00, 4.85701380e+00, 6.22950441e+00,1.43984095e-01]), array([ 0.63743035, -0.49979846,  0.65130789,  6.61256977])]
[array([2.89996013e+02, 1.80030938e+00, 4.95650288e+00, 6.78322941e+00,1.53237616e-01]), array([ 0.69770189, -0.55447262,  0.60713532,  4.92601436])]
[array([3.51861996e+02, 1.76974109e+00, 4.49911163e+00, 7.04539521e+00,1.41569305e-01]), array([ 0.63582871, -0.64688165,  0.79516124,  5.93787611])]


SEED100
3 repeats give more reasonable results - 2821s- converge success
16min/repeat
[array([3.22800661e+02, 2.54120345e+00, 4.61385585e+00, 9.47127065e+00, 7.64739233e-02]), array([ 0.54407861, -0.68330605,  0.69709006,  8.3389994 ])]
rounded upper_interval is [ 6.932e+02  3.200e+00  5.100e+00  1.250e+01  2.000e-01  8.000e-01 -3.000e-01  1.100e+00  1.300e+01]
rounded lower_interval is [150.3   2.    4.2   7.2   0.    0.3  -1.    0.3   3.6]

SEED101
3 repeats give more reasonable results - 5616s - converge success
31min/repeat
[array([3.57431561e+02, 2.33156016e+00, 4.53771902e+00, 9.80979891e+00, 2.02848217e-01]), array([ 0.50958769, -0.54728553,  0.8415862 ,  9.79865064])]
rounded upper_interval is [ 7.381e+02  2.800e+00  5.100e+00  1.300e+01  3.000e-01  7.000e-01 -2.000e-01  1.200e+00  1.440e+01]
rounded lower_interval is [ 1.731e+02  1.900e+00  4.100e+00  7.400e+00  1.000e-01  3.000e-01 -9.000e-01  5.000e-01  5.200e+00

SEED102
4 repeats give more reasonable results.
[array([2.96943771e+02, 1.97000844e+00, 2.67814854e+00, 3.31316566e+01, 8.69612697e-03]), array([ 0.99851298,  0.65848213, -0.57199369,  3.88713077])]
rounded upper_interval is [5.059e+02 2.300e+00 3.100e+00 4.430e+01 7.062e+02 1.600e+00 1.400e+00 4.000e-01 1.770e+01]
rounded lower_interval is [ 1.743e+02  1.700e+00  2.300e+00  2.480e+01  0.000e+00  4.000e-01 -1.000e-01 -1.500e+00 -9.900e+00]

SEED103
2 out of 3 give more reasonable results - 2954s - converge success
16min/repeat
[array([6.26052922e+02, 2.29085852e-01, 1.07239869e-01, 3.10595208e+01, 8.93589914e-02]), array([-0.05893172, -1.11351942,  1.3685745 , 21.05127171])]
rounded upper_interval is [ 8.451e+02  2.000e-01  2.000e-01  4.280e+01  1.000e-01 -0.000e+00 -8.000e-01  1.700e+00  2.390e+01]
rounded lower_interval is [ 4.638e+02  2.000e-01  1.000e-01  2.260e+01  1.000e-01 -1.000e-01 -1.400e+00  1.100e+00  1.820e+01]

SEED104
3 repeats give more reasonable results - 2093s - converge success
12min/repeat
[array([3.65251986e+02, 2.01167294e+00, 4.82924322e+00, 8.25444537e+00, 1.77722259e-01]), array([ 0.50050571, -0.605062,  0.73408066,  9.58670849])]
rounded upper_interval is [ 7.004e+02  2.400e+00  5.400e+00  1.100e+01  2.000e-01  7.000e-01 -3.000e-01  1.000e+00  1.380e+01]
rounded lower_interval is [ 1.905e+02  1.700e+00  4.300e+00  6.200e+00  1.000e-01  3.000e-01 -9.000e-01  4.000e-01  5.400e+00]


SEED105
3 repeats give more reasonable results - 2458s - converge success
14min/repeat
[array([3.63835390e+02, 1.72361448e+00, 3.81552499e+00, 6.26418285e+00, 7.01194210e-02]), array([ 0.52127751, -0.74253944,  0.9498881 ,  8.62597908])]
rounded upper_interval is [ 6.45e+02  2.00e+00  4.30e+00  8.20e+00  1.00e-01  6.00e-01 -5.00e-01 1.20e+00  1.17e+01]
rounded lower_interval is [205.2   1.5   3.4   4.8   0.    0.4  -1.    0.7   5.6]


SEED106
3 repeats give more reasonable results - 3919s
22min/repeat
[array([4.20700773e+02, 2.13197233e-01, 2.80073802e-04, 1.82733814e+01,1.80111943e-01]), array([-9.01116296e-03, -1.07679964e+00,  1.38347133e+00,  1.99213503e+01])]
rounded upper_interval is [ 5.156e+02  2.000e-01  0.000e+00  2.290e+01  2.000e-01  0.000e+00 -8.000e-01  1.700e+00  2.240e+01]
rounded lower_interval is [ 3.433e+02  2.000e-01  0.000e+00  1.460e+01  2.000e-01 -0.000e+00 -1.400e+00  1.100e+00  1.750e+01]

SEED107
3 repeats give more reasonable results - 2962s
16min/repeat
[array([4.22233820e+02, 1.54046595e-01, 2.58420813e-04, 1.18390935e+01, 1.46695877e-01]), array([-0.09184533, -1.3381777,  1.67776233, 18.82555493])]
rounded upper_interval is [ 5.076e+02  2.000e-01  0.000e+00  1.620e+01  2.000e-01 -1.000e-01 -1.100e+00  1.900e+00  2.090e+01]
rounded lower_interval is [ 3.513e+02  1.000e-01  0.000e+00  8.600e+00  1.000e-01 -1.000e-01 -1.600e+00  1.400e+00  1.670e+01]

SEED108
3 repeats give more reasonable results - 4482s - converge success
25min/repeat
[array([4.34973119e+02, 2.12622240e+00, 4.72608682e+00, 9.01221696e+00, 2.58946063e-01]), array([ 0.43620853, -0.4754233,  0.80782628, 12.97356217])]
rounded upper_interval is [ 8.801e+02  2.600e+00  5.300e+00  1.200e+01  3.000e-01  6.000e-01 -1.000e-01  1.100e+00  1.620e+01]
rounded lower_interval is [ 2.15e+02  1.80e+00  4.20e+00  6.80e+00  2.00e-01  3.00e-01 -8.00e-01 5.00e-01  9.70e+00]

SEED109
3 repeats give more reasonable results - 2618s
15min/repeat
[array([4.77271872e+02, 3.52305393e-14, 1.14952271e-09, 1.60891697e+01, 3.56597623e-01]), array([-0.06385502, -1.33134528,  1.39924266, 18.06101233])]
rounded upper_interval is [ 5.794e+02  0.000e+00  0.000e+00  2.280e+01  4.000e-01 -0.000e+00 -1.000e+00  1.900e+00  2.100e+01]
rounded lower_interval is [ 3.931e+02  0.000e+00  0.000e+00  1.140e+01  3.000e-01 -1.000e-01 -1.700e+00  9.000e-01  1.520e+01]

SEED110
3 repeats give same esults - 58808s - converge success
16hour/repeat
[array([2.99999705e+02, 1.47167585e+00, 4.60307958e+00, 5.62952941e+00, 1.43983120e-01]), array([ 0.62273737, -0.64436076,  0.7511759 ,  6.59485476])]
rounded upper_interval is [ 4.838e+02  1.800e+00  5.000e+00  6.800e+00  2.000e-01  7.000e-01 -4.000e-01  1.000e+00  9.500e+00]
rounded lower_interval is [ 1.86e+02  1.20e+00  4.20e+00  4.70e+00  1.00e-01  5.00e-01 -8.00e-01 5.00e-01  3.70e+00]

SEED111
2 out of 3 repeats give same esults - 23864s - converge success
7hour/repeat
[array([3.33777037e+02, 1.78394320e+00, 4.81612525e+00, 7.54948178e+00, 1.42665809e-01]), array([ 0.64703781, -0.55572811,  0.61478058,  6.28554463])]
rounded upper_interval is [ 5.87e+02  2.10e+00  5.30e+00  8.90e+00  2.00e-01  8.00e-01 -3.00e-01 9.00e-01  9.90e+00]
rounded lower_interval is [ 1.898e+02  1.500e+00  4.400e+00  6.400e+00  1.000e-01  5.000e-01 -8.000e-01  3.000e-01  2.700e+00]

SEED112
3 repeats give more reasonable results - 16074s
4.5hour/repeat
#result after removing the mean
#[array([3.14265106e+01, 6.78922238e-03, 2.63050989e-04, 4.95708187e-02,9.43522173e-02]), array([ 0.75543821, -1.08139473,  1.3192238 , 19.99444997])]
[array([4.73259548e+02, 1.66378058e-03, 3.61646603e-07, 2.73417703e+01, 5.99216627e-01]), array([ 0.09304695, -1.20357031,  0.80844794, 19.70071545])]
rounded upper_interval is [ 5.279e+02  0.000e+00  0.000e+00  3.090e+01  8.000e-01  1.000e-01 -1.100e+00  1.000e+00  2.060e+01]
rounded lower_interval is [ 4.243e+02  0.000e+00  0.000e+00  2.420e+01  5.000e-01  1.000e-01 -1.300e+00  6.000e-01  1.880e+01]

SEED113
3 repeats give more reasonable results - 14135s
4hour/repeat
#result after removing the mean
#[array([3.40912004e+01, 2.88053052e-02, 3.16930686e-06, 1.06310894e+01,7.05285829e-01]), array([-0.38713266, -1.13978969,  1.33774919, 19.98281115])]
[array([4.25271288e+02, 6.85571415e-03, 1.05133842e-04, 1.07801495e+01, 6.36996196e-01]), array([ 0.10621493, -1.19031652,  1.25539856, 18.88930779])]
rounded upper_interval is [ 4.667e+02  0.000e+00  0.000e+00  1.290e+01  7.000e-01  1.000e-01 -1.000e+00  1.600e+00  2.060e+01]
rounded lower_interval is [ 3.875e+02  0.000e+00  0.000e+00  9.000e+00  6.000e-01  1.000e-01 -1.400e+00  1.000e+00  1.720e+01]

SEED114
3 repeats give more reasonable results - 20365s
6hour/repeat
#result after removing the mean
#[array([2.69110790e+01, 1.51595247e-02, 2.25790663e-04, 1.61559103e+01,3.23324701e-01]), array([ 0.23476775, -1.14538939,  1.18789624, 19.35332021])]
[array([5.25274965e+02, 1.72114145e-04, 3.95489720e-07, 1.21455784e+03, 2.25733795e+00]), array([-9.63123998e-02,  1.09995109e+01,  2.63131806e-01,  1.12170020e+02])]
upper_interval is [ 6.48392403e+02  2.52619809e-04  1.05423513e-06  1.67672037e+03 2.48078642e+00 -8.69586032e-02  1.19103728e+01  6.74036889e-01 1.20280630e+02]
lower_interval is [ 4.25535197e+02  1.17264275e-04  1.48365497e-07  8.79783399e+02 2.05401584e+00 -1.05666196e-01  1.00886491e+01 -1.47773276e-01 1.04059410e+02]

SEED115
2 out of 3 repeats give same esults - 24671s - converge success
7hour/repeat
#result after removing the mean - NOT converge
#[array([3.44770187e+01, 3.94475220e-02, 5.54659822e-06, 1.18366306e+01,3.02338956e-01]), array([ 0.27768926, -1.3679281 ,  1.25092948, 17.70081046])]
[array([3.36168686e+02, 1.62257744e+00, 4.88172021e+00, 6.16126132e+00, 1.13201430e-01]), array([ 0.62122213, -0.88060062,  0.76168632,  4.59096707])]
rounded upper_interval is [ 5.57e+02  1.90e+00  5.30e+00  7.10e+00  1.00e-01  8.00e-01 -7.00e-01 1.00e+00  8.10e+00]
rounded lower_interval is [ 2.029e+02  1.400e+00  4.500e+00  5.300e+00  1.000e-01  5.000e-01 -1.100e+00  5.000e-01  1.100e+00]

SEED116
3 repeats give more reasonable results - 22166s
6hour/repeat
[array([3.89229001e+02, 2.59156701e-04, 1.25885381e-19, 1.69599544e-02, 2.09641552e+01]), array([ 0.20142198, -1.20647823,  1.35738602, 19.03663243])]
rounded upper_interval is [ 4.45e+02  0.00e+00  0.00e+00  4.70e+00  6.05e+01  2.00e-01 -1.10e+00 1.50e+00  2.03e+01]
rounded lower_interval is [ 3.404e+02  0.000e+00  0.000e+00  0.000e+00  7.300e+00  2.000e-01 -1.300e+00  1.200e+00  1.770e+01]

SEED117
2 out of 3 repeats give same esults - 29997s - converge success
8hour/repeat
#result after removing the mean - NOT converge
#[array([4.86730837e+01, 4.76586209e-03, 7.96602263e-02, 1.03467248e+01, 6.00932416e-01]), array([-0.25919224, -1.20377598,  1.21691518, 19.03919942])]
[array([3.42770151e+02, 1.94880911e+00, 4.85701380e+00, 6.22950441e+00,1.43984095e-01]), array([ 0.63743035, -0.49979846,  0.65130789,  6.61256977])]
rounded upper_interval is [ 6.156e+02  2.200e+00  5.300e+00  7.200e+00  2.000e-01  8.000e-01 -3.000e-01  9.000e-01  9.400e+00]
rounded lower_interval is [ 1.909e+02  1.700e+00  4.500e+00  5.400e+00  1.000e-01  5.000e-01 -7.000e-01  4.000e-01  3.800e+00]

SEED118
2 out of 3 repeats give same esults - 18170s - converge success
5hour/repeat
#result after removing the mean - NOT converge
#[array([3.62342930e+01, 1.27476970e-05, 9.36817893e-02, 1.23640225e+01,1.89905009e-01]), array([ 0.0205317 , -1.13933791,  1.57543625, 20.42045084])]
[array([2.89996013e+02, 1.80030938e+00, 4.95650288e+00, 6.78322941e+00,1.53237616e-01]), array([ 0.69770189, -0.55447262,  0.60713532,  4.92601436])]
rounded upper_interval is [ 4.924e+02  2.100e+00  5.400e+00  8.000e+00  2.000e-01  9.000e-01 -3.000e-01  9.000e-01  8.600e+00]
rounded lower_interval is [ 1.708e+02  1.600e+00  4.500e+00  5.700e+00  1.000e-01  5.000e-01 -8.000e-01  3.000e-01  1.200e+00]

SEED119
3 repeats give more reasonable results - 20847s - converge success
6hour/repeat
[array([3.51861996e+02, 1.76974109e+00, 4.49911163e+00, 7.04539521e+00,1.41569305e-01]), array([ 0.63582871, -0.64688165,  0.79516124,  5.93787611])]
rounded upper_interval is [ 6.38e+02  2.00e+00  4.90e+00  8.30e+00  2.00e-01  8.00e-01 -4.00e-01 1.00e+00  8.80e+00]
rounded lower_interval is [ 1.94e+02  1.60e+00  4.10e+00  6.00e+00  1.00e-01  5.00e-01 -8.00e-01 6.00e-01  3.00e+00]
import countries
cc = countries.CountryChecker('./TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp')
print cc.getCountry(countries.Point(49.7821, 3.5708)).iso

[array([2.99999705e+02, 1.47167585e+00, 4.60307958e+00, 5.62952941e+00, 1.43983120e-01]), array([ 0.62273737, -0.64436076,  0.7511759 ,  6.59485476])]
[array([3.33777037e+02, 1.78394320e+00, 4.81612525e+00, 7.54948178e+00, 1.42665809e-01]), array([ 0.64703781, -0.55572811,  0.61478058,  6.28554463])]
[array([3.42770151e+02, 1.94880911e+00, 4.85701380e+00, 6.22950441e+00,1.43984095e-01]), array([ 0.63743035, -0.49979846,  0.65130789,  6.61256977])]
[array([2.89996013e+02, 1.80030938e+00, 4.95650288e+00, 6.78322941e+00,1.53237616e-01]), array([ 0.69770189, -0.55447262,  0.60713532,  4.92601436])]
[array([3.51861996e+02, 1.76974109e+00, 4.49911163e+00, 7.04539521e+00,1.41569305e-01]), array([ 0.63582871, -0.64688165,  0.79516124,  5.93787611])]

import sys
try:
	from osgeo import ogr, osr, gdal
except:
	sys.exit('ERROR: cannot find GDAL/OGR modules')



