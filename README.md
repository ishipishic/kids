Week 1 Written Report: 
Using the ARIMA Model for Time Series Forecasting 
1. What I've Learned About the ARIMA Model

One of the most popular statistical methods for predicting time series data is the ARIMA model. I discovered that ARIMA, or AutoRegressive Integrated Moving Average, integrates three crucial elements:

Autoregressive (AR-parameter p): This component simulates the relationship between a series' current value and its past values. It records the data's momentum or persistence.

Integrated (I-parameter d): This indicates how many times the series must be differentiated in order to reach stationarity.

Moving Average (MA; parameter q): This part simulates how previous forecast errors affect the current observation.

When a series exhibits temporal dependence in a linear form, ARIMA is typically employed. It works well with financial and economic data like production output, stock prices, inflation, and rainfall. I gained a better understanding of how ARIMA offers an interpretable baseline prior to using more sophisticated models thanks to the implementation exercise.

ARIMA presumptions

I was aware that ARIMA is predicated on a number of assumptions:

After differencing, the underlying series ought to be stationary.

The fitted model's residuals must have a constant variance and zero mean, resembling white noise.

It is assumed that relationships between lags are additive and linear.

Over time, the dependence structure doesn't change.
Differencing
To address this issue I applied first order differencing (d = 1):
yt′=yt−yt−1

After differencing, the ADF p-value became small, confirming that the transformed series was stationary and suitable for ARIMA modeling.
3. Understanding ACF and PACF Plots
3.1 Autocorrelation Function (ACF)
The ACF plot measures the correlation between the series and its lagged values. I learned that:
slow decay in ACF → non stationary
sharp cutoff → suggests MA order.
3.2 Partial Autocorrelation Function (PACF)
PACF measures correlation with a lag after removing effects of intermediate lags. It helps detect:
AR order p through significant spikes.
3.3 Parameter Selection
By examining both ACF and PACF of the differenced series, I observed that:
lag 1 showed prominent spikes
higher lags were insignificant.
Hence ARIMA(1,1,1) was chosen as an initial model. I also learned that in practice one can apply grid search using AIC/BIC to select optimal parameters rather than manual inspection.
4. Learnings from the Coding Exercise
4.1 Challenges Faced
Understanding how to test stationarity correctly
choosing (p,d,q) from plots
interpreting ARIMA summary table.
4.2 Observations Made
The fitted ARIMA model successfully captured the general trend of the training data.
Forecasts for the test period were smooth and followed direction of the actual series.
Residual plots were centered near zero with no visible periodic structure.
4.3 Insights
ARIMA works well for short-term forecasting where dependence is linear.
The model is sensitive to the differencing order; without stationarity performance drops sharply.
5. Explanation of Obtained Plots
5.1 Time Series Plot
The original curve showed an upward drifting pattern, confirming non-stationarity.
5.2 Forecast vs Actual
The comparison plot indicated:
predictions close to actual path
slight lag during rapid changes.
5.3 ACF/PACF
These curves demonstrated that:
dependence mainly exists in first few lags
higher lags contribute little.
5.4 Residual Plot
Residuals behaved randomly → indicating model assumptions largely satisfied.
6. Model Evaluation Interpretation
Error metrics were calculated:
MAE – average absolute deviation
MSE – squared error penalizing large gaps
RMSE – interpretable in original units.
The obtained RMSE was moderate, meaning ARIMA provided reasonable performance but not perfect accuracy.
7. Summary of Findings
This assignment helped me understand:
Why stationarity is essential for ARIMA
how ACF and PACF guide identification
how to fit ARIMA using statsmodels
how to evaluate forecasts on unseen data.
8. Limitations of the Model
ARIMA assumes linearity and cannot capture non-linear structural breaks.
It is a univariate model ignoring external features.
Forecasts are often overly smooth.
9. Possible Improvements
Apply SARIMA if seasonality exists.
Use AIC/BIC grid search for better (p,q).
Try transformations like log returns.
Walk-forward validation.


Week 2 Assignment Report

LSTM & ARIMA Comparison for Stock Price Forecasting
1. Introduction

This week I have learned that time series forecasting can be done by two different methods; the ARIMA model, which is a statistical method based on linear relationships, and the LSTM model, a deep learning technique that can capture non-linear relationships over time. 
The coding practice gave me insights into how the dataset's characteristics, the models' assumptions, and the data preparation steps can all affect the accuracy of the forecast directly.

2. Understanding ARIMA and LSTM Models
ARIMA Model

The ARIMA (AutoRegressive Integrated Moving Average) model is parameterized using (p, d, q):

p – Autoregressive order: considers previous values

d – Number of times the original time series has been differenced: guarantees stationarity

q – Order of the moving average: factors in past forecast inaccuracies 

The model assumes:

positively correlated previous values occur in the series

linear relationships among the variables

the noise is white.

LSTM Model

The LSTM (Long Short-Term Memory) network is a type of RNN (Recurrent Neural Network) that is able to maintain a stable performance from the start to the end of a long sequence. It includes:

Forget gate – what information can be discarded?

Input gate – what new information can be added to the memory?

Output gate – which part of the state of memory should be shown?

In contrast to ARIMA, LSTM:

makes no absolute requirement for data to be stationary 

detects not just simple but also intricate non-linear patterns 

accepts multiple input features also.

3. Important Concepts Learned

3.1 Stationarity and Differencing

Stationarity refers to a situation where the mean and variance of a time series do not change over time. 
Stock prices are typically non-stationary because they usually have random movements with trends. The ADF test in Week 1 pointed to a high p-value, hence:
first differencing was necessary for ARIMA
without differencing, ARIMA forecasts become spurious.

3.2 ACF and PACF Plots

ACF (Autocorrelation Function): assesses the relationship with previous lags → assists in selecting q

PACF: full correlation minus intermediate effects → assists in selecting p.

The low-order AR and MA terms were indicated by initial lags’ spikes. The identification of parameters through these curves was very important before applying ARIMA.

3.3 Sliding Window Technique for LSTM

Neural networks require the data to be in supervised format.
The LSTM was allowed to capture the sequential context instead of just learning isolated points through the use of a 60-day window.

3.4 Data Normalization

Normalization in the range of 0-1 is very critical due to the following reasons:

LSTM applies gradient descent

large numbers decrease the speed of convergence

unscaled values lead to unstable training.

The MinMaxScaler processed the time series so that the learning of the network was efficient.

4. Insights Gained from Implementation
4.1 Challenges Faced

ARIMA

the mapping of ACF/PACF to (p,d,q) was really confusing

the ARIMA forecasts were always smooth and lagging.

LSTM

3D input reshaping [samples, timesteps, features]

epoch/batch size selection

overfitting prevention.

4.2 Model Performance Observations

ARIMA closely followed the overall upward trend, but turning points were missed.

LSTM predictions were faster in response and, thus, closer to the actual prices.

Residual plots:

ARIMA residuals appeared random → the model fulfilled assumptions.

LSTM residuals were smaller but still indicated slight clustering → this suggests that hyperparameter tuning is still needed.

Learning curves:

the training loss dropped consistently

the validation loss had begun to plateau → an early indicator of mild overfitting.

5. Visualization Explanation
5.1 Stock Price vs Time Plot

The projection indicated:

in blue the real observed prices

in orange the ARIMA forecast

in green the LSTM forecast.

The gap between the curves was the measure of prediction error.

5.2 Residual Plots

The ARIMA residuals which were centered around zero showed that a linear structure was captured.

The LSTM residuals which were of lower magnitude showed that non-linear relations were learned.

5.3 ACF/PACF

The ACF/PACF helped in identifying the dependence only in the initial lags which in turn justified the ARIMA(1,1,1) model.

5.4 LSTM Learning Curves

The gap between the training and validation loss showed the need for:

dropout,

fewer neurons,

and more data.

6. Comparison of ARIMA and LSTM
Error Metric Comparison

The MAE / RMSE / MAPE were calculated on the same test set.

The LSTM model obtained lower values.


Conceptual Differences

Dimension
ARIMA
LSTM
Nature
Linear statistical
Non-linear neural
Stationarity
Required
Not strict
Trend
Captures via differencing
Learns directly
Seasonality
Needs SARIMA
Learns if present
Computation
Very cheap
Expensive


7. Advantages and Drawbacks
7.1 ARIMA

Advantages

comprehensible

quick

suitable for financial series that are linear.

Restrictions

unable to record non-linear jumps

Univariate

sensitive to the assumption of stationarity.

7.2 LSTM

Advantages

captures temporal dependence that is not linear.

Adaptable

Multivariate extension is possible.

Restrictions

requires normalization.

hyperparameter-sensitive

computationally demanding.

8. Key Findings Synopsis

Because stock prices are not stationary, ARIMA requires differencing.

Identification requires ACF/PACF.

Sliding window and normalization are necessary for LSTM.

Better prediction accuracy was obtained at a higher cost using the deep learning approach.

9. Potential Enhancements and Upcoming Expansions

Instead of using manual identification, use grid search ARIMA.

Apply:

LSTM in both directions

mechanism of attention

dropout rate of 0.3

Add moving averages and volume.

For both models, use walk-forward validation.

