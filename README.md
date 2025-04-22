# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

 data=pd.read_csv('silver.csv')

N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
X = data['USD'][::-1].reset_index(drop=True)
plt.plot(X)
plt.title('Silver Prices (USD)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Silver Prices ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Silver Prices PACF')

plt.tight_layout()
plt.show()
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()

```
### OUTPUT:

#### Original data:

![image](https://github.com/user-attachments/assets/8edfd0bb-bbda-4c76-ae52-3fa441091549)

#### Partial Autocorrelation

![image](https://github.com/user-attachments/assets/32d306cb-7903-4d1b-8b7e-8b8d58d2547e)

#### Autocorrelation

![image](https://github.com/user-attachments/assets/bb2de61e-1bb7-4e45-8e37-002c41079dcd)

#### SIMULATED ARMA(1,1) PROCESS:

![Screenshot 2025-04-15 085213](https://github.com/user-attachments/assets/22fdc324-0b6f-44e0-bd27-903c1ad82389)

#### Partial Autocorrelation

![image](https://github.com/user-attachments/assets/571d886f-cfaf-40cc-b249-30f0dd404e23)

#### Autocorrelation

![image](https://github.com/user-attachments/assets/0ea159d3-de7e-4e7a-9a92-032aaef5053f)

#### SIMULATED ARMA(2,2) PROCESS:

![image](https://github.com/user-attachments/assets/8d501d69-39e4-4d0b-8e6a-d643ed964762)

#### Partial Autocorrelation

![image](https://github.com/user-attachments/assets/f306389e-2537-462c-a18a-cf63123626e7)

#### Autocorrelation

![image](https://github.com/user-attachments/assets/9b1b0ff1-1e89-4ba7-81fd-6632ebeaa2bb)

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
