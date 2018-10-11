import numpy as np
import matplotlib.pyplot as plt
import libconf
from class_mortality import mortality
from scipy.optimize import curve_fit

"""
Cleans the survival function (new bin, moving average).
Works out the mortality function from the clean survival function.
Does Weibull and Gompertz fit on a specific range. The initial values
for the Weibull (k0, lambda) and the Gompertz (a, b) fits need to be tell.
"""

# Data configuration
path_config = './config.cfg' # input('Config path:')
with open(path_config) as cfg:
	config = libconf.load(cfg)

path_data = config['path_data'] # Data location
b = config['b'] # It takes 1/b data points in the full data set
m = config['m'] # Moving average over m points
start_fit_time = config['start_fit_time'] # Start of the fit (real time)
end_fit_time = config['end_fit_time'] # End of the fit (real time)
k0 = config['k0'] # Initial value of k (Weibull)
lambda_0 = config['lambda_0'] # Initial value of lambda (Weibull)
a0 = config['a0'] # Initial value of a (Gompertz)
b0 = config['b0'] # Initial value of b (Gompertz)
unit_time = config['unit_time'] # X label (plot)

# Data loading, object calling
t, s = np.loadtxt(path_data, unpack=True)
mt = mortality(t, s, b, m)

# Survival function normalization and data cleaning
s_scaled = mt.scaling()
t_clean, s_clean = mt.cleaning()

# Mortality calculation
mortality_t, mortality_s = mt.mortality_function()

# Fit boundaries and fit over the survival function and/or the mortality function
start_fit = mt.time_index(t_clean, start_fit_time)
end_fit = mt.time_index(t_clean, end_fit_time)

#popt_weibull_mort, pcov_weibull_mort = curve_fit(mt.weibull_mortality, mortality_t[start_fit:end_fit], mortality_s[start_fit:end_fit], p0=[k0, lambda_0])
#popt_gompertz_mort, pcov_gompertz_mort = curve_fit(mt.gompertz_mortality, mortality_t[start_fit:end_fit], mortality_s[start_fit:end_fit], p0=[a0, b0])

###popt_weibull, pcov_weibull = curve_fit(mt.weibull, t_clean[start_fit:end_fit], s_clean[start_fit:end_fit], p0=[k0, lambda_0])
###popt_gompertz, pcov_gompertz = curve_fit(mt.gompertz, t_clean[start_fit:end_fit], s_clean[start_fit:end_fit], p0=[a0, b0])

# Displaying of Weibull/Gompertz parameters
###print('Weibull: k = %s, lambda = %s' % (popt_weibull[0], popt_weibull[1]))
###print('Gompertz: a = %s, b = %s' %(popt_gompertz[0], popt_gompertz[1]))

# Plotting section
plt.figure(0)
plt.semilogy(t, s_scaled, 'ro', ms=1)
plt.plot(t_clean, s_clean, 'r-')
plt.plot(mortality_t, mortality_s, 'b*')
###plt.plot(np.arange(int(start_fit_time), int(end_fit_time), 0.01), mt.weibull_mortality(np.arange(int(start_fit_time), int(end_fit_time), 0.01), popt_weibull[0], popt_weibull[1]), 'b--')
###plt.plot(np.arange(int(start_fit_time), int(end_fit_time), 0.01), mt.gompertz_mortality(np.arange(int(start_fit_time), int(end_fit_time), 0.01), popt_gompertz[0], popt_gompertz[1]), 'b-.')
plt.title('Film Mortality')
plt.xlabel(unit_time)
plt.xlim([0, t[-1]])
plt.ylim([0.0001, 1])
legend = ['Survival function', 'Clean data', 'Mortality function', 'Weibull fit', 'Gompertz fit']
plt.legend(legend, loc=3)

plt.figure(1)
plt.semilogy(t, s_scaled, 'ro', ms=1)
###plt.plot(np.arange(int(start_fit_time), int(end_fit_time), 0.01), mt.weibull(np.arange(int(start_fit_time), int(end_fit_time), 0.01), popt_weibull[0], popt_weibull[1]), 'r--')
###plt.plot(np.arange(int(start_fit_time), int(end_fit_time), 0.01), mt.gompertz(np.arange(int(start_fit_time), int(end_fit_time), 0.01), popt_gompertz[0], popt_gompertz[1]), 'r-.')
plt.title('Survival Function')
plt.xlabel(unit_time)
legend = ['Survival function', 'Weibull fit', 'Gompertz fit']
plt.xlim([0, t[-1]])
plt.ylim([0, 1])
plt.legend(legend, loc=3)

plt.show()
