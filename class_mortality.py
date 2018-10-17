import numpy as np

class mortality:


	def __init__(self, t, s, b, m):
		"""
		Input: t (numpy array), s (numpy array), b (int), m (int).
		-----
		Output: None.
		-----
		Comment: Constructor. Takes as self parameters the data time t, the data survival
		function s, b as 1/b the number of taken data points and m the number of points in
		the moving average window.
		-----
		"""
		self.t = t
		self.s = s
		self.b = b
		self.m = m


	def scaling(self):
		"""
		Input: None.
		-----
		Output: s_scaled (numpy array).
		-----
		Comment: Normalizes the survival function (decreasing function going from 1 to 0).
		-----
		"""
		s_scaled = []
		for i in range(len(self.s)):
			s_scaled.append(self.s[i] / self.s[0])
		return s_scaled


	def derivative(self, x, y):
		"""
		Input: x (numpy array), y (numpy array).
		-----
		Output: derivative_x (numpy array), derivative_y (numpy array).
		-----
		Comment: Works out the derivative dx/dy using the straightforward method.
		-----
		"""
		derivative_x = x[:-1]
		derivative_y = []
		for i in range(len(x)-1):
			derivative_y.append((y[i+1]-y[i])/(x[i+1]-x[i]))
		return derivative_x, derivative_y


	def derivative_central_difference_method(self, x, y):
		"""
		Input: x (numpy array), y (numpy array).
		-----
		Output: derivative_x (numpy array), derivative_y (numpy array).
		-----
		Comment: Works out the derivative dx/dy using the central difference method.
		-----
		"""
		derivative_x = x[1:-1]
		derivative_y = []
		for i in range(len(x)-2):
			derivative_y.append((y[i+2]-y[i])/(x[i+2]-x[i]))
		return derivative_x, derivative_y


	def new_bin(self, x, y, b):
		"""
		Input: x (numpy array), y (numpy array), b (int).
		-----
		Output: x_bin (numpy array), y_bin (numpy array).
		-----
		Comment: Takes 1/b data points in the all data set.
		-----
		"""
		x_bin = []
		y_bin = []
		for i in range(int(len(x)/b)):
			x_bin.append(x[i*b])
			y_bin.append(y[i*b])
		return x_bin, y_bin


	def moving_average(self, x, m):
		"""
		Input: x (numpy array), m (int).
		-----
		Output: x_moving[m - 1:] / m (numpy array).
		-----
		Comment: Does the moving average in x with a window of m points.
		-----
		"""
		x_moving = np.cumsum(x, dtype=float)
		x_moving[m:] = x_moving[m:] - x_moving[:-m]
		return x_moving[m - 1:] / m


	def cleaning(self):
		"""
		Input: None.
		-----
		Output: t_clean (numpy array), s_clean (numpy array).
		-----
		Comment: Cleans the data set: normalization, new bin, moving average, add up
		the initial value (at t = 0) to the t_clean and s_clean.
		-----
		"""
		s_scaled = self.scaling()
		t_bin, s_bin = self.new_bin(self.t, s_scaled, self.b)
		t_clean = self.moving_average(t_bin, self.m)
		s_clean = self.moving_average(s_bin, self.m)
		return t_clean, s_clean


	def mortality_function(self):
		"""
		Input: None.
		-----
		Output: mortality_t (numpy array), mortality_s (numpy array).
		-----
		Comment: Does the cleaning and works out the mortality afterwards. The function could
		add up the initial value (at t = 0) to mortality_t and mortality_s (commented lines).
		-----
		"""
		t_clean, s_clean = self.cleaning()
		mortality_t, mortality_s = self.derivative_central_difference_method(t_clean, -np.log(s_clean))
		#mortality_t = np.insert(mortality_t, 0, 0)
		#mortality_s = np.insert(mortality_s, 0, (np.log(s_clean[0]) - np.log(s_clean[1])) / (t_clean[1]-t_clean[0]))
		return mortality_t, mortality_s


	def time_index(self, t, t_real):
		"""
		Input: t (numpy array), t_real (float).
		-----
		Output: t_index (int).
		-----
		Comment: Returns the index t_index in t corresponding to the time t_real.
		-----
		"""
		t_index = 0
		if t_real > max(t):
			t_index = int(len(t)-1)
		else:
			for i in range(len(t)-1):
				if t[i] <= t_real <= t[i+1]:
					t_index = i
					break
		return t_index


	def weibull(self, x, k, lambda_):
		"""
		Input: x (numpy array), k (float), lambda_ (float).
		-----
		Output: np.exp(-(x/lambda_)**k) (numpy array)
		-----
		Comment: Works out the Weibull function from x.
		-----
		"""
		return np.exp(-(x/lambda_)**k)


	def gompertz(self, x, a, b):
		"""
		Input: x (numpy array), a (float), b (float).
		-----
		Output: np.exp(-a*(np.exp(b*x)-1)) (numpy array)
		-----
		Comment: Works out the Gompertz function from x.
		-----
		"""
		return np.exp(-a*(np.exp(b*x)-1))


	def weibull_mortality(self, x, k, lambda_):
		"""
		Input: x (numpy array), k (float), lambda_ (float).
		-----
		Output: (k/lambda_)*(x/lambda_)**(k-1) (numpy array)
		-----
		Comment: Works out the Weibull mortality function from x.
		-----
		"""
		return (k/lambda_)*(x/lambda_)**(k-1)


	def gompertz_mortality(self, x, a, b):
		"""
		Input: x (numpy array), a (float), b (float).
		-----
		Output: a*b*np.exp(b*x) (numpy array)
		-----
		Comment: Works out the Gompertz mortality function from x.
		-----
		"""
		return a*b*np.exp(b*x)
