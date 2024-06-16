import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

class IRM:
    def __init__(self, theta: float, mu: float, sigma: float, r0: float):
        """
        Initialize the interest rate model parameters.
        
        :param theta: Speed of reversion
        :param mu: Long-term mean level
        :param sigma: Volatility
        :param r0: Initial interest rate
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.r0 = r0
    
    def vasicek_milstein_path(self, T: float, N: int, n_paths: int):
        """
        Simulate vasicek paths using the Milstein scheme.
        
        :param T: Time horizon
        :param N: Number of time steps
        :param n_paths: Number of simulated paths
        :return: Simulated paths
        """
        dt = T / N
        rates = np.zeros((N + 1, n_paths))
        rates[0] = self.r0
        for i in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            rates[i] = rates[i - 1] + self.theta * (self.mu - rates[i - 1]) * dt + self.sigma * dW + 0.5 * self.sigma**2 * (dW**2 - dt)
        return rates
    
    def CIR_milstein_path(self, T: float, N: int, n_paths: int):
        """
        Simulate Cox–Ingersoll–Ross paths using the Milstein scheme.
        
        :param T: Time horizon
        :param N: Number of time steps
        :param n_paths: Number of simulated paths
        :return: Simulated paths
        """
        dt = T / N
        rates = np.zeros((N + 1, n_paths))
        rates[0] = self.r0
        for i in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            rates[i] = rates[i - 1] + self.theta * (self.mu - rates[i - 1]) * dt + self.sigma * np.sqrt(rates[i - 1]) * dW + 0.5 * self.sigma**2 * (dW**2 - dt) / (rates[i - 1])
        return rates

    def calibrate(self, market_data: np.ndarray, path_method = 'vasicek_milstein_path'):
        """
        Calibrate the model parameters to market data.
        
        :param market_data: Array of market data for calibration
        :param path_method: A name of path method to be used for calibration, if none chosen, vasicek_milstein_path is the default
            Options: vasicek_milstein_path, CIR_milstein_path
        """
        def objective(params):
            theta, mu, sigma = params
            model = IRM(theta, mu, sigma, self.r0)
            method = getattr(model, path_method)
            simulated_data = method(T=len(market_data)-1, N=len(market_data)-1, n_paths=1)
            simulated_data = simulated_data.flatten()
            return np.sum((market_data - simulated_data)**2)
        
        initial_guess = [self.theta, np.mean(market_data), np.std(market_data)]
        result = minimize(objective, initial_guess, bounds=[(0, None), (0, None), (0, None)])
        self.theta, self.mu, self.sigma = result.x
        print(f"Calibrated parameters: theta={self.theta}, mu={self.mu}, sigma={self.sigma}")
        
    
    def plot_paths(self, paths: np.ndarray, title: str = "Simulated Interest Rate Paths"):
        """
        Plot the simulated interest rate paths.
        
        :param paths: Simulated paths to plot
        :param title: Title of the plot
        """
        plt.figure(figsize=(10, 6))
        for path in paths.T:
            plt.plot(path)
        plt.title(title)
        plt.xlabel("Time steps")
        plt.ylabel("Interest Rate")
        plt.show()

# Example usage
# Define the file path 
file_path = './market_data_PL.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path, header=None, names=['Date', 'Period', 'Value'])

# Convert the 'Value' column to float
data['Value'] = data['Value'].astype(float)

market_data = list(map(lambda value: value[2], data.values))
print (np.std(market_data))

model = IRM(r0=market_data[0], theta=0.5, mu=0.03, sigma=0.02)
model.calibrate(market_data)
milstein_paths = model.vasicek_milstein_path(T=10, N=1_000_000, n_paths=10)
model.plot_paths(milstein_paths, title='vasicek')


model = IRM(r0=market_data[0], theta=0.5, mu=0.03, sigma=0.02)
model.calibrate(market_data, 'CIR_milstein_path')
milstein_paths = model.CIR_milstein_path(T=10, N=1_000_000, n_paths=10)
model.plot_paths(milstein_paths, title="CIR_milstein_path")

