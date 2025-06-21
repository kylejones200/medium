# normal_distribution_plot.py

import numpy as np
import matplotlib.pyplot as plt

def plot_norm_hist(s, mu, sigma, vline=True, title=True):
    count, bins, ignored = plt.hist(s, 30, density=True, alpha=0.5)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp(-(bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')

    if vline:
        lline = -.67*sigma + mu
        uline = .67*sigma + mu
        plt.axvline(lline, color='g')
        plt.axvline(uline, color='g')

    if title:
        plt.title(f"Normal distribution with mean: {mu:.2f} and StDev: {sigma:.2f}")
    plt.savefig("normal_hist.png")
    plt.show()

# Example 1: Standard normal distribution
mu1, sigma1 = 0, 1
s1 = np.random.normal(mu1, sigma1, 1000)
plot_norm_hist(s1, mu1, sigma1)

# Example 2: Larger sample size
mu2, sigma2 = 50, 10
s2 = np.random.normal(mu2, sigma2, 100000)
plt.hist(s2, 30, density=True, alpha=.3)
plt.plot(np.linspace(mu2 - 4*sigma2, mu2 + 4*sigma2, 1000),
         1/(sigma2 * np.sqrt(2 * np.pi)) *
         np.exp(-(np.linspace(mu2 - 4*sigma2, mu2 + 4*sigma2, 1000) - mu2)**2 / (2 * sigma2**2)),
         linewidth=2, color='r')
plt.axvline(-.67*sigma2 + mu2, color='g')
plt.axvline(.67*sigma2 + mu2, color='g')
plt.title(f"Normal distribution with mean: {mu2:.2f} and StDev: {sigma2:.2f}")
plt.savefig("normal_hist_large_sample.png")
plt.show()

# Example 3: Highlighted regions using axvspan
mu3, sigma3 = 0, 1
s3 = np.random.normal(mu3, sigma3, 1000)
plt.hist(s3, 30, density=True, alpha=.5)
plt.plot(np.linspace(-4, 4, 1000),
         1/(sigma3 * np.sqrt(2 * np.pi)) *
         np.exp(-(np.linspace(-4, 4, 1000) - mu3)**2 / (2 * sigma3**2)),
         linewidth=2, color='r')
plt.axvspan(-4, -.67, color='g', alpha=0.1)
plt.axvspan(-.67, 0, color='g', alpha=0.2)
plt.axvspan(0, .67, color='g', alpha=0.3)
plt.axvspan(.67, 4, color='g', alpha=.4)
plt.savefig("normal_hist_highlighted.png")
plt.show()

# Example 4: Box plot
fig, ax = plt.subplots()
ax.set_title('Box Plot of Normal Distribution')
ax.boxplot(s3, showfliers=False, vert=False)
plt.savefig("box_plot.png")
plt.show()