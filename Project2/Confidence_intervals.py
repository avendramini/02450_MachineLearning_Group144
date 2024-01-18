import matplotlib.pyplot as plt
import statistics
from math import sqrt


def plot_confidence_interval(x, values, z=1, color='#2187bb', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval


plt.xticks([1, 2, 3], ['BLvsLR', 'BLvsANN', 'ANNvsLR'])
plt.title('Confidence Interval')
plt.axhline(y=0, color='black', linestyle='--')
plot_confidence_interval(1, [94.96, 133.16])
plot_confidence_interval(2, [93.75, 138.59])
plot_confidence_interval(3, [-12.25, 7.94])
plt.show()