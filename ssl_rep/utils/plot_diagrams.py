import numpy as np
import matplotlib.pyplot as plt

from ssl_rep.utils.util import generate_data

# Use pyplot styling for a bit nicer diagrams
plt.style.use('fivethirtyeight')


# pyplot colors for class 0, class 1, and class unlabeled
colors = ['r', 'b', 'k']

# Definition for the axes limits on all figures
xlimits = [-1.5, 2.5]
ylimits = [-1, 1.5]

xlabel = 'Feature 1, x_1'
ylabel = 'Feature 0, x_0'

# Our basic data set of two points and a query point
X_two, y_two = np.array([[1.0, 0.0], [0.1, 0.0]]), np.array([0, 1])
X_query = np.array([[2, 0]])

# Generate the (random) unlabeled data
X_unlabeled, y_unlabeled = generate_data(num_samples=1000)

# Negligible small number used for plotting
eps = 1E-12

do_save = True  # Toggle to view the diagrams within the run or just save them in relative locations

# // Image of two points, with query
plt.figure(figsize=(20, 10))
plt.scatter(X_two[0, 0], X_two[0, 1], c=colors[y_two[0]], s=100, label=f'Label {y_two[0]}')
plt.scatter(X_two[1, 0], X_two[1, 1], c=colors[y_two[1]], s=100, label=f'Label {y_two[1]}')
plt.scatter(X_query[0, 0], X_query[0, 1], c=colors[2], s=100, label=f'Query point?')
plt.text(X_two[0, 0], X_two[0, 1], f'Label {y_two[0]}')
plt.text(X_two[1, 0], X_two[1, 1], f'Label {y_two[1]}')
plt.text(X_query[0, 0], X_query[0, 1], f'Query point?')
plt.ylabel(xlabel)
plt.xlabel(ylabel)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.grid(b=None)
plt.legend()
if not do_save:
    plt.show()
else:
    plt.savefig('../im/two_points_query.svg')


# // Image of two points with decision boundary, with query
plt.figure(figsize=(20, 10))
plt.scatter(X_two[0, 0], X_two[0, 1], c=colors[y_two[0]], s=100, label=f'Label {y_two[0]}')
plt.scatter(X_two[1, 0], X_two[1, 1], c=colors[y_two[1]], s=100, label=f'Label {y_two[1]}')
plt.scatter(X_query[0, 0], X_query[0, 1], c=colors[2], s=100, label=f'Predict label 0 ??')
plt.plot([0.55 + eps, 0.55 - eps], ylimits, 'k--', label='Decision boundary')
plt.text(X_two[0, 0], X_two[0, 1], f'Label {y_two[0]}')
plt.text(X_two[1, 0], X_two[1, 1], f'Label {y_two[1]}')
plt.text(X_query[0, 0], X_query[0, 1], f'Predict label 0 ??')
plt.ylabel(xlabel)
plt.xlabel(ylabel)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.grid(b=None)
plt.legend()
if not do_save:
    plt.show()
else:
    plt.savefig('../im/two_points_query_boundary.svg')


# // Image of two points, with 10000 unlabeled points.
plt.figure(figsize=(20, 10))
plt.scatter(X_two[0, 0], X_two[0, 1], c=colors[y_two[0]], s=100, label=f'Label {y_two[0]}')
plt.scatter(X_two[1, 0], X_two[1, 1], c=colors[y_two[1]], s=100, label=f'Label {y_two[1]}')
plt.scatter(X_query[0, 0], X_query[0, 1], c=colors[2], s=100, label=f'Query point?')
plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=colors[2], s=20, alpha=0.25, label='Unlabeled data')
plt.text(X_two[0, 0], X_two[0, 1], f'Label {y_two[0]}')
plt.text(X_two[1, 0], X_two[1, 1], f'Label {y_two[1]}')
plt.text(X_query[0, 0], X_query[0, 1], f'Query point?')
plt.ylabel(xlabel)
plt.xlabel(ylabel)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.grid(b=None)
plt.legend()
if not do_save:
    plt.show()
else:
    plt.savefig('../im/two_points_query_unlabeled.svg')


# // Image of two points, with 10000 unlabeled points, with decision boundary.
xrange = np.linspace(xlimits[0], xlimits[1], num=1000)
yrange = 0.25 + 0.5 * np.cos(np.pi * xrange)

plt.figure(figsize=(20, 10))
plt.scatter(X_two[0, 0], X_two[0, 1], c=colors[y_two[0]], s=100, label=f'Label {y_two[0]}')
plt.scatter(X_two[1, 0], X_two[1, 1], c=colors[y_two[1]], s=100, label=f'Label {y_two[1]}')
plt.scatter(X_query[0, 0], X_query[0, 1], c=colors[2], s=100, label=f'Label 1 ??')
plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=colors[2], s=20, alpha=0.25, label='Unlabeled data')
plt.plot(xrange, yrange, 'k--', label='Decision boundary')
plt.text(X_two[0, 0], X_two[0, 1], f'Label {y_two[0]}')
plt.text(X_two[1, 0], X_two[1, 1], f'Label {y_two[1]}')
plt.text(X_query[0, 0], X_query[0, 1], f'Label 1 ??')
plt.ylabel(xlabel)
plt.xlabel(ylabel)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.grid(b=None)
plt.legend()
if not do_save:
    plt.show()
else:
    plt.savefig('../im/two_points_query_unlabeled_boundary.svg')
