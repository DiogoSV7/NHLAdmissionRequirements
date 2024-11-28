import csv
import numpy as np
import matplotlib.pyplot as plt

# QUESTION 3

# Answer to Question 1:
# - The issue in the original code is that the precision and recall values were being used incorrectly 
#   in the plot function. The x-axis should represent recall and the y-axis should represent precision.
# - Additionally, there was a potential issue with data formatting, as strings were being passed where 
#   numerical values were expected. 

# Answer to Question 2:
# - The solution is to ensure that the precision values are plotted on the x-axis and recall values on the y-axis.
# - We also need to make sure that the data is properly read as numerical values (i.e., floats) and then passed to the plot function.

def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.

    :param csv_file_path: The CSV file containing the data to plot.

    | ``1   For some reason the plot is not showing correctly, can you find out what is going wrong?``
    | ``2   How could this be fixed?``

    This example demonstrates the issue.
    It first generates some data in a csv file format and then plots it using the `plot_data` method.
    If you manually check the coordinates and then check the plot, they do not correspond.
    """

    # Step-by-step explanation of the code:
    # 1. We open the CSV file and read its content using the `csv.reader`.
    # 2. We skip the header row using `next(csv_reader)`.
    # 3. We iterate over the remaining rows and convert each element to a float before storing it in the results list.
    # 4. The `results` list is then converted to a NumPy array to facilitate easier plotting.

    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            results.append([float(x) for x in row]) 
        results = np.array(results)

    # plot precision-recall curve
    plt.plot(results[:, 1], results[:, 0])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

f = open("data_file.csv", "w")
w = csv.writer(f)
_ = w.writerow(["precision", "recall"])
w.writerows([[0.013, 0.951],
             [0.376, 0.851],
             [0.441, 0.839],
             [0.570, 0.758],
             [0.635, 0.674],
             [0.721, 0.604],
             [0.837, 0.531],
             [0.860, 0.453],
             [0.962, 0.348],
             [0.982, 0.273],
             [1.0, 0.0]])
f.close()
plot_data('data_file.csv')
