import matplotlib.pyplot as plt
import csv

x=[]
y=[]

with open('./output/test_DEPTH_CALI_predict.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))


plt.plot(x,y, marker='.')

plt.title('Data from the CSV File: actual and predict')

plt.xlabel('Actual Data')
plt.ylabel('Prediction Data')

plt.show()
