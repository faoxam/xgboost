import sys

import pandas as pd
import matplotlib.pyplot as plt



print("Parameter pass: ", sys.argv[1])

plt.rcParams["figure.figsize"] = [10.00, 10.0]
plt.rcParams["figure.autolayout"] = True
columns = ["INDEX", "Actual", "Predicted"]
df = pd.read_csv(sys.argv[1], usecols=columns)
print("Contents in csv file:\n", df)
plt.plot(df.INDEX, df.Actual)
plt.plot(df.INDEX, df.Predicted)
plt.show()
