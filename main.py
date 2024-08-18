import pandas as pd

from tensor import TensorStream


T = 20
W = 10
rank = 5
start_time = 0
epoch = 200
ALGORITHM = "SNS_MAT"


data = pd.read_csv("ecommerce_region_W.csv")

# counts at non-temporal each mode + W
dimensions = [data.iloc[:, i].nunique() for i in range(data.shape[1] - 2)] + [W]

uniques = [list(data.iloc[:, i].unique()) for i in range(data.shape[1] - 2)]

# grouped by time
data['time'] = pd.to_datetime(data['time'])
events = [[([uniques[i].index(row[i + 1]) for i in range(data.shape[1] - 2)], row[data.shape[1]])
           for row in event.itertuples()]
           for _, event in data.groupby(by='time', sort=True)]


ts = TensorStream(events, dimensions, T, rank, start_time, ALGORITHM)

for e in range(epoch):
    ts.updateCurrent()
    ts.updateTensor()

    print("epoch%d: time=%d, rmse=%f, fitness=%f" % (e + 1, ts.current_time, ts.tensor.rmse(), ts.tensor.fitness()))