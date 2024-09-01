import pandas as pd
from sklearn import preprocessing

from tensor import TensorStream


T = 1
W = 30
rank = 5
start_time = 0
epoch = 200
ALGORITHM = "SNS_MAT"
category_labels = ["ecommerce", "region"]
time_label = "time"
value_label = "value"


events = pd.read_csv("ecommerce_region_W.csv")
events[time_label] = pd.to_datetime(events[time_label])
events = events.sort_values(time_label)

oe = preprocessing.OrdinalEncoder()
events[category_labels + [time_label]] = oe.fit_transform(events[category_labels + [time_label]]).astype(int)

# counts at non-temporal each mode + W
dimensions = [W] + (events[category_labels].max().values + 1).tolist()

ts = TensorStream(dimensions, T, rank, start_time, ALGORITHM, events, category_labels, time_label, value_label)

for e in range(epoch):
    ts.updateCurrent()
    ts.updateTensor()

    print("epoch%d: time=%d, rmse=%f, fitness=%f" % (e + 1, ts.current_time, ts.tensor.rmse(), ts.tensor.fitness()))