import time
from datetime import timedelta

from data import get_events
from tensor import TensorStream

'''
file_name = "ecommerce_region_W.csv"
td = timedelta(days=7)
T = 1
W = 104
rank = 9
start_time = 200
epoch = 200
ALGORITHM = "SNS_VEC"
category_labels = ["ecommerce", "region"]
time_label = "time"
value_label = "value"
'''
file_name = "kz.csv"
td = timedelta(seconds=1)
T = 60 * 60 * 24 * 30
W = 10
rank = 10
start_time = 0
epoch = 200
ALGORITHM = "SNS_VEC"
category_labels = ["category_code", "brand"]
time_label = "event_time"
value_label = None
#'''

events = get_events(file_name, td, category_labels, time_label)

# W + counts at non-temporal each mode
dimensions = [W] + (events[category_labels].max().values + 1).tolist()

ts = TensorStream(dimensions, T, rank, start_time, ALGORITHM, events, category_labels, time_label, value_label)

for e in range(epoch):
    ts.updateCurrent()
    
    tic = time.process_time()
    ts.updateTensor()
    toc = time.process_time() - tic

    print("epoch%d: time=%d, rmse=%f, fitness=%f, elapsed_time=%f" % (e + 1, ts.current_time, ts.tensor.rmse(), ts.tensor.fitness(), toc))