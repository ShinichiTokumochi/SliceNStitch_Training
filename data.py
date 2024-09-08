from sklearn import preprocessing
import pandas as pd
from datetime import timedelta

def get_events(file_name: str, td: timedelta, category_labels: list[str], time_label: str):
    #events = pd.read_csv(file_name, nrows=100000)
    events = pd.read_csv(file_name)
    events = events.dropna(how='any')
    events[time_label] = pd.to_datetime(events[time_label])
    events = events.sort_values(time_label)
    events[time_label] = (events[time_label] - events[time_label].min()) // td

    oe = preprocessing.OrdinalEncoder()
    events[category_labels] = oe.fit_transform(events[category_labels]).astype(int)
    return events