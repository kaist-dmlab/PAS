import datetime as dt
import pandas as pd
import csv
import pickle
import os.path
import numpy as np

def getSample():
    timestamps = list()
    data = dict()
    feature_data = dict()

    start_time = dt.datetime.strptime('2016-11-14 13:00:00', '%Y-%m-%d %H:%M:%S')
    for day in range(4):
        for minute in range(5):
            timestamps.append(start_time + dt.timedelta(minutes=60 * minute))
        start_time += dt.timedelta(days=1)
    '''
    data_dict = {
        'Location': ['Home', 'Office', 'Cafe', 'Office', 'Cafe',
                     'Home', 'Office', 'Restaurant', 'Office', 'Restaurant',
                     'Home', 'Cafe', 'Home', 'Cafe', 'Home',
                     'Home', 'Restaurant', 'Home', 'Restaurant', 'Home'],
        'App': ['Music', 'Katalk', 'FB', 'Internet', 'Instagram',
                'Music', 'FB', 'Internet', 'Instagram', 'Katalk',
                'Music', 'Internet', 'Instagram', 'Katalk', 'FB',
                'Music', 'Instagram', 'Katalk', 'FB', 'Internet']
    }
    '''
    '''
    data_dict = {
        'Location': ['Home', 'Office', 'Cafe', 'Office', 'Cafe',
                     'Home', 'Cafe', 'Cafe', 'Home', 'Cafe',
                     'Home', 'Office', 'Cafe', 'Office', 'Cafe',
                     'Home', 'Office', 'Cafe', 'Office', 'Cafe'],
        'App': ['Music', 'Katalk', 'FB', 'Internet', 'Instagram',
                'Music', 'FB', 'Internet', 'Instagram', 'Katalk',
                'Music', 'Internet', 'Instagram', 'Katalk', 'FB',
                'Music', 'Instagram', 'Katalk', 'FB', 'Internet']
    }
    '''

    data_dict = {
        'Location': ['Home', 'Home', 'Home', 'Office', 'Office',
                     'Cafe', 'Cafe', 'Cafe', 'Office', 'Cafe',
                     'Cafe', 'Cafe', 'Cafe', 'Office', 'Cafe',
                     'Home', 'Home', 'Home', 'Office', 'Office']
    }

    for feature in data_dict:
        feature_data = dict()
        for i, value in enumerate(data_dict[feature]):
            feature_data[timestamps[i]] = value
        data[feature] = pd.Series(feature_data)

    print data

    return data, timestamps


def readPandasData(file_path):
    # data = pd.read_pickle(file_path)
    data = pickle.load(open(file_path))
    timestamps = data[data.keys()[0]].index

    print data.keys()
    print timestamps

    return data, timestamps


def exportToCSV(data, timestamps, filename, granularity_min, include_class=False, include_null=False, forced_save=False):
    '''
    if (not forced_save) and os.path.isfile(filename):
        print "ALREADY EXIST: %s" % filename
        return
    else:
        print filename
    '''
    print filename
    daily_time_list = set()
    granularity_timestamp_idx = dict()
    tid_to_granualarity_tid = dict()
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, timestamp in enumerate(timestamps):
            if i % 10000 == 0:
                print str(i) + "/" + str(len(timestamps))
            transaction = []
            if include_class:
                if timestamp in data['class']:
                    if data['class'][timestamp] == 1:
                        transaction.append("class:True")
                    else:
                        transaction.append("class:False")
                else:
                    transaction.append("")
            transaction.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            granularity_timestamp = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([timestamp]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
            daily_time = granularity_timestamp.strftime("%H:%M:%S")
            daily_feature = "time_daily:" + str(daily_time)

            daily_time_str = str(daily_time)
            daily_time_list.add(daily_time_str)
            granularity_timestamp_str = str(granularity_timestamp)
            if granularity_timestamp_str not in granularity_timestamp_idx:
                granularity_timestamp_idx[granularity_timestamp_str] = len(granularity_timestamp_idx)
            tid_to_granualarity_tid[i] = granularity_timestamp_idx[granularity_timestamp_str]

            transaction.append(daily_feature)
            feature_list = list(data.keys())
            for feature in feature_list:
                if feature != 'class':
                    if include_null or not pd.isnull(data[feature][timestamp]):
                        if pd.isnull(data[feature][timestamp]):
                            transaction.append("")
                        else:
                            transaction.append(feature + ":" + str(data[feature][timestamp]))
            writer.writerow(transaction)

    daily_time_list = sorted(list(daily_time_list))
    '''
    print daily_time_list
    print tid_to_granualarity_tid

    for i, timestamp in enumerate(timestamps):
        granularity_tid = tid_to_granualarity_tid[i]
        print str(timestamp) + "\t" + str(granularity_tid)
    '''
    return daily_time_list, tid_to_granualarity_tid


def exportClassifierDataToCSV(data, class_name, filename, granularity_min):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        feature_list = list(data.keys())
        attributes = ['timestamp', 'time_daily']
        for feature in feature_list:
            if feature != class_name:
                attributes.append(feature)
        attributes.append(class_name)
        writer.writerow(attributes)

        data[class_name] = data[class_name].dropna()
        timestamps = data[class_name].index
        valid_class_index = list()
        for i, timestamp in enumerate(timestamps):
            if i % 1000 == 0:
                print str(i) + "/" + str(len(timestamps))
            transaction = []
            transaction.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            daily_time = timestamp.strftime("%H:%M:%S")
            granularity_timestamp = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([timestamp]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
            daily_time = granularity_timestamp.strftime("%H:%M:%S")
            daily_feature = "time_daily:" + str(daily_time)
            transaction.append(daily_feature)
            for feature in feature_list:
                if feature != class_name:
                    if pd.isnull(data[feature][timestamp]):
                        transaction.append("")
                    else:
                        transaction.append(feature + ":" + str(data[feature][timestamp]))
            if data[class_name][timestamp] == 1:
                transaction.append("True")
            else:
                transaction.append("False")

            valid_cnt = 0
            for item in transaction:
                if item != "":
                    valid_cnt += 1
            if valid_cnt > 3:
                # print transaction
                valid_class_index.append(timestamp)
                writer.writerow(transaction)
    return valid_class_index


if __name__ == "__main__":
    # data, timestamps = getSample()
    # exportToCSV(data, timestamps, 'dataset/sample/sample2.csv')

    data, timestamps = readPandasData('dataset/interpolated/2dffaa0ee6aca9c50bb0fd4c2e167dcfec6cdfbe_timeseries_dict_2014-08-15_00-00-00_2014-11-14_23-55-00_reduced_feature_removed_default_value.pickle')
    # exportToCSV(data, timestamps, 'dataset/interpolated/4773aad1530386ed47db7ef0dd06ccb47eed8f63_timeseries_dict_2014-08-24_00-00-00_2014-12-26_23-55-00_reduced_feature_removed_default_value.csv', granularity_min = 60, include_class=True, forced_save=True)
    exportClassifierDataToCSV(data, 'class', 'dataset/interpolated/2dffaa0ee6aca9c50bb0fd4c2e167dcfec6cdfbe_timeseries_dict_2014-08-15_00-00-00_2014-11-14_23-55-00_reduced_feature_removed_default_value.csv', granularity_min=30)


    '''
    data, timestamps = readPandasData('dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-05_00-00-00_2013-09-01_23-55-00_reduced_feature_removed_default_value.pickle')
    exportToCSV(data, timestamps, 'dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-05_00-00-00_2013-09-01_23-55-00_reduced_feature_removed_default_value.csv', include_class=True)
    exportClassifierDataToCSV(data, 'class', 'dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-05_00-00-00_2013-09-01_23-55-00_reduced_feature_removed_default_value_including_class.csv')
    '''
    '''
    data, timestamps = readPandasData('dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-05_00-00-00_2013-09-01_23-55-00_reduced_feature.pickle')
    exportToCSV(data, timestamps, 'dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-05_00-00-00_2013-09-01_23-55-00_reduced_feature.csv', include_class=True)
    exportClassifierDataToCSV(data, 'class', 'dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-05_00-00-00_2013-09-01_23-55-00_reduced_feature_including_class.csv')
    '''
