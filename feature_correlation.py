import ruduce_category_features
import removed_default_value
import os.path
import operator
import pickle
from pandas.tseries.offsets import Day
import datetime as dt
import pandas as pd
import numpy as np
import scipy.stats as ss
import sys


def Reindex(file_path, output_dir, term_min, forced_save=False):
    output_filename = output_dir + "/" + file_path.split('.pickle')[0].split('/')[-1] + '_reindexed.pickle'
    if (not forced_save) and os.path.isfile(output_filename):
        print "ALREADY EXIST: %s" % output_filename
        return pickle.load(open(output_filename)), output_filename

    print "Reindxing - %s" % output_filename
    data = pickle.load(open(file_path))
    # No interpolated, no fill strategy
    start_timestamp = None
    end_timestamp = None
    reindexed_data = dict()
    for feature in data.keys():
        # print feature
        timestamp_list = data[feature].index
        if len(timestamp_list) == 0:
            del data[feature]
            continue
        if feature == 'class':
            # Re-arrange at 5min interval (Class feature, floor)
            # To avoid triggered features when ringing (Retinale: Use previous state when ringing)
            rearrange_timestamp_list = pd.DatetimeIndex(((np.floor(pd.DatetimeIndex(timestamp_list).asi8 / (1e9 * 60 * term_min))) * 1e9 * 60 * term_min).astype(np.int64))
            # print timestamp_list[:5]
            # print rearrange_timestamp_list[:5]
            rearrange_timestamp_list = sorted(set(rearrange_timestamp_list))
            reindexed_data[feature] = data[feature].reindex(rearrange_timestamp_list, method='backfill')
            # print data[feature][:5]
            # print reindexed_data[feature][:5]
        else:
            # For find global start and end time
            if start_timestamp is None or start_timestamp > timestamp_list[0]:
                start_timestamp = timestamp_list[0]
            if end_timestamp is None or end_timestamp < timestamp_list[-1]:
                end_timestamp = timestamp_list[-1]

            # Re-arrange at 5min interval (Other feature, Ceil)
            rearrange_timestamp_list = pd.DatetimeIndex(((np.ceil(pd.DatetimeIndex(timestamp_list).asi8 / (1e9 * 60 * term_min))) * 1e9 * 60 * term_min).astype(np.int64))
            # print timestamp_list[:5]
            # print rearrange_timestamp_list[:5]
            rearrange_timestamp_list = sorted(set(rearrange_timestamp_list))
            reindexed_data[feature] = data[feature].reindex(rearrange_timestamp_list, method='ffill')
            # print data[feature][:10]
            # print reindexed_data[feature][:5]


    # Find global start and end time to fix same timerange
    start_timestamp = dt.datetime(start_timestamp.year, start_timestamp.month, start_timestamp.day, 0, 0, 0) + Day(1)
    end_timestamp = dt.datetime(end_timestamp.year, end_timestamp.month, end_timestamp.day, 23, 59, 59) - Day(1)

    sampling_ts = pd.date_range(start=start_timestamp, end=end_timestamp, freq="%smin" % term_min)
    for feature in reindexed_data.keys():
        reindexed_data[feature] = reindexed_data[feature].reindex(sampling_ts)

    SCREEN_FEATURE = 'screen_power[cat]'
    APPCAT_FEATURE = 'appcat[cat]'
    APPNAME_FEATURE = 'appname[cat]'
    # print reindexed_data.keys()
    if SCREEN_FEATURE in reindexed_data:
        screen_data = reindexed_data[SCREEN_FEATURE]
        for timestamp in screen_data.index:
            value = screen_data[timestamp]
            if value != True:
                if APPCAT_FEATURE in reindexed_data.keys():
                    reindexed_data[APPCAT_FEATURE][timestamp] = np.nan
                if APPNAME_FEATURE in reindexed_data.keys():
                    reindexed_data[APPNAME_FEATURE][timestamp] = np.nan

    f = open(output_filename, 'w')
    pickle.dump(reindexed_data, f)
    f.close()

    return reindexed_data, output_filename


def Chisquare(class_data, period_data, class_feature, feature):
    # http://connor-johnson.com/2014/12/31/the-pearson-chi-squared-test-with-python-and-r/
    # http://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix

    class_data = class_data[class_feature]
    feature_data = period_data[feature]
    # Generate confusion matrix
    feature_value_frequency_for_class_value = dict()
    class_value_set = set()
    feature_value_set = set()
    # print "%s\t%s" % (class_feature, feature)
    for timestamp in class_data.index:
        class_value = class_data[timestamp]
        feature_value = feature_data[timestamp]
        if pd.isnull(class_value) or pd.isnull(feature_value):
            continue
        if class_value not in feature_value_frequency_for_class_value:
            feature_value_frequency_for_class_value[class_value] = dict()
        if feature_value not in feature_value_frequency_for_class_value[class_value]:
            feature_value_frequency_for_class_value[class_value][feature_value] = 0
        feature_value_frequency_for_class_value[class_value][feature_value] += 1
        class_value_set.add(class_value)
        feature_value_set.add(feature_value)
        # print "%s\t%s" % (class_value, feature_value)

    # print class_value_set
    # print feature_value_set

    if len(class_value_set) < 2 or len(feature_value_set) < 2:
        return None, None, None

    confusion_matrix = list()
    for class_value in class_value_set:
        row = list()
        for feature_value in feature_value_set:
            freq = feature_value_frequency_for_class_value[class_value].get(feature_value, 0)
            row.append(freq)
        confusion_matrix.append(row)
    confusion_matrix = np.array(confusion_matrix)
    # print confusion_matrix
    # Chisquare, Cramer's V
    chi2, p, dof, ex = ss.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum()
    cramer_v = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

    # Chisquare Test Assumption
    all_cnt = 0
    low_cnt = 0
    for ex_row in ex:
        for ex_value in ex_row:
            all_cnt += 1
            if ex_value < 5:
                low_cnt += 1

    lower_cell_ratio = low_cnt / float(all_cnt)
    # print "Lower Cell Ratio: %f" % lower_cell_ratio
    if lower_cell_ratio >= 0.25:
        return None, None, None

    return chi2, p, cramer_v


def AnalysisFeatureCorrelation(file_info, period_data, class_data, period_feature_set, class_feature_set):

    f = open('correlation_result/%s.txt' % file_info, 'w')

    feature_correlation_list = list()
    for period_idx, period_feature in enumerate(period_feature_set):
        # print "%d/%d - %s" % (class_idx, len(period_feature_set), class_feature)
        for class_feature in class_feature_set:
            if period_feature == class_feature:
                continue
            chi2, p, cramer_v = Chisquare(period_data, class_data, period_feature, class_feature)
            f.write("%s\t%s\t%s\t%s\t%s\n" % (period_feature, class_feature, str(chi2), str(p), str(cramer_v)))
            feature_correlation_list.append((period_feature, class_feature, chi2, p, cramer_v))

    '''
    print data.keys()
    feature_list = data.keys()
    for class_idx, class_feature in enumerate(feature_list):
        print "%d/%d - %s" % (class_idx, len(feature_list), class_feature)
        for feature in feature_list[class_idx + 1:]:
            if class_feature == feature:
                continue
            chi2, p, cramer_v = Chisquare(data, class_feature, feature)
            f.write("%s\t%s\t%s\t%s\t%s\n" % (class_feature, feature, str(chi2), str(p), str(cramer_v)))
    '''
    f.close()
    feature_correlation_list = sorted(feature_correlation_list, key=lambda x: x[4], reverse=True)
    return feature_correlation_list


if __name__ == '__main__':
    dir_path = 'dataset/raw'
    output_dir = 'dataset/raw'

    getall = [[files, os.path.getsize(dir_path + "/" + files)] for files in os.listdir(dir_path)]
    file_info_list = list()
    for file_name, file_size in sorted(getall, key=operator.itemgetter(1)):
        if file_name[-22:] == 'timeseries_dict.pickle':
            file_info_list.append(file_name.split('.')[0])
    print file_info_list
    file_info_list = ["53fd4aede3cfe8a2e7b52f5129488043c6cf6eb6_timeseries_dict"]

    for user_idx, file_info in enumerate(file_info_list[:30]):
        print "%d/%d - %s" % (user_idx, len(file_info_list), file_info)
        AnalysisFeatureCorrelation(file_info, dir_path, output_dir, period_feature_set=set(['celllocation_lac[cat]']))
