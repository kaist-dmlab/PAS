import pickle
import numpy as np
import pandas as pd
import datetime as dt
import os.path


def removed_default_value(file_path, output_dir, DEFAULT_TRESHOLD=0.4, NIGHT_START_TIMESTAMP_STRING=None, NIGHT_END_TIMESTAMP_STRING=None, forced_save=False, drop_default=True, drop_feature=True, is_full=False):
    input_filename_prefix = file_path.split('.')[0]
    input_filename = '%s.pickle' % input_filename_prefix
    if is_full:
        output_filename = '%s/%s_removed_default_value_full_feature.pickle' % (output_dir, input_filename_prefix.split('/')[-1])
    else:
        output_filename = '%s/%s_removed_default_value.pickle' % (output_dir, input_filename_prefix.split('/')[-1])
    target_feature_list = ['appcat[cat]', 'celllocation_lac[cat]', 'charge_detail_state[cat]', 'headset[cat]', 'ringermode[cat]', 'wifi_conn_state[cat]']
    # target_feature_list = ['celllocation_lac[cat]', 'class']
    if not forced_save and os.path.isfile(output_filename):
        print "ALREADY EXIST: %s" % output_filename
        return pickle.load(open(output_filename)), output_filename
    else:
        print output_filename

    data = pickle.load(open(input_filename))

    default_value_dict = dict()
    # Remove unneccesary feature

    if drop_feature:
        # ['wifi_onoff_state[cat]', 'light[conti]', 'battery_temperature[conti]', 'display_orientation[cat]', 'airplane[cat]',
        # 'bluetooth_onoff_state[cat]', 'celllocation_laccid[cat]', 'celllocation_cid[cat]', 'charge_simple_state[cat]', 'screen_power[cat]', 'battery_level[conti]',
        # 'appname[cat]']:
        for feature in data.keys():
            if feature not in target_feature_list:
                del data[feature]

        # print data.keys()
    if drop_default:
        # Remove default value
        # default_value_dict['airplane[cat]'] = [False]
        # default_value_dict['appname[cat]'] = ['appname_others[cat]']
        # default_value_dict['bluetooth_onoff_state[cat]'] = [False]
        # default_value_dict['accel[conti]'] = ['b[9.35~9.81]', 'a[4.95~9.35]']
        if 'accel[conti]' in data:
            accel_value_counts = data['accel[conti]'].value_counts()
            default_value_dict['accel[conti]'] = list()
            for value in accel_value_counts.index:
                symbol = value[0]
                max_value = [float(minmax_value) for minmax_value in value[2:-1].split('~')][1]
                # print value
                # print max_value
                if max_value < 2.0 or symbol == 'a':
                    default_value_dict['accel[conti]'].append(value)
        default_value_dict['charge_detail_state[cat]'] = ['charge_detail_state_disconnected[cat]']
        default_value_dict['charge_simple_state[cat]'] = [False]
        # default_value_dict['display_orientation[cat]'] = ['display_orientation_portrait[cat]']
        default_value_dict['headset[cat]'] = [False]
        default_value_dict['mobile_conn_state[cat]'] = [False]
        # default_value_dict['ringermode[cat]'] = ['ringermode_normal[cat]'] #, 'ringermode_silent[cat]']
        default_value_dict['screen_power[cat]'] = [False]
        default_value_dict['wifi_conn_state[cat]'] = [False]

        '''
        if 'wifi_onoff_state[cat]' in data:
            wifi_onoff_value_counts = data['wifi_onoff_state[cat]'].value_counts()
            if len(wifi_onoff_value_counts.values) > 1:
                if wifi_onoff_value_counts.values[0] < wifi_onoff_value_counts.values[1]:
                    print "wifi_onoff_state default value: %s" % wifi_onoff_value_counts.index[1]
                    default_value_dict['wifi_onoff_state[cat]'] = [wifi_onoff_value_counts.index[1]]
                else:
                    print "wifi_onoff_state default value: %s" % wifi_onoff_value_counts.index[0]
                    default_value_dict['wifi_onoff_state[cat]'] = [wifi_onoff_value_counts.index[0]]
            else:
                del data['wifi_onoff_state[cat]']
        if 'ringermode[cat]' in data:
            value_counts = data['ringermode[cat]'].value_counts()
            print value_counts
            max_value_num = (None, 0)
            for value in value_counts.index:
                num = value_counts[value]
                if max_value_num[1] < num:
                    max_value_num = (value, num)
            default_value_dict['ringermode[cat]'] = [max_value_num[0]]
        '''

        # Remove major value as default
        # Remove data existing lower than thereshold
        if DEFAULT_TRESHOLD is not None:
            for feature in data.keys():
                if feature in default_value_dict.keys() or feature == 'class':
                    continue
                # print feature
                value_counts = data[feature].value_counts()
                sum_unique_values_without_nan = sum([value for value in value_counts])
                for value in value_counts.index:
                    freq = value_counts[value] / float(sum_unique_values_without_nan)
                    # print ">> %s\t%d\t%f" % (value, value_counts[value], freq)
                    if freq >= DEFAULT_TRESHOLD:
                        if feature not in default_value_dict:
                            default_value_dict[feature] = list()
                        default_value_dict[feature].append(value)

        if 'appcat[cat]' not in default_value_dict:
            default_value_dict['appcat[cat]'] = list()
        default_value_dict['appcat[cat]'].append('UNKNOWN')
        # default_value_dict['appcat[cat]'].append('COMMUNICATION')



    for feature in data:
        if feature == 'class' or feature not in default_value_dict.keys():
            data[feature] = data[feature]
        else:
            # print feature
            value_list = default_value_dict[feature]
            for value in value_list:
                temp = data[feature].replace(to_replace=value, value=np.nan)
                data[feature] = temp

    # Remove sleeping time
    if NIGHT_START_TIMESTAMP_STRING is not None and NIGHT_END_TIMESTAMP_STRING is not None:
        NIGHT_START_TIMESTAMP = dt.datetime.strptime(NIGHT_START_TIMESTAMP_STRING, '%H-%M-%S').time()
        NIGHT_END_TIMESTAMP = dt.datetime.strptime(NIGHT_END_TIMESTAMP_STRING, '%H-%M-%S').time()
        print "NIGHT TIME: %s ~ %s" % (NIGHT_START_TIMESTAMP, NIGHT_END_TIMESTAMP)
        for feature in data:
            # print feature
            temp = dict()
            for timestamp in data[feature].index:
                timestamp_time = timestamp.time()
                if not (timestamp_time >= NIGHT_START_TIMESTAMP and timestamp_time <= NIGHT_END_TIMESTAMP):
                    value = data[feature][timestamp]
                    temp[timestamp] = value
                else:
                    temp[timestamp] = np.nan
            data[feature] = pd.Series(temp)

    '''
    # Remove data existing lower than thereshold
    if DEFAULT_TRESHOLD is not None:
        for feature in data:
            if feature in default_value_dict.keys():
                continue
            else:
                print feature
                value_counts = data[feature].value_counts()
                sum_unique_values_without_nan = sum([value for value in value_counts])
                for value in value_counts.index:
                    freq = value_counts[value] / float(sum_unique_values_without_nan)
                    if freq >= DEFAULT_TRESHOLD:
                        temp = data[feature].replace(to_replace=value, value=np.nan)
                        data[feature] = temp
    '''

    # Remove date that does not contain any data

    timestamp_list = list()
    for feature in data:
        if feature != 'class':
            timestamp_list = data[feature].index

    datestamp_has_features = dict()
    for timestamp in timestamp_list:
        datestamp_str = timestamp.strftime('%Y-%m-%d')
        if datestamp_str not in datestamp_has_features:
            datestamp_has_features[datestamp_str] = set()
        for feature in data:
            if feature != 'class':
                if not pd.isnull(data[feature][timestamp]):
                    datestamp_has_features[datestamp_str].add(feature)

    for feature in data:
        # print "Deleting empty date -- %s" % feature
        deleted_timestamp_list = list()
        for timestamp in data[feature].index:
            datestamp_str = timestamp.strftime('%Y-%m-%d')
            if len(datestamp_has_features[datestamp_str]) == 0:
                # print "Deleted %s-%s" % (datestamp_str, timestamp)
                deleted_timestamp_list.append(timestamp)
        data[feature] = data[feature].drop(deleted_timestamp_list)

    f = open(output_filename, 'w')
    pickle.dump(data, f)
    f.close()

    return data, output_filename


if __name__ == "__main__":
    file_info = "4773aad1530386ed47db7ef0dd06ccb47eed8f63_granularity_60_timeseries_dict_2014-08-24_00-00-00_2014-12-27_00-00-00_reduced_feature"
    file_path = 'dataset/interpolated/%s.pickle' % file_info
    output_dir = 'dataset/interpolated'
    removed_default_value(file_path, output_dir, NIGHT_START_TIMESTAMP_STRING='01-00-00', NIGHT_END_TIMESTAMP_STRING='07-00-00', forced_save=True)
