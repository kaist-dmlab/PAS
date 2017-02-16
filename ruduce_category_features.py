import pandas as pd
import numpy as np
import pickle
import datetime as dt
import os.path
import pickle
from module.saxpy.saxpy import SAX
from feature_correlation import Reindex

def reduce_catergory_feature(filepath, output_dir, START_TIME_STRING=None, END_TIME_STRING=None, forced_save=False):
    # interpolated_data = pd.read_pickle('dataset/sample/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict_2013-08-01_00:00:00_2013-08-07_23:59:55.pickle')
    # interpolated_data = pd.read_pickle('dataset/interpolated/44acb24475fb0c0b84cea3f4926a36cddbbbff0f_timeseries_dict.pickle')
    # interpolated_data = pd.read_pickle(filepath)

    interpolated_data, filepath = Reindex(filepath, output_dir, term_min=5)

    IS_ALLDATA = False

    if START_TIME_STRING is None and END_TIME_STRING is None:
        IS_ALLDATA = True

    if START_TIME_STRING is None:
        START_TIMESTAMP = interpolated_data[interpolated_data.keys()[0]].index[0]
        START_TIME_STRING = START_TIMESTAMP.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        START_TIMESTAMP = dt.datetime.strptime(START_TIME_STRING, '%Y-%m-%d_%H-%M-%S')

    if END_TIME_STRING is None:
        END_TIMESTAMP = interpolated_data[interpolated_data.keys()[0]].index[-1]
        END_TIME_STRING = END_TIMESTAMP.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        END_TIMESTAMP = dt.datetime.strptime(END_TIME_STRING, '%Y-%m-%d_%H-%M-%S')

    '''
    START_TIME_STRING = '2013-08-05_00-00-00'
    START_TIMESTAMP = dt.datetime.strptime(START_TIME_STRING, '%Y-%m-%d_%H-%M-%S')
    END_TIME_STRING = '2013-09-01_23-55-00'
    END_TIMESTAMP = dt.datetime.strptime(END_TIME_STRING, '%Y-%m-%d_%H-%M-%S')
    '''
    print START_TIMESTAMP
    print END_TIMESTAMP

    output_filename = output_dir + "/" + filepath.split('.pickle')[0].split('/')[-1] + '_%s_%s_reduced_feature.pickle' % (START_TIME_STRING, END_TIME_STRING)
    if (not forced_save) and os.path.isfile(output_filename):
        print "ALREADY EXIST: %s" % output_filename
        return pickle.load(open(output_filename)), output_filename
    else:
        print output_filename

    # VALID_FEATURE_SET = interpolated_data.keys()
    # print len(VALID_FEATURE_SET)

    if 'celllocation_laccid[cat]' in interpolated_data:
        interpolated_data['celllocation_lac[cat]'] = pd.Series(interpolated_data['celllocation_laccid[cat]'])
        value_counts = interpolated_data['celllocation_lac[cat]'].value_counts()
        for value in value_counts.index:
            temp = interpolated_data['celllocation_lac[cat]'].replace(to_replace=value, value=value.split('.')[0])
            interpolated_data['celllocation_lac[cat]'] = temp

        interpolated_data['celllocation_cid[cat]'] = pd.Series(interpolated_data['celllocation_laccid[cat]'])
        value_counts = interpolated_data['celllocation_cid[cat]'].value_counts()
        for value in value_counts.index:
            temp = interpolated_data['celllocation_cid[cat]'].replace(to_replace=value, value=value.split('.')[1])
            interpolated_data['celllocation_cid[cat]'] = temp

    for feature_idx, feature in enumerate(interpolated_data.keys()):
        # print "%d/%d" % (feature_idx, len(interpolated_data.keys()))
        if '[conti]' in feature:
            # print feature
            if feature == 'accel[conti]':
                for timestamp in interpolated_data[feature].index:
                    if not pd.isnull(interpolated_data[feature][timestamp]):
                        interpolated_data[feature][timestamp] -= 9.8
                        interpolated_data[feature][timestamp] = abs(interpolated_data[feature][timestamp])
            data_notnull = interpolated_data[feature].dropna()
            if feature == 'accel[conti]':
                sax = SAX(len(data_notnull.index), 5, 1e-6)
            else:
                sax = SAX(len(data_notnull.index), 5, 1e-6)
            values = data_notnull.values
            (letters, indices) = sax.to_letter_rep(values)
            temp_data = pd.Series()
            letter_min_max = dict()
            for i, timestamp in enumerate(data_notnull.index):
                #print "%s\t%f\t%s" % (timestamp, data_notnull[timestamp], letters[i])
                letter = letters[i]
                value = data_notnull[timestamp]
                if letter not in letter_min_max:
                    letter_min_max[letter] = [9999999, 0]
                if value < letter_min_max[letter][0]:
                    letter_min_max[letter][0] = value
                if value > letter_min_max[letter][1]:
                    letter_min_max[letter][1] = value
                temp_data[timestamp] = letters[i]

            for letter in letter_min_max:
                new_letter = "%s[%.2f~%.2f]" % (letter, letter_min_max[letter][0], letter_min_max[letter][1])
                temp_data = temp_data.replace(to_replace=letter, value=new_letter)
            temp_data = temp_data.reindex(interpolated_data[feature].index)
            interpolated_data[feature] = temp_data

    VALID_FEATURE_LIST = list()

    for feature in interpolated_data.keys():
        if '[cat]' in feature:
            if 'celllocation_laccid' not in feature or feature == 'celllocation_laccid[cat]':
                VALID_FEATURE_LIST.append(feature)
        elif '[disc]' in feature and 'sms_unread_count[disc]' != feature:
            VALID_FEATURE_LIST.append(feature)
        elif '[conti]' in feature:
            VALID_FEATURE_LIST.append(feature)
        elif 'class' == feature:
            VALID_FEATURE_LIST.append(feature)

    print "Valid feature list: %s" % VALID_FEATURE_LIST

    reduced_feature_data = dict()
    reduced_features = set()
    for i, feature in enumerate(sorted(interpolated_data.keys())):
        print "%s - (%d/%d)" % (feature, i, len(interpolated_data.keys()))
        if feature not in VALID_FEATURE_LIST:
            continue
        reduced_name = None
        # if 'appcat' in feature:
        #     reduced_name = 'appcat[cat]'
        # elif 'appname' in feature:
        #     reduced_name = 'appname[cat]'
        # elif 'celllocation_laccid' in feature:
        #     reduced_name = 'celllocation_laccid[cat]'
        if 'charge_detail_state' in feature:
            reduced_name = 'charge_detail_state[cat]'
        elif 'display_orientation' in feature:
            reduced_name = 'display_orientation[cat]'
        elif 'ringermode' in feature:
            reduced_name = 'ringermode[cat]'
        else:
            # print feature
            if feature in ['appcat[cat]', 'appname[cat]', 'celllocation_laccid[cat]', 'celllocation_lac[cat]', 'celllocation_cid[cat]']:
                value_counts = interpolated_data[feature].value_counts()
                # print value_counts
                for value in value_counts.index[5:]:
                    temp = interpolated_data[feature].replace(to_replace=value, value=np.nan)
                    interpolated_data[feature] = temp

            if '[disc]' in feature:
                value_counts = interpolated_data[feature].value_counts()
                values = value_counts.index

                for value in values:
                    disc_value = None
                    if value >= 0 and value <= 0.2:
                        disc_value = 0.2
                    elif value > 0.2 and value <= 0.4:
                        disc_value = 0.4
                    elif value > 0.4 and value <= 0.6:
                        disc_value = 0.6
                    elif value > 0.6 and value <= 0.8:
                        disc_value = 0.8
                    elif value > 0.8 and value <= 1.0:
                        disc_value = 1.0
                    temp = interpolated_data[feature].replace(to_replace=value, value=disc_value)
                    interpolated_data[feature] = temp

            temp = dict()
            if IS_ALLDATA:
                reduced_feature_data[feature] = pd.Series(interpolated_data[feature])
            else:
                for timestamp in interpolated_data[feature].index:
                    if timestamp >= START_TIMESTAMP and timestamp <= END_TIMESTAMP:
                        value = interpolated_data[feature][timestamp]
                        temp[timestamp] = value
                reduced_feature_data[feature] = pd.Series(temp)
            # print "----"
            # print interpolated_data[feature].value_counts()
            # print reduced_feature_data[feature].value_counts()
            # print "----"

        if reduced_name is not None:
            reduced_features.add(reduced_name)

        if reduced_name is not None:
            if reduced_name not in reduced_feature_data:
                reduced_feature_data[reduced_name] = pd.Series()
            reduced_data = reduced_feature_data[reduced_name]
            # print feature
            # print interpolated_data[feature].value_counts()

            for timestamp in interpolated_data[feature].index:
                if IS_ALLDATA or timestamp >= START_TIMESTAMP and timestamp <= END_TIMESTAMP:
                    value = interpolated_data[feature][timestamp]
                    if timestamp not in reduced_data:
                        reduced_data[timestamp] = np.nan
                    if value is True:
                        reduced_data[timestamp] = feature

    print reduced_feature_data.keys()

    f = open(output_filename, 'w')
    pickle.dump(reduced_feature_data, f)
    f.close()

    return reduced_feature_data, output_filename


if __name__ == "__main__":
    file_info = "4773aad1530386ed47db7ef0dd06ccb47eed8f63_granularity_60_timeseries_dict"
    file_path = 'dataset/interpolated/%s.pickle' % file_info
    output_dir = 'dataset/interpolated'
    reduce_catergory_feature(file_path, output_dir, forced_save=True)
