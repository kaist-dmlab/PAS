import datetime as dt
import module.apriori.apriori as Apriori
import pickle
import ruduce_category_features
import removed_default_value
import dictToCsv
import os.path
import operator
import numpy as np
import pandas as pd
import AssociationExperiment2 as AssociationExperiment
IS_DEBUG = False


def debug_print(str):
    if IS_DEBUG:
        print str


def getTidsHavePatternList(patterns, itemTimestampIndexDict):
    tids_list = ()
    for pattern_set in patterns:
        tsSet = set()
        for i, pattern in enumerate(pattern_set):
            patternTsSet = itemTimestampIndexDict[pattern]
            patternTsSet = set([tid_to_granualarity_tid[ts] for ts in patternTsSet])
            if i == 0:
                tsSet = patternTsSet
            else:
                tsSet = tsSet & patternTsSet
        tids_list += (list(tsSet),)
    return tids_list


def getSupportFromTids(tids, timestampList):
    timestampSet = set()
    for tid in tids:
        tid = tid[0]
        timestamp = timestampList[tid]
        # timestamp = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        timestamp = timestamp.strftime('%Y-%m-%d')
        timestampSet.add(timestamp)
    return len(timestampSet) / float(len(dateStampSet))


def findIndexWithKey(key, row):
    for idx, attribute in enumerate(row):
        if key in attribute:
            return idx
    return None


def isValidateCombanation(min_sup, left1, left2, pruned_candidate, able_candidate, timestampList, itemTimestampIndexDict, daily_time_idx_dict, max_gap=1, left1_for_right1=None, left2_for_right2=None, left_tids_for_right=None, is_right=False):
    global test_combination_cnt
    global pattern_all_tids

    debug_print("Join Check\t" + str(left1) + "\t" + str(left2))
    if not is_right:
        sequence_key = ('left', left1, left2, None)
    else:
        # If this check expansion of right, it have to know gap information from merged left.
        if left_tids_for_right is not None and len(left_tids_for_right) > 0:
            left_tids = left_tids_for_right[0]
            idx_delta_list = list() # Gap array, If <{1pm}{2pm}{4pm}>, idx_delta_list is [1,2].
            prev_tid = None
            for tid in left_tids:
                if prev_tid is not None:
                    idx_delta_list.append(tid - prev_tid)
                prev_tid = tid
            sequence_key = ('right', left1, left2, tuple(idx_delta_list))
        else:
            sequence_key = ('right', left1, left2, None)

    is_sequence = False

    test_combination_cnt += 1

    # Caching
    if pruned_candidate.get(sequence_key, -1) == 1:
        return is_sequence, None, None, None
    if able_candidate.get(sequence_key, -1) != -1:
        is_sequence = True
        new_pattern = able_candidate[sequence_key][0]
        output_tids = able_candidate[sequence_key][1]
        return is_sequence, new_pattern, output_tids, idx_delta_list

    debug_print("LEFT1: " + str(left1))
    debug_print("LEFT2: " + str(left2))

    # Check joinable
    pattern_len = len(left1)
    if pattern_len == 1:
        left1_time_daily_index = findIndexWithKey('time_daily', left1[0])
        left2_time_daily_index = findIndexWithKey('time_daily', left2[0])

        if left1_time_daily_index is not None and left2_time_daily_index is not None:
            if left1[0][left1_time_daily_index] == left2[0][left2_time_daily_index]:
                return is_sequence, None, None, None
        new_pattern = left1 + left2
    else:
        if left1[-(pattern_len - 1):] == left2[:pattern_len - 1]:
            new_pattern = left1 + left2[-1:]
        else:
            return is_sequence, None, [], None

    left1_left2_for_right1_right2 = None
    if left1_for_right1 is not None and left2_for_right2 is not None:
        if pattern_len == 1:
            left1_left2_for_right1_right2 = left1_for_right1 + left2_for_right2
        else:
            if left1_for_right1[-(pattern_len - 1):] == left2_for_right2[:pattern_len - 1]:
                left1_left2_for_right1_right2 = left1_for_right1 + left2_for_right2[-1:]

    # When joining left find tids of patterns and gap array
    # When joining right get tids of pattern if cached or find tids and cache.
    if new_pattern not in pattern_all_tids:
        all_tids = list()
        valid_pattern_idx_list = []
        idx_delta_list = []
        prev_idx = None
        pattern_idx_dict = dict()
        # If <{A}{B}{C}>
        for i, pattern in enumerate(new_pattern):
            # {A}=1, {B}=2, {C}=3
            if pattern not in pattern_idx_dict:
                pattern_idx_dict[pattern] = len(pattern_idx_dict)
            pattern_idx = pattern_idx_dict[pattern]
            # Find tids of {A}
            tids = getTidsHavePatternList(patterns=(pattern,), itemTimestampIndexDict=itemTimestampIndexDict)
            tids = tids[0]
            tids_with_pattern = [(t, pattern_idx) for t in tids]

            all_tids += tids_with_pattern
            valid_pattern_idx_list.append(pattern_idx) # At last, this is [1,2,3]. it means <{A}{B}{C}>

            # When joining left, check gap between previous pattern and current pattern
            if not is_right:
                if prev_idx is not None:
                    # print new_pattern[prev_idx]
                    # print left1_for_right1
                    # print left2_for_right2
                    # print new_pattern[i]
                    if left1_left2_for_right1_right2 is not None: # This state is not rechable code.
                        current_pattern_time = left1_left2_for_right1_right2[i][findIndexWithKey('time_daily', left1_left2_for_right1_right2[i])].split('time_daily:')[1]
                        prev_pattern_time = left1_left2_for_right1_right2[prev_idx][findIndexWithKey('time_daily', left1_left2_for_right1_right2[prev_idx])].split('time_daily:')[1]
                    else:
                        current_pattern_time = new_pattern[i][findIndexWithKey('time_daily', new_pattern[i])].split('time_daily:')[1]
                        prev_pattern_time = new_pattern[prev_idx][findIndexWithKey('time_daily', new_pattern[prev_idx])].split('time_daily:')[1]

                    current_pattern_time_idx = daily_time_idx_dict[current_pattern_time]
                    prev_pattern_time_idx = daily_time_idx_dict[prev_pattern_time]

                    delta = current_pattern_time_idx - prev_pattern_time_idx
                    if delta < 0:
                        delta = len(daily_time_idx_dict) + delta

                    if delta > max_gap:
                        pruned_candidate[sequence_key] = 1
                        return is_sequence, None, None, None

                    idx_delta_list.append(delta)
                prev_idx = i

        all_tids = sorted(list(set(all_tids)))
        if is_right:
            pattern_all_tids[new_pattern] = all_tids
        debug_print("ALL TIDS:" + str(all_tids))
        debug_print("PATTERN IDX LIST: " + str(valid_pattern_idx_list))
        debug_print("PATTERN IDX DELTA LIST: " + str(idx_delta_list))
    else:
        all_tids = pattern_all_tids[new_pattern]

    if is_right and (left_tids_for_right is None or len(left_tids_for_right) == 0):
        return is_sequence, None, None, None

    # When joining left, find list of tids. For example, In the ABCABDABC, the result of {A}{B}{C} is [(1,2,3), (7,8,9)]
    if not is_right:
        check_pattern_idx = 0
        prev_tid_pattern_idx = None
        candidate_tids = tuple()
        output_tids = list()
        for start in range(len(all_tids)):
            for i, tid_with_pattern_idx in enumerate(all_tids[start:]):
                tid = tid_with_pattern_idx[0]
                pattern_idx = tid_with_pattern_idx[1]
                # Check that there(pattern_idx) is expected pattern(check_pattern_idx).
                if valid_pattern_idx_list[check_pattern_idx] == pattern_idx:
                    # First pattern or pattern exists in expected gap
                    if prev_tid_pattern_idx is None or (tid - prev_tid_pattern_idx[0]) == (idx_delta_list[check_pattern_idx - 1]):
                        prev_tid_pattern_idx = (tid, pattern_idx)
                        candidate_tids += (tid,)
                        check_pattern_idx += 1
                        if check_pattern_idx == len(valid_pattern_idx_list):
                            output_tids.append(candidate_tids)
                            candidate_tids = tuple()
                            check_pattern_idx = 0
                            prev_tid_pattern_idx = None
                            break
                    else:
                        # If gap is larger than expected, restart find at first pattern
                        if (tid - prev_tid_pattern_idx[0]) > (idx_delta_list[check_pattern_idx - 1]): # tid != prev_tid_pattern_idx[0] and i < len(all_tids) - 1 and tid != all_tids[i + 1][0]:
                            candidate_tids = tuple()
                            check_pattern_idx = 0
                            prev_tid_pattern_idx = None
                            break
                # Or, initialize to find list of tids
                else:
                    if check_pattern_idx == 0:
                        candidate_tids = tuple()
                        check_pattern_idx = 0
                        prev_tid_pattern_idx = None
                        break
        debug_print("OUTPUT TIDS:" + str(output_tids))

        # Calculating support
        if getSupportFromTids(output_tids, timestampList) >= min_sup:
            is_sequence = True
    # When joining rights
    else:
        pattern_idx_dict = dict()
        valid_pattern_idx_list = list()
        for pattern in new_pattern:
            if pattern not in pattern_idx_dict:
                pattern_idx_dict[pattern] = len(pattern_idx_dict)
            pattern_idx = pattern_idx_dict[pattern]
            valid_pattern_idx_list.append(pattern_idx)
        '''
        all_tids_dict = dict()
        for tid_info in all_tids:
            all_tids_dict[tid_info] = 1
        is_sequence = True
        output_tids = list()
        for left_tids in left_tids_for_right:
            is_include = False
            for j, left_tid in enumerate(left_tids):
                check_pattern_idx = pattern_idx_list[j]
                if all_tids_dict.get((left_tid, check_pattern_idx), -1) != -1:
                    is_include = True
                else:
                    is_include = False
                    break
            if is_include is True:
                output_tids.append(left_tids)
        '''
        # Get gap information from joined left
        left_tids = left_tids_for_right[0]
        idx_delta_list = list()
        prev_tid = None
        for tid in left_tids:
            if prev_tid is not None:
                idx_delta_list.append(tid - prev_tid)
            prev_tid = tid
        # print idx_delta_list
        # print left_tids

        check_pattern_idx = 0
        prev_tid_pattern_idx = None
        candidate_tids = tuple()
        output_tids = list()
        for start in range(len(all_tids)):
            for i, tid_with_pattern_idx in enumerate(all_tids[start:]):
                tid = tid_with_pattern_idx[0]
                pattern_idx = tid_with_pattern_idx[1]
                if valid_pattern_idx_list[check_pattern_idx] == pattern_idx:
                    if prev_tid_pattern_idx is None or (tid - prev_tid_pattern_idx[0]) == (idx_delta_list[check_pattern_idx - 1]):
                        prev_tid_pattern_idx = (tid, pattern_idx)
                        candidate_tids += (tid,)
                        check_pattern_idx += 1
                        if check_pattern_idx == len(valid_pattern_idx_list):
                            output_tids.append(candidate_tids)
                            candidate_tids = tuple()
                            check_pattern_idx = 0
                            prev_tid_pattern_idx = None
                            break
                    else:
                        if (tid - prev_tid_pattern_idx[0]) > (idx_delta_list[check_pattern_idx - 1]): # tid != prev_tid_pattern_idx[0] and i < len(all_tids) - 1 and tid != all_tids[i + 1][0]:
                            candidate_tids = tuple()
                            check_pattern_idx = 0
                            prev_tid_pattern_idx = None
                            break
                else:
                    if check_pattern_idx == 0:
                        candidate_tids = tuple()
                        check_pattern_idx = 0
                        prev_tid_pattern_idx = None
                        break

        if getSupportFromTids(output_tids, timestampList) >= min_sup:
            is_sequence = True

    if not is_sequence:
        pruned_candidate[sequence_key] = 1
    else:
        able_candidate[sequence_key] = (new_pattern, output_tids)

    return is_sequence, new_pattern, output_tids, idx_delta_list


def concatLeftCandidate(k, left_candidates, left1, itemTimestampIndexDict, min_sup, min_conf, timestampList, nextRules, validRules, daily_time_idx_dict, max_gap):
    for left2_index in range(len(left_candidates)):
        debug_print("--------------------------------")
        debug_print("Given Left1: " + str(left1))

        left2 = left_candidates[left2_index]

        # Check to be able to merge left
        left_is_sequence, new_left, left_tids, left_gap_list = isValidateCombanation(min_sup, left1, left2, pruned_candidate, able_candidate, timestampList, itemTimestampIndexDict, daily_time_idx_dict, max_gap=max_gap)

        # Pruning rule 1 (Remove right combination when left combination is invalid.)
        if left_is_sequence:
            for right1 in left_right_dict[left1]:
                right2_candidates = set(left_right_dict[left2])

                for right2 in right2_candidates:
                    # Check to be able to merge right
                    right_is_sequence, new_right, right_tids, left_gap_list = isValidateCombanation(min_sup, right1, right2, pruned_candidate, able_candidate, timestampList, itemTimestampIndexDict, daily_time_idx_dict, left1_for_right1=left1, left2_for_right2=left2, left_tids_for_right=left_tids, is_right=True, max_gap=max_gap)
                    debug_print("Right Symbol Check: " + str(right_is_sequence))

                    if not right_is_sequence and right_tids is not None:
                        debug_print("Delete left that is only one right from that right")
                        left1_list_from_right = right_left_dict[right1]
                        deleted_left1_list = list()
                        for deleted_left_candidate1 in left1_list_from_right:
                            if deleted_left_candidate1 != left1 and len(left_right_dict[deleted_left_candidate1]) == 1:
                                deleted_left1_list.append(deleted_left_candidate1)

                        left2_list_from_right = right_left_dict[right2]
                        deleted_left2_list = list()
                        for deleted_left_candidate2 in left2_list_from_right:
                            if deleted_left_candidate2 != left2 and len(left_right_dict[deleted_left_candidate2]) == 1:
                                deleted_left2_list.append(deleted_left_candidate2)

                        # Remove left combination that has only one right that is uncombinatable right set.
                        for deleted_left1 in deleted_left1_list:
                            for deleted_left2 in deleted_left2_list:
                                pattern_len = len(deleted_left1)
                                if pattern_len == 1:
                                    left1_time_daily_index = findIndexWithKey('time_daily', deleted_left1[0])
                                    left2_time_daily_index = findIndexWithKey('time_daily', deleted_left2[0])
                                    new_pattern = left1 + left2
                                else:
                                    if left1[-(pattern_len - 1):] == left2[:pattern_len - 1]:
                                        new_pattern = left1 + left2[-1:]
                                    else:
                                        continue

                                idx_delta_list = []
                                prev_idx = None
                                pattern_idx_dict = dict()
                                for i, pattern in enumerate(new_pattern):
                                    if prev_idx is not None:
                                        current_pattern_time = new_pattern[i][findIndexWithKey('time_daily', new_pattern[i])].split('time_daily:')[1]
                                        prev_pattern_time = new_pattern[prev_idx][findIndexWithKey('time_daily', new_pattern[prev_idx])].split('time_daily:')[1]

                                        current_pattern_time_idx = daily_time_idx_dict[current_pattern_time]
                                        prev_pattern_time_idx = daily_time_idx_dict[prev_pattern_time]

                                        delta = current_pattern_time_idx - prev_pattern_time_idx
                                        if delta < 0:
                                            delta = len(daily_time_idx_dict) + delta
                                        idx_delta_list.append(delta)
                                    prev_idx = i

                                if left_gap_list == idx_delta_list:
                                    debug_print(str(deleted_left1) + "\t" + str(deleted_left2))
                                    pruned_candidate[('left', deleted_left1, deleted_left2, tuple(idx_delta_list))] = 1
                                    debug_print("DELETE LEFT WHERE ONLT RIGHT FROM RIGHT" + "\t" + str(deleted_left1) + "\t" + str(deleted_left2))

                        # # Pruning rule 3 (Remove left pair that is same with uncombinatable right pair)
                        # pruned_candidate[(right1, right2)] = 1
                        debug_print("Remove uncombinatable right pair in left side")
                    # Caculating support of new rule `JoinedLeft -> JoinedRight`
                    if left_is_sequence and right_is_sequence:
                        debug_print(">> Rule")

                        new_both_tids = list(set(left_tids) & set(right_tids))

                        left_support = getSupportFromTids(tids=left_tids, timestampList=timestampList)
                        right_support = getSupportFromTids(tids=right_tids, timestampList=timestampList)
                        support = getSupportFromTids(tids=new_both_tids, timestampList=timestampList)
                        confidence = support / left_support

                        rule_info = (new_left, new_right, left_tids, new_both_tids, support, confidence, right_tids, left_support, right_support)  # , new_left_symbol, new_right_symbol)

                        if support >= min_sup and confidence >= min_conf:
                            nextRules.append(rule_info)
                            debug_print("\t".join(str(element) for element in rule_info))
                            if (left1, right1) in unique_candidate_rules:
                                del unique_candidate_rules[(left1, right1)]
                            if (left2, right2) in unique_candidate_rules:
                                del unique_candidate_rules[(left2, right2)]
                            validRules.append(rule_info)
                        debug_print("-------------")


def getClassFromFile(filename):
    f = open(filename)
    tid = 0
    class_list = list()
    for line in f:
        line = line.split('\n')[0].strip().rstrip(',')
        class_value = line.split(',')[0]
        if class_value != "":
            timestamp = line.split(',')[1]
            class_list.append((class_value, tid, timestamp))
        tid += 1
    f.close()
    return class_list


dir_path = 'dataset/KAIST'
output_dir = dir_path
rules_dir = 'rules'
granularity_min = 60
MIN_DATE_LENGTH = 7 # In the Device Analyzer 30, In the KAIST 10

max_gap = 1
min_sup = 2.0 / 10.0
min_conf = 2.0 / 10.0

# Sorting files as ascending file size
getall = [[files, os.path.getsize(dir_path + "/" + files)] for files in os.listdir(dir_path)]
file_info_list = list()
for file_name, file_size in sorted(getall, key=operator.itemgetter(1)):
    if file_name[-22:] == 'timeseries_dict.pickle':
        file_info_list.append(file_name.split('.')[0])

print file_info_list
print len(file_info_list)

for user_idx, file_info in enumerate(file_info_list):
    print "%d/%d - %s" % (user_idx, len(file_info_list), file_info)

    file_path = dir_path + '/%s.pickle' % file_info
    # Change binary feature to category feature 
    data, file_path = ruduce_category_features.reduce_catergory_feature(file_path, output_dir)
    print data.keys()
    # Get full data (not removing default value)
    full_data, _ = removed_default_value.removed_default_value(file_path, output_dir, NIGHT_START_TIMESTAMP_STRING='01-00-00', NIGHT_END_TIMESTAMP_STRING='07-00-00', drop_default=False, drop_feature=False, is_full=True)
    print full_data.keys()
    # Get data removed default data
    data, file_path = removed_default_value.removed_default_value(file_path, output_dir, NIGHT_START_TIMESTAMP_STRING='01-00-00', NIGHT_END_TIMESTAMP_STRING='07-00-00')

    timestamps = data[data.keys()[0]].index
    if 'class' not in full_data:
        print "!!! NO CLASS !!!"
        # log.write('%s\tNO_CLASS\n' % file_info)
        continue

    # Set to run Association Rule #
    csv_file = output_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_%d.csv' % (granularity_min)
    daily_time_list, tid_to_granualarity_tid = dictToCsv.exportToCSV(data, timestamps, csv_file, include_class=False, granularity_min=granularity_min)
    class_list = getClassFromFile(csv_file)
    inFile = Apriori.dataFromFile(csv_file, hasClass=False)
    print "Complete Read CSV for Apriori"
    # End Association Rule#

    ''' START GETTING MCPP '''

    if not os.path.isdir('entropy_result'):
        os.mkdir('entropy_result')

    f_entropy_result_pickle = open('entropy_result/%s_pattern_pair_entropy_%.2f_%.2f.pickle' % (file_info, min_sup, min_conf), 'w')
    f_entropy_result = open('entropy_result/%s_pattern_pair_entropy_%.2f_%.2f.txt' % (file_info, min_sup, min_conf), 'w')
    f_entropy_avg_result = open('entropy_result/%s_class_avg_entropy_%.2f_%.2f.txt' % (file_info, min_sup, min_conf), 'w')
    pattern_pair_entropy_result_list = list()

    test_combination_cnt = 0
    pattern_all_tids = dict()

    # All Rules
    ALL_RULE_PATH = rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_MCPP_%d_%.2f_%.2f.pickle' % (granularity_min, min_sup, min_conf)
    # Remove superset rules (If there are <{A},{B}>-><{C},{D}>, <{A}>-><{C}>, remove <{A}>-><{C}>)
    ALL_UNIQUE_PATH = rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_UNIQUE_MCPP_%d_%.2f_%.2f.pickle' % (granularity_min, min_sup, min_conf)
    APRIORI_RESULT_PATH = rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_MCPP_1_%d_%.2f_%.2f.pickle' % (granularity_min, min_sup, min_conf)

    existRules = False

    if os.path.exists(ALL_RULE_PATH) and os.path.exists(ALL_UNIQUE_PATH) and os.path.exists(APRIORI_RESULT_PATH):
        print "ALREADY EXIST: " + str(ALL_RULE_PATH)
        print "ALREADY EXIST: " + str(ALL_UNIQUE_PATH)

        all_unique_rule_list = pickle.load(open(ALL_UNIQUE_PATH))
        all_rule_list = pickle.load(open(ALL_RULE_PATH))
        items, rules_1, itemTimestampIndexDict, timestampList = pickle.load(open(APRIORI_RESULT_PATH))

        dateStampSet = set()

        for timestamp in timestampList:
                timestamp = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                dateStampSet.add(timestamp)

        existRules = True

    if not existRules:
        items, rules_1, itemTimestampIndexDict, timestampList = Apriori.runApriori(inFile, min_sup, min_conf, tid_to_granualarity_tid, granularity_min)

        # Change time interval to granularity_min (If granularity_min=30, 11:27 -> 11:30)
        granularityTimestampList = set()
        for timestamp in timestampList:
            granularity_timestamp = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([timestamp]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
            granularityTimestampList.add(granularity_timestamp)
        granularityTimestampList = sorted(granularityTimestampList)

        daily_time_idx_dict = dict()
        for i, daily_time in enumerate(daily_time_list):
            daily_time_idx_dict[daily_time] = i
        # Apriori.printResults(items, rules)

        print "# Rules (%d sequences): " % 1 + str(len(rules_1))

        dateStampSet = set()

        for timestamp in timestampList:
                timestamp = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                dateStampSet.add(timestamp)

        print "# DATE: %d" % len(dateStampSet)

        if len(dateStampSet) < MIN_DATE_LENGTH:
            print "!!! NOT ENOUGH DATES !!! %d" % (len(dateStampSet))
            # log.write('%s\tNOT_ENOGH_DATE(%d)\n' % (file_info, len(dateStampSet)))
            continue

        f = open(APRIORI_RESULT_PATH, 'w')
        f_all_rule = open(ALL_RULE_PATH, 'w')
        f_all_unique_rule = open(ALL_UNIQUE_PATH, 'w')
        f_pattern = open(rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_MCPP_%d.txt' % granularity_min, 'w')
        f_unique_pattern = open(rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_UNIQUE_MCPP_%d.txt' % granularity_min, 'w')

        rules_1_result = (items, rules_1, itemTimestampIndexDict, timestampList)
        pickle.dump(rules_1_result, f)
        f.close()

        rules = list()
        left_right_dict = dict()
        right_left_dict = dict()
        left1_candidates = dict()
        right1_candidate = dict()
        unique_candidate_rules = dict()
        all_rule_list = list()

        print "Building LEFT1, RIGHT1 relationship"

        # Fill pattern information
        all_mcpp_1 = list()
        for i, (rule, confidence, support) in enumerate(rules_1):
            if rule[0] != (('*:*',),):
                left = (rule[0],)
                right = (rule[1],)
            else:
                left = rule[0]
                right = rule[1]
            both_tids = set()
            left_tids = set()
            right_tids = set()

            for i, item in enumerate(rule[0]):
                tids = itemTimestampIndexDict[item]
                tids = set([tid_to_granualarity_tid[tid] for tid in tids])
                if i == 0:
                    left_tids = tids
                else:
                    left_tids = left_tids & tids

            both_tids = left_tids
            for i, item in enumerate(rule[1]):
                tids = itemTimestampIndexDict[item]
                tids = set([tid_to_granualarity_tid[tid] for tid in tids])
                if i == 0:
                    right_tids = tids
                else:
                    right_tids = right_tids & tids

            both_tids = left_tids & right_tids

            left_support = getSupportFromTids(tids=[[tid] for tid in left_tids], timestampList=granularityTimestampList)
            right_support = getSupportFromTids(tids=[[tid] for tid in right_tids], timestampList=granularityTimestampList)

            rule = (left, right, list(left_tids), list(both_tids), support, confidence, list(right_tids), left_support, right_support)
            all_mcpp_1.append(rule)

        # Get closed patterns (If {A,B} -> {C} and {A} -> {C}, then remove {A} -> {C})
        removed_mcpp_list = list()
        for candidate_non_closed_mcpp in all_mcpp_1:
            for candidate_closed_mcpp in all_mcpp_1:
                if candidate_non_closed_mcpp == candidate_closed_mcpp:
                    continue
                non_closed_left = set(candidate_non_closed_mcpp[0][0])
                non_closed_right = set(candidate_non_closed_mcpp[1][0])
                closed_left = set(candidate_closed_mcpp[0][0])
                closed_right = set(candidate_closed_mcpp[1][0])

                if not (non_closed_left.issubset(closed_left) and non_closed_right.issubset(closed_right)):
                    continue
                if set(candidate_non_closed_mcpp[2]) != set(candidate_closed_mcpp[2]) or set(candidate_non_closed_mcpp[3]) != set(candidate_closed_mcpp[3]):
                    continue

                removed_mcpp_list.append(candidate_non_closed_mcpp)
                break

        print "# Non-closed MCPP: %d" % len(removed_mcpp_list)

        rules_1 = list()
        for mcpp in all_mcpp_1:
            if mcpp not in removed_mcpp_list:
                rules_1.append(mcpp)

        for i, rule in enumerate(rules_1):
            left = rule[0]
            right = rule[1]
            unique_candidate_rules[(left, right)] = rule
            all_rule_list.append(rule)
            rules.append(rule)

            # Construct graph for sequence exapansion
            if left not in left_right_dict:
                left_right_dict[left] = list()
            left_right_dict[left].append(right)

            if right not in right_left_dict:
                right_left_dict[right] = dict()
            right_left_dict[right][left] = 1

            if left not in left1_candidates:
                left1_candidates[left] = 0
            if right not in right1_candidate:
                right1_candidate[right] = 0

            left1_candidates[left] += 1
            right1_candidate[right] += 1

            rule = (rule[0], rule[1], rule[4], rule[5])

            debug_print("\t".join(str(element) for element in rule))
            f_pattern.write(str(len(left)) + "\t" + "\t".join(str(element) for element in rule) + "\n")

        print "# LEFT1: %d" % (len(left1_candidates))
        print "# RIGHT1: %d" % (len(right1_candidate))

        all_unique_rule_list = list()

        k = 1
        while True:
            k += 1

            print "START %d-MCPP" % k

            able_candidate = dict()
            pruned_candidate = dict()
            # new_rules = list()
            sorted_left1 = sorted(left1_candidates.keys(), key=lambda x: len(left_right_dict[x]), reverse=True)
            all_rules = list()
            next_rules = list() # Save rules in concatLeftCandidate
            valid_rules = list() # Same with next_rules (unnecessary variable)

            # Left Expansion
            for i, left1 in enumerate(sorted_left1):
                concatLeftCandidate(k, sorted_left1, left1, itemTimestampIndexDict, min_sup, min_conf, granularityTimestampList, next_rules, valid_rules, daily_time_idx_dict, max_gap=max_gap)
                if i % 100 == 0:
                    print "LEFT1: %d / %d" % (i, len(sorted_left1))

            for key in unique_candidate_rules:
                rule = unique_candidate_rules[key]
                all_unique_rule_list.append(rule)
                rule = (rule[0], rule[1], rule[4], rule[5])
                f_unique_pattern.write("\t".join(str(element) for element in rule) + "\n")

            # print "# Stop Growth Rules (%d sequences): %d" % (k - 1, len(unique_candidate_rules))

            left_right_dict = dict()
            right_left_dict = dict()
            left1_candidates = dict()
            right1_candidate = dict()
            unique_candidate_rules = dict()

            for rule in next_rules:
                left = rule[0]
                right = rule[1]
                unique_candidate_rules[(left, right)] = rule
                all_rule_list.append(rule)

                # Construct graph for sequence exapansion
                if left not in left_right_dict:
                    left_right_dict[left] = list()
                left_right_dict[left].append(right)

                if right not in right_left_dict:
                    right_left_dict[right] = dict()
                right_left_dict[right][left] = 1

                if left not in left1_candidates:
                    left1_candidates[left] = 0
                if right not in right1_candidate:
                    right1_candidate[right] = 0

                left1_candidates[left] += 1
                right1_candidate[right] += 1

                rule = (rule[0], rule[1], rule[4], rule[5])

                debug_print("\t".join(str(element) for element in rule))
                f_pattern.write(str(len(left)) + "\t" + "\t".join(str(element) for element in rule) + "\n")

            print "# Rules (%d sequences): " % k + str(len(next_rules))
            print "# Valid Rules (%d sequences): " % k + str(len(valid_rules))
            print "Combination Test Cnt: " + str(test_combination_cnt)

            if k == 24 or len(next_rules) == 0:  # len(timestampList):
                print "Complete"
                break

        for key in unique_candidate_rules:
            rule = unique_candidate_rules[key]
            all_unique_rule_list.append(rule)
            rule = (rule[0], rule[1], rule[4], rule[5])
            f_unique_pattern.write("\t".join(str(element) for element in rule) + "\n")

        f_pattern.close()
        f_unique_pattern.close()

        pickle.dump(all_unique_rule_list, f_all_unique_rule)
        f_all_unique_rule.close()

        pickle.dump(all_rule_list, f_all_rule)
        f_all_rule.close()

    ''' Complete gettting MCPP. After this line, calculating entropy in fully pattern and unfully pattern '''

    # Get features from MCPP
    period_feature_set = set()
    period_feature_count_dict = dict()
    for rule in all_rule_list:
        left = rule[0]
        right = rule[1]
        if len(left) != 1: # Count feature weight for period pattern in 1-MCPP
            continue
        for freq_pattern in [left, right]:
            for freq_pattern_item in freq_pattern:
                for item in freq_pattern_item:
                    feature = item.split(':')[0]
                    if feature == 'time_daily':
                        continue
                    period_feature_set.add(feature)
                    if feature not in period_feature_count_dict:
                        period_feature_count_dict[feature] = 0
                    period_feature_count_dict[feature] += 1

    print "Freqeuncy of Period Feature in Periodic Pattern (MCPP-1) - %s" % str(period_feature_count_dict)
    feature_set = set(data.keys())
    period_feature_set = feature_set & period_feature_set
    # feature_set = feature_set - period_feature_set
    if len(period_feature_set) == 0:
        continue

    class_threshold = int(len(dateStampSet) * min_sup)
    feature_set = set(full_data.keys()) - set(['celllocation_cid[cat]', 'celllocation_laccid[cat]'])
    class_feature_set = feature_set - period_feature_set

    for class_idx, class_label in enumerate(class_feature_set):
        print "Calculating entropy for class feature: %s (%d/%d)" % (class_label, class_idx, len(class_feature_set))
        pattern_len_fully_unfully_pattern_dict = dict()
        fully_unfully_entropy_list = AssociationExperiment.AnalysisEntropy(full_data, all_unique_rule_list, file_path, class_threshold, granularity_min, class_label)
        for fully_unfully_entropy in fully_unfully_entropy_list:
            fully_pattern, unfully_pattern_list, fully_class_list, unfully_class_list, fully_class_entropy, unfully_class_entropy = fully_unfully_entropy
            pattern_len = len(fully_pattern[0])
            if pattern_len not in pattern_len_fully_unfully_pattern_dict:
                pattern_len_fully_unfully_pattern_dict[pattern_len] = dict()
                pattern_len_fully_unfully_pattern_dict[pattern_len]["fully"] = list()
                pattern_len_fully_unfully_pattern_dict[pattern_len]["unfully"] = list()

            # print "%s\t%s\t%s\t%s\t%d\t%d\t%f\t%f" % (file_info, class_label, fully_pattern, unfully_pattern_list, len(fully_class_list), len(unfully_class_list), fully_class_entropy, unfully_class_entropy)
            max_conditional_len = 0
            for fully_mcpp_1 in fully_pattern[0]:
                if max_conditional_len < len(fully_mcpp_1):
                    max_conditional_len = len(fully_mcpp_1)
            if max_conditional_len > 1:
                pattern_len_fully_unfully_pattern_dict[pattern_len]["fully"].append(fully_class_entropy)
                pattern_len_fully_unfully_pattern_dict[pattern_len]["unfully"].append(unfully_class_entropy)
            f_entropy_result.write("%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d\t%f\t%f\n" % (file_info, class_label, pattern_len, max_conditional_len, str(fully_pattern[0]) + "-" + str(fully_pattern[1]), [str(unfully_pattern[0]) + "-" + str(unfully_pattern[1]) for unfully_pattern in unfully_pattern_list], len(fully_class_list), len(unfully_class_list), fully_class_entropy, unfully_class_entropy))
            pattern_pair_entropy_result_list.append((file_info, class_label, pattern_len, max_conditional_len, str(fully_pattern[0]) + "-" + str(fully_pattern[1]), [str(unfully_pattern[0]) + "-" + str(unfully_pattern[1]) for unfully_pattern in unfully_pattern_list], fully_class_list, unfully_class_list, fully_class_entropy, unfully_class_entropy))
        pattern_len_list = sorted(pattern_len_fully_unfully_pattern_dict.keys())
        for pattern_len in pattern_len_fully_unfully_pattern_dict:
            fully_entropy_list = pattern_len_fully_unfully_pattern_dict[pattern_len]["fully"]
            unfully_entropy_list = pattern_len_fully_unfully_pattern_dict[pattern_len]["unfully"]
            fully_entropy_mean = np.mean(fully_entropy_list) if len(fully_entropy_list) > 0 else -1
            unfully_entropy_mean = np.mean(unfully_entropy_list) if len(unfully_entropy_list) > 0 else -1

            f_entropy_avg_result.write("%s\t%s\t%d\t%d\t%d\t%f\t%f\n" % (file_info, class_label, pattern_len, len(fully_entropy_list), len(unfully_entropy_list), fully_entropy_mean, unfully_entropy_mean))
    f_entropy_result.flush()
    f_entropy_avg_result.flush()
    # PlotData.PlotData(dir_path, output_dir, file_info, granularity_min=granularity_min)
    pickle.dump(pattern_pair_entropy_result_list, f_entropy_result_pickle)
    f_entropy_result.close()
    f_entropy_result_pickle.close()
