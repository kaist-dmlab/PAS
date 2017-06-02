import datetime as dt
import time
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
import sys


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


def isValidateCombanation(min_sup, left1, left2, pruned_candidate, able_candidate, timestampList, itemTimestampIndexDict, daily_time_idx_dict, max_partial_len=1, left1_for_right1=None, left2_for_right2=None, left_tids_for_right=None, is_right=False, pattern_to_upper_bound_support=None):
    global test_combination_cnt
    global pattern_all_tids

    start_method_time = time.time()

    debug_print("Join Check\t" + str(left1) + "\t" + str(left2))
    if not is_right:
        sequence_key = ('left', left1, left2, None)
    else:
        if left_tids_for_right is not None and len(left_tids_for_right) > 0:
            left_tids = left_tids_for_right[0]
            idx_delta_list = list()
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

    # Pruning rule 3
    if PRUNING_RULE2_ON:
        if not is_right: # It means that this is checking left combination
            if pruned_candidate.get(sequence_key, -1) == 1:
                return is_sequence, None, None, None
            if able_candidate.get(sequence_key, -1) != -1:
                is_sequence = True
                new_pattern = able_candidate[sequence_key][0]
                output_tids = able_candidate[sequence_key][1]
                return is_sequence, new_pattern, output_tids, idx_delta_list

    '''
    if PRUNING_RULE3_ON:
        if is_right:
            if pruned_candidate.get(sequence_key, -1) == 1:
                return is_sequence, None, None, None
            if able_candidate.get(sequence_key, -1) != -1:
                is_sequence = True
                new_pattern = able_candidate[sequence_key][0]
                output_tids = able_candidate[sequence_key][1]
                return is_sequence, new_pattern, output_tids, idx_delta_list
    '''

    debug_print("LEFT1: " + str(left1))
    debug_print("LEFT2: " + str(left2))

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

    # Pruning rule 3
    if not PRUNING_RULE3_ON or new_pattern not in pattern_all_tids:   #(not PRUNING_RULE1_ON and not PRUNING_RULE2_ON)
        all_tids = list()
        valid_pattern_idx_list = []
        idx_delta_list = []
        prev_idx = None
        pattern_idx_dict = dict()
        for i, pattern in enumerate(new_pattern):
            if pattern not in pattern_idx_dict:
                pattern_idx_dict[pattern] = len(pattern_idx_dict)
            pattern_idx = pattern_idx_dict[pattern]
            tids = getTidsHavePatternList(patterns=(pattern,), itemTimestampIndexDict=itemTimestampIndexDict)
            tids = tids[0]
            tids_with_pattern = [(t, pattern_idx) for t in tids]

            all_tids += tids_with_pattern
            valid_pattern_idx_list.append(pattern_idx)

            if not is_right:
                if prev_idx is not None:
                    # print new_pattern[prev_idx]
                    # print left1_for_right1
                    # print left2_for_right2
                    # print new_pattern[i]
                    if left1_left2_for_right1_right2 is not None:
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

                    if delta > max_partial_len:
                        pruned_candidate[sequence_key] = 1
                        return is_sequence, None, None, None

                    idx_delta_list.append(delta)
                prev_idx = i

        all_tids = sorted(list(set(all_tids)))
        if is_right: # Because, right has differenct all value depending on left
            pattern_all_tids[new_pattern] = all_tids
        debug_print("ALL TIDS:" + str(all_tids))
        debug_print("PATTERN IDX LIST: " + str(valid_pattern_idx_list))
        debug_print("PATTERN IDX DELTA LIST: " + str(idx_delta_list))
    else:
        all_tids = pattern_all_tids[new_pattern]


    if is_right and (left_tids_for_right is None or len(left_tids_for_right) == 0):
        return is_sequence, None, None, None

    partial_test = False
    if not is_right:
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
                        if tid != prev_tid_pattern_idx[0] and i < len(all_tids) - 1 and tid != all_tids[i + 1][0]:
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
        debug_print("OUTPUT TIDS:" + str(output_tids))

        next_pattern_support = getSupportFromTids(output_tids, timestampList)
        prev_pattern_to_support[(new_pattern, tuple(idx_delta_list))] = next_pattern_support

        if next_pattern_support >= min_sup:
            is_sequence = True

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
                        if tid != prev_tid_pattern_idx[0] and i < len(all_tids) - 1 and tid != all_tids[i + 1][0]:
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

        next_pattern_support = getSupportFromTids(output_tids, timestampList)
        prev_pattern_to_support[(new_pattern, tuple(idx_delta_list))] = next_pattern_support
        if next_pattern_support >= min_sup:
            is_sequence = True

    if not is_sequence:
        pruned_candidate[sequence_key] = 1
    else:
        able_candidate[sequence_key] = (new_pattern, output_tids)

    end_method_time = time.time()

    if pattern_to_upper_bound_support is not None:
        new_pattern_key = (new_pattern, tuple(idx_delta_list))
        if new_pattern_key in pattern_to_upper_bound_support:
            upper_bound_info = pattern_to_upper_bound_support[new_pattern_key]
            updated_upper_bound_support = prev_pattern_to_support[new_pattern_key]
            F_UPPER.write(str(next_pattern_support) + "\t" + str(updated_upper_bound_support) + "\t".join(str(i) for i in list(upper_bound_info)) + "\n") # + str(upper_time) + "\t" + str(end_method_time - start_method_time) + "\n")
    
    return is_sequence, new_pattern, output_tids, idx_delta_list





def getIsSupersetPatternWithAddedPattern(subset_pattern, superset_pattern, subset_gap_list, superset_gap_list):
    # subset_pattern: {A,B}{C,D}, superset_pattern: {A,B,E}{C,D,F}
    # subset_pattern: {A,B}{C,D}, superset_pattern: {A,B}{C,D}{E,F}
    # subset_pattern: {A,B}{C,D}, superset_pattern: {A,B}{E,F}{C,D}

    # print 'Check following two patterns'
    # print subset_pattern
    # print superset_pattern
    subset_pattern_idx = 0
    unsame_superset_pattern_idx_list = list()
    superset_pattern_gap = 0
    for superset_pattern_idx in range(len(superset_pattern)):
        if subset_pattern_idx < len(subset_pattern):
            subset_item = set(subset_pattern[subset_pattern_idx])
            superset_item = set(superset_pattern[superset_pattern_idx])
            if (superset_pattern_idx - 1 >= 0):
                superset_pattern_gap = superset_pattern_gap + superset_gap_list[superset_pattern_idx - 1]
            subset_pattern_gap = 0
            if (subset_pattern_idx - 1 >= 0):
                subset_pattern_gap = subset_gap_list[subset_pattern_idx - 1]

            if subset_item != superset_item or superset_pattern_gap != subset_pattern_gap:
                unsame_superset_pattern_idx_list.append(superset_pattern_idx)
            if (subset_item == superset_item or subset_item.issubset(superset_item)) and superset_pattern_gap == subset_pattern_gap: # Check subset
                subset_pattern_idx += 1       
                superset_pattern_gap = 0             
        else:
            unsame_superset_pattern_idx_list.append(superset_pattern_idx)

    added_pattern = list() # Pattern C in P(A,C)
    for unsame_superset_pattern_idx in unsame_superset_pattern_idx_list:
        added_pattern.append(superset_pattern[unsame_superset_pattern_idx])

    if subset_pattern_idx == len(subset_pattern) and len(added_pattern) > 0: # If added_pattern is empty, this is same pattern
        is_superset = True
    else:
        is_superset = False
    # print is_superset

    return is_superset, tuple(added_pattern)

def getSupportFromPrevRules(pattern, left_pattern = None):
    for prev_rule in prev_rules:
        if pattern == prev_rule[0]:
            return prev_rule[7]
        elif left_pattern is not None and left_pattern == prev_rule[0] and pattern == prev_rule[1]:
            return prev_rule[8]

def getUpperBoundSupport(pattern1, pattern2, is_antecedent, left_pattern1 = None, left_pattern2 = None):
    # If pattern1 and pattern2 are consequent patterns, ancedent pattern (left_pattern1, left_pattern2) are needed.
    # Rule Info: (new_left, new_right, left_tids, new_both_tids, support, confidence, right_tids, left_support, right_support)  # , new_left_symbol, new_right_symbol)
    if is_antecedent:
        pattern_type = 0
    else:
        pattern_type = 1
    
    # Naive Method

    # print "pattern 1: " + str(pattern1)
    # print "pattern 2: " + str(pattern2)

    pattern1_supp = getSupportFromPrevRules(pattern1, left_pattern1)
    pattern2_supp = getSupportFromPrevRules(pattern2, left_pattern2)
    upper_bound = min(pattern1_supp, pattern2_supp)

    # print pattern1_supp
    # print pattern2_supp
    # print upper_bound

    if len(pattern1) == 1:
        return upper_bound, None

    # Get common pattern C
    # pattern_len = len(pattern2)
    # common_pattern = pattern2[:pattern_len - 1]

    # print common_pattern

    # Get Gap List of Pattern1, Pattern2
    if is_antecedent:
        pattern1_gap_list = getGapListInPattern(pattern1)
        pattern2_gap_list = getGapListInPattern(pattern2)
    else:
        pattern1_gap_list = getGapListInPattern(left_pattern1)
        pattern2_gap_list = getGapListInPattern(left_pattern2)

    # Find supp(A,C) and supp(C,B) where C is `added_pattern`.
    superset_common_pattern_to_superset_pair_support = dict()
    for prev_pattern_info in prev_pattern_to_support:
        candidate_pattern = prev_pattern_info[0] # It can be supp(A,C) or supp(C,B).
        candidate_pattern_gap_list = prev_pattern_info[1]
        candidate_pattern_supp = prev_pattern_to_support[prev_pattern_info]
        
        is_superset_of_pattern1, added_pattern_of_pattern1 = getIsSupersetPatternWithAddedPattern(pattern1, candidate_pattern, pattern1_gap_list, candidate_pattern_gap_list)
        if is_superset_of_pattern1:
            # supp(A,C)
            if not added_pattern_of_pattern1 in superset_common_pattern_to_superset_pair_support:
                superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern1] = dict()
                superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern1][0] = list()
                superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern1][1] = list()

            superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern1][0].append((candidate_pattern, candidate_pattern_supp))

        is_superset_of_pattern2, added_pattern_of_pattern2 = getIsSupersetPatternWithAddedPattern(pattern2, candidate_pattern, pattern2_gap_list, candidate_pattern_gap_list)
        if is_superset_of_pattern2:
            # supp(C,B)
            if not added_pattern_of_pattern2 in superset_common_pattern_to_superset_pair_support:
                superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern2] = dict()
                superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern2][0] = list()
                superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern2][1] = list()

            superset_common_pattern_to_superset_pair_support[added_pattern_of_pattern2][1].append((candidate_pattern, candidate_pattern_supp))
    
    # Find C maximizing |supp(A,C) - supp(C,B)| 
    max_superset_pair_support = None
    max_superset_common_pattern = None
    min_upper_bound = upper_bound

    for superset_common_pattern in superset_common_pattern_to_superset_pair_support:
        superset_pair_support = superset_common_pattern_to_superset_pair_support[superset_common_pattern]
        if len(superset_pair_support.keys()) == 2:
            # print "--------------------------------"
            # print pattern1
            # print pattern2
            # print superset_common_pattern
            # print superset_pair_support[0][0]
            # print superset_pair_support[1][0]

            # Calculate upper bound
            # if supp(B) > supp(A)
            # min(supp(A,C), supp(C,B)) + min(supp(B) - max(0, supp(C,B) - supp(A,C)), upper_bound) - supp(A)
            upper_bound = min(pattern1_supp, pattern2_supp) # Naive
            for superset_pair_support_pattern1 in superset_pair_support[0]:
                for superset_pair_support_pattern2 in superset_pair_support[1]:
                    pattern1_common_pattern_supp = superset_pair_support_pattern1[1]
                    pattern2_common_pattern_supp = superset_pair_support_pattern2[1]
                    if pattern1_common_pattern_supp < pattern2_common_pattern_supp:
                        upper_bound = upper_bound + min(((pattern2_supp - upper_bound) - (pattern2_common_pattern_supp - pattern1_common_pattern_supp)),0)
                    else:
                        upper_bound = upper_bound + min(((pattern1_supp - upper_bound) - (pattern1_common_pattern_supp - pattern2_common_pattern_supp)),0)
                    '''
                    if pattern1_supp < pattern2_supp:
                        upper_bound = min(pattern1_common_pattern_supp, pattern2_common_pattern_supp) + min(pattern2_supp - max(0, pattern2_common_pattern_supp - pattern1_common_pattern_supp), upper_bound) - pattern1_common_pattern_supp
                    else:
                        upper_bound = min(pattern1_common_pattern_supp, pattern2_common_pattern_supp) + min(pattern1_supp - max(0, pattern1_common_pattern_supp - pattern2_common_pattern_supp), upper_bound) - pattern2_common_pattern_supp
                    '''
                    if min_upper_bound > upper_bound:
                        min_upper_bound = upper_bound
                        max_superset_pair_support = (superset_pair_support_pattern1, superset_pair_support_pattern2)
                        max_superset_common_pattern = superset_common_pattern

    upper_bound = min_upper_bound
    '''
    if max_superset_pair_support is not None:
        print pattern1
        print pattern2
        print pattern1_supp
        print pattern2_supp
        print max_superset_pair_support
        print "Upper Bound: " + str(upper_bound)
    '''

    return upper_bound, (pattern1, pattern2, pattern1_supp, pattern2_supp, upper_bound, max_superset_pair_support, max_superset_common_pattern)

def concatLeftCandidate(k, left_candidates, left1, itemTimestampIndexDict, min_sup, min_conf, timestampList, nextRules, validRules, daily_time_idx_dict, max_partial_len, pattern_to_upper_bound_support):
    global next_pattern_to_support
    global prev_pattern_to_support
    for left2_index in range(len(left_candidates)):
        debug_print("--------------------------------")
        debug_print("Given Left1: " + str(left1))

        left2 = left_candidates[left2_index]

        tttt1 = time.time()
        left_is_sequence, new_left, left_tids, left_gap_list = isValidateCombanation(min_sup, left1, left2, pruned_candidate, able_candidate, timestampList, itemTimestampIndexDict, daily_time_idx_dict, max_partial_len=max_partial_len, pattern_to_upper_bound_support=pattern_to_upper_bound_support)
        tttt2 = time.time()

        # Pruning rule 1 (Remove right combination when left combination is invalid.)
        if left_is_sequence or not PRUNING_RULE1_ON:
            tt1 = time.time()
            
            right_pattern_to_upper_bound_support = dict()
            if k > 2:
                right_pattern_to_upper_bound_support = UpdateUpperBoundSupportPairSets(left_right_dict[left1], left_right_dict[left2], right_pattern_to_upper_bound_support, True, left1, left2)
                '''
                print "Consequent Upper Bound"
                # Calculate upper bound support for right pairs
                for right1 in left_right_dict[left1]:
                    for right2 in set(left_right_dict[left2]):
                        new_pattern_info, upper_bound_info = calculateUpperBoundSupport(right1, right2, True, left1, left2)
                        if new_pattern_info is not None and upper_bound_info is not None:
                            right_pattern_to_upper_bound_support[new_pattern_info] = upper_bound_info
                # Update upper bound support for each pattern
                print "Upper bound update information", len(right_pattern_to_upper_bound_support.keys())
                # Update superset pattern support based on subset pattern
                for subset_pattern_info in right_pattern_to_upper_bound_support:
                    subset_pattern = subset_pattern_info[0]
                    subset_pattern_gap_list = subset_pattern_info[1]
                    subset_pattern_support = right_pattern_to_upper_bound_support[subset_pattern_info][4]
                    for superset_pattern_info in right_pattern_to_upper_bound_support:
                        superset_pattern = superset_pattern_info[0]
                        superset_pattern_gap_list = superset_pattern_info[1]
                        superset_pattern_support = right_pattern_to_upper_bound_support[superset_pattern_info][4]
                        is_superset_of_pattern1, added_pattern_of_pattern1 = getIsSupersetPatternWithAddedPattern(subset_pattern, superset_pattern, subset_pattern_gap_list, superset_pattern_gap_list)
                        if is_superset_of_pattern1 and superset_pattern_support > subset_pattern_support:
                            print subset_pattern_info
                            print superset_pattern_info
                            print subset_pattern_support
                            print superset_pattern_support
                            right_pattern_to_upper_bound_support[superset_pattern_info] = right_pattern_to_upper_bound_support[superset_pattern_info] + (subset_pattern_support,)
            '''
            for right1 in left_right_dict[left1]:
                right2_candidates = set(left_right_dict[left2])

                for right2 in right2_candidates:
                    right_is_sequence, new_right, right_tids, left_gap_list = isValidateCombanation(min_sup, right1, right2, pruned_candidate, able_candidate, timestampList, itemTimestampIndexDict, daily_time_idx_dict, left1_for_right1=left1, left2_for_right2=left2, left_tids_for_right=left_tids, is_right=True, max_partial_len=max_partial_len, pattern_to_upper_bound_support=right_pattern_to_upper_bound_support)
                    debug_print("Right Symbol Check: " + str(right_is_sequence))

                    if PRUNING_RULE2_ON and not right_is_sequence and right_tids is not None: # Already known
                        # Pruning rule 2 (Remove left combination that has only one right that is uncombinatable right set.)
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
                            '''
                            if (left1, right1) in unique_candidate_rules:
                                del unique_candidate_rules[(left1, right1)]
                            if (left2, right2) in unique_candidate_rules:
                                del unique_candidate_rules[(left2, right2)]
                            '''
                            validRules.append(rule_info)
                        debug_print("-------------")
        tt2 = time.time()


def getGapListInPattern(pattern):
    # This method works only on ancedent pattern (In other words, for patterns containing 'time_daily')
    # Calculate Gap
    idx_delta_list = []
    prev_idx = None
    pattern_idx_dict = dict()
    for i, item in enumerate(pattern):
        if prev_idx is not None:
            current_pattern_time = pattern[i][findIndexWithKey('time_daily', pattern[i])].split('time_daily:')[1]
            prev_pattern_time = pattern[prev_idx][findIndexWithKey('time_daily', pattern[prev_idx])].split('time_daily:')[1]

            current_pattern_time_idx = daily_time_idx_dict[current_pattern_time]
            prev_pattern_time_idx = daily_time_idx_dict[prev_pattern_time]

            delta = current_pattern_time_idx - prev_pattern_time_idx
            if delta < 0:
                delta = len(daily_time_idx_dict) + delta
            idx_delta_list.append(delta)
        prev_idx = i
    return idx_delta_list


def getJoinedPattern(pattern1, pattern2):
    # Check if patterns are connected
    pattern_len = len(pattern1)
    if pattern_len == 1:
        pattern1_time_daily_index = findIndexWithKey('time_daily', pattern1[0])
        pattern2_time_daily_index = findIndexWithKey('time_daily', pattern2[0])

        if pattern1_time_daily_index is not None and pattern2_time_daily_index is not None:
            if pattern1[0][pattern1_time_daily_index] == pattern2[0][pattern2_time_daily_index]:
                return None
        new_pattern = pattern1 + pattern2
        return new_pattern
    else:
        if pattern1[-(pattern_len - 1):] == pattern2[:pattern_len - 1]:
            new_pattern = pattern1 + pattern2[-1:]
            return new_pattern
        else:
            return None
 


def calculateUpperBoundSupport(pattern1, pattern2, is_right = False, left1 = None, left2 = None):
   
    tttt1 = time.time()

    debug_print("Pattern1: " + str(pattern1))
    debug_print("Pattern2: " + str(pattern2))
 
    # Check if patterns are connected
    pattern_len = len(pattern1)
    if is_right:
        left_new_pattern = getJoinedPattern(left1, left2)
        new_pattern = getJoinedPattern(pattern1, pattern2)
    else:
        left_new_pattern = getJoinedPattern(pattern1, pattern2)
        new_pattern = left_new_pattern
    
    if left_new_pattern is None or new_pattern is None: # Not joinable
        return None, None, None

    # Gap for new pattern
    if not is_right:
        new_pattern_gap_list = getGapListInPattern(new_pattern)
    else:
        new_pattern_gap_list = getGapListInPattern(left_new_pattern)

    new_pattern_info = (new_pattern, tuple(new_pattern_gap_list))

    # Calculate upper bound
    upper_bound = None
    upper_time = None
    if pattern_len > 1:
        upper_start_ts = time.time()
        upper_bound_support, upper_bound_info = getUpperBoundSupport(pattern1, pattern2, not is_right, left1, left2)
        upper_end_ts = time.time()
        upper_time = upper_end_ts - upper_start_ts
        
    tttt2 = time.time()
    return new_pattern_info, upper_bound_info, upper_bound_support

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


def UpdateUpperBoundSupportPairSets(pattern_list1, pattern_list2, pattern_to_upper_bound_support, is_right = False, left1 = None, left2 = None):
    global prev_pattern_to_support
    
    next_pattern_to_support = dict()
    
    endUpdate = True
    updateCnt = 0
    while endUpdate:
        # Update upper bound support until upper bound is not changed.
        print "Upper Bound Update: %d" % updateCnt
        for pattern1 in pattern_list1:
            for pattern2 in pattern_list2:
                new_pattern_info, upper_bound_info, upper_bound_support = calculateUpperBoundSupport(pattern1, pattern2, is_right, left1, left2)
                if new_pattern_info is not None and upper_bound_info is not None:
                    pattern_to_upper_bound_support[new_pattern_info] = upper_bound_info
                    next_pattern_to_support[new_pattern_info] = upper_bound_support #Key contain gap list (new_pattern, gap_list)
                    if new_pattern_info not in prev_pattern_to_support or prev_pattern_to_support[new_pattern_info] != upper_bound_support:
                        print "-----------------------"
                        print new_pattern_info
                        print upper_bound_info
                        print upper_bound_support
                        if new_pattern_info in prev_pattern_to_support:
                            print "Old", prev_pattern_to_support[new_pattern_info]
                        print "-----------------------"
        # Update upper bound support for each pattern
        print "Upper bound update information", len(next_pattern_to_support.keys())

        # Update prev_pattern_to_support based on next_pattern_to_support (upper bound pattern)
        for new_pattern_info in next_pattern_to_support:
            prev_pattern_to_support[new_pattern_info] = next_pattern_to_support[new_pattern_info]

        # Update superset pattern support based on subset pattern
        for subset_pattern_info in prev_pattern_to_support:
            subset_pattern = subset_pattern_info[0]
            subset_pattern_gap_list = subset_pattern_info[1]
            subset_pattern_support = prev_pattern_to_support[subset_pattern_info]
            for superset_pattern_info in prev_pattern_to_support:
                superset_pattern = superset_pattern_info[0]
                superset_pattern_gap_list = superset_pattern_info[1]
                superset_pattern_support = prev_pattern_to_support[superset_pattern_info]
                is_superset_of_pattern1, added_pattern_of_pattern1 = getIsSupersetPatternWithAddedPattern(subset_pattern, superset_pattern, subset_pattern_gap_list, superset_pattern_gap_list)
                if is_superset_of_pattern1 and superset_pattern_support > subset_pattern_support:
                    print subset_pattern_info
                    print superset_pattern_info
                    print subset_pattern_support
                    print superset_pattern_support
                    prev_pattern_to_support[superset_pattern_info] = subset_pattern_support

        if len(next_pattern_to_support.keys()) == 0:
            endUpdate = False
        next_pattern_to_support = dict()
    return pattern_to_upper_bound_support

# min_sup = 1.0 / 10.0
# min_conf = 2.0 / 10.0

# dir_path = '/NAS/periodicity/raw2'
dir_path = 'dataset/raw'
getall = [[files, os.path.getsize(dir_path + "/" + files)] for files in os.listdir(dir_path)]
file_info_list = list()
for file_name, file_size in sorted(getall, key=operator.itemgetter(1)):
    if file_name[-22:] == 'timeseries_dict.pickle':
        file_info_list.append(file_name.split('.')[0])
file_info_list = file_info_list
print file_info_list
print len(file_info_list)
# sys.exit()
# file_info_list = ['4786ada47b323ec6740a7b1a9793db0fde6790ae_timeseries_dict']

if len(sys.argv) == 1:
    start_idx = 0
    end_idx = len(file_info_list)
elif len(sys.argv) == 3:
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
else:
    print "Argument error, <python filename start_idx end_idx>"
    sys.exit()

print "START %d ~ END %d" % (start_idx, end_idx)

# output_dir = '/NAS/periodicity/preprocessed'
output_dir = dir_path
# rules_dir = '/NAS/periodicity/rules'
rules_dir = 'rules'

granularity_min = 60
MIN_DATE_LENGTH = 30

# log = open('user_log.txt', 'w')
F_UPPER = open('output/upper_bound_result.txt', 'w')
F_PERFORM = open('output/performance_result.txt', 'w')
file_info_list = file_info_list[start_idx:end_idx]
for user_idx, file_info in enumerate(file_info_list):
    print "%d/%d - %s" % (user_idx, len(file_info_list), file_info)

    file_path = dir_path + '/%s.pickle' % file_info
    data, file_path = ruduce_category_features.reduce_catergory_feature(file_path, output_dir)
    print data.keys()
    full_data, _ = removed_default_value.removed_default_value(file_path, output_dir, NIGHT_START_TIMESTAMP_STRING='01-00-00', NIGHT_END_TIMESTAMP_STRING='07-00-00', drop_default=False, drop_feature=False, is_full=True)
    print full_data.keys()
    data, file_path = removed_default_value.removed_default_value(file_path, output_dir, NIGHT_START_TIMESTAMP_STRING='01-00-00', NIGHT_END_TIMESTAMP_STRING='07-00-00')
    print data.keys()

    timestamps = data[data.keys()[0]].index
    if 'class' not in data:
        print "!!! NO CLASS !!!"
        # log.write('%s\tNO_CLASS\n' % file_info)
        continue

    csv_file = output_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_%d.csv' % (granularity_min)
    daily_time_list, tid_to_granualarity_tid = dictToCsv.exportToCSV(data, timestamps, csv_file, include_class=True, granularity_min=granularity_min)
    class_list = getClassFromFile(csv_file)
    inFile = Apriori.dataFromFile(csv_file, hasClass=True)
    print "Complete Read CSV for Apriori"

    ''' START GETTING MCPP '''
    # min_sup = 1.0 / 10.0
    # min_conf = 2.0 / 10.0
    for try_idx in range(1):
        for performance_type in ['conf']: # , 'sup', 'partial']: Only once
            if performance_type == 'conf':
                min_sup = 10 / 100.0
                min_conf = 10 / 100.0
                max_partial_len = 1
            elif performance_type == 'sup':
                min_sup = 10 / 100.0
                min_conf = 10 / 100.0
                max_partial_len = 1
            elif performance_type == 'partial':
                min_sup = 10 / 100.0
                min_conf = 10 / 100.0
                max_partial_len = 1
            is_stop = False
            while not is_stop:
                for PRUNING_RULE1_ON, PRUNING_RULE2_ON, PRUNING_RULE3_ON in [(True, True, False)]: #, (False, True, False), (True, False, False), (False, False, False)]: Only once
                    print ">>>>>> min_sup: %f, min_conf: %f, max_partial_len: %d <<<<<<" % (min_sup, min_conf, max_partial_len)
                    print "Pruning Rule (Foward Stop + Prune, Backward Prune): %s, %s" % (PRUNING_RULE1_ON, PRUNING_RULE2_ON)
                    # f_entropy_result_pickle = open('entropy_result/%s_pattern_pair_entropy_%.2f_%.2f.pickle' % (file_info, min_sup, min_conf), 'w')
                    # f_entropy_result = open('entropy_result/%s_pattern_pair_entropy_%.2f_%.2f.txt' % (file_info, min_sup, min_conf), 'w')
                    # f_entropy_avg_result = open('entropy_result/%s_class_avg_entropy_%.2f_%.2f.txt' % (file_info, min_sup, min_conf), 'w')
                    # pattern_pair_entropy_result_list = list()

                    test_combination_cnt = 0
                    pattern_all_tids = dict()

                    ALL_RULE_PATH = rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_MCPP_%d_%.2f_%.2f.pickle' % (granularity_min, min_sup, min_conf)
                    ALL_UNIQUE_PATH = rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_UNIQUE_MCPP_%d_%.2f_%.2f.pickle' % (granularity_min, min_sup, min_conf)
                    APRIORI_RESULT_PATH = rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_MCPP_1_%d_%.2f_%.2f.pickle' % (granularity_min, min_sup, min_conf)

                    ALL_STARTTIME = time.time()

                    existRules = False

                    if False and os.path.exists(ALL_RULE_PATH) and os.path.exists(ALL_UNIQUE_PATH) and os.path.exists(APRIORI_RESULT_PATH):
                        print "ALREADY EXIST: " + str(ALL_RULE_PATH)
                        print "ALREADY EXIST: " + str(ALL_UNIQUE_PATH)

                        all_unique_rule_list = pickle.load(open(ALL_UNIQUE_PATH))
                        all_rule_list = pickle.load(open(ALL_RULE_PATH))
                        items, rules_1, itemTimestampIndexDict, timestampList = pickle.load(open(APRIORI_RESULT_PATH))

                        dateStampSet = set()
                        t1 = time.time()
                        for timestamp in timestampList:
                                timestamp = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                                dateStampSet.add(timestamp)
                        t2 = time.time()
                        existRules = True

                    if not existRules:
                        ts1 = time.time()
                        MCPP1_STARTTIME = time.time()
                        items, rules_1, itemTimestampIndexDict, timestampList = Apriori.runApriori(inFile, min_sup, min_conf, tid_to_granualarity_tid, granularity_min)
                        MCPP1_ENDTIME = time.time()
                        print "MCPP 1 Time: %f" % (MCPP1_ENDTIME - MCPP1_STARTTIME)
                        F_PERFORM.write('%s\t%s\t%d\t%f\t%f\t%d\t%s\t%s\t%f\n' % (file_info, performance_type, 1, min_sup, min_conf, max_partial_len, PRUNING_RULE1_ON, PRUNING_RULE2_ON, MCPP1_ENDTIME - MCPP1_STARTTIME))
                        ts2 = time.time()

                        tatatata = time.time()

                        granularityTimestampList = set()
                        for timestamp in timestampList:
                            granularity_timestamp = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([timestamp]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
                            granularityTimestampList.add(granularity_timestamp)
                        granularityTimestampList = sorted(granularityTimestampList)

                        daily_time_idx_dict = dict()
                        for i, daily_time in enumerate(daily_time_list):
                            daily_time_idx_dict[daily_time] = i
                        # Apriori.printResults(items, rules)

                        print "GENERATE 1-MCPP: %f" % (ts2 - ts1)
                        print "# Rules (%d sequences): " % 1 + str(len(rules_1))

                        dateStampSet = set()
                        t1 = time.time()
                        for timestamp in timestampList:
                                timestamp = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                                dateStampSet.add(timestamp)
                        t2 = time.time()

                        print "# DATE: %d" % len(dateStampSet)

                        if len(dateStampSet) < MIN_DATE_LENGTH:
                            print "!!! NOT ENOUGH DATES !!! %d" % (len(dateStampSet))
                            # log.write('%s\tNOT_ENOGH_DATE(%d)\n' % (file_info, len(dateStampSet)))
                            continue

                        # f = open(APRIORI_RESULT_PATH, 'w')
                        # f_all_rule = open(ALL_RULE_PATH, 'w')
                        # f_all_unique_rule = open(ALL_UNIQUE_PATH, 'w')
                        # f_pattern = open(rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_MCPP_%d.txt' % granularity_min, 'w')
                        # f_unique_pattern = open(rules_dir + "/" + file_path.split('/')[-1].split('.')[0] + '_UNIQUE_MCPP_%d.txt' % granularity_min, 'w')

                        # rules_1_result = (items, rules_1, itemTimestampIndexDict, timestampList)
                        # pickle.dump(rules_1_result, f)
                        # f.close()

                        rules = list()
                        left_right_dict = dict()
                        right_left_dict = dict()
                        left1_candidates = dict()
                        right1_candidate = dict()
                        # unique_candidate_rules = dict()
                        all_rule_list = list()

                        print "Building LEFT1, RIGHT1 relationship"

                        aaa = time.time()

                        """ Get Closed Patterns """
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

                        print "Print ALL MCPP 1 Afterprocessing Time: %f" % (time.time() - aaa)

                        aaa = time.time()

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
                        print "Remove Closed Time: %f" % (time.time() - aaa)

                        bbb = time.time()
                        
                        rules_1 = list()
                        for mcpp in all_mcpp_1:
                            if mcpp not in removed_mcpp_list:
                                rules_1.append(mcpp)
                        
                        rules_1 = all_mcpp_1
                        next_rules = rules_1
                        for i, rule in enumerate(rules_1):
                            left = rule[0]
                            right = rule[1]
                            # unique_candidate_rules[(left, right)] = rule
                            all_rule_list.append(rule)
                            rules.append(rule)

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
                            # f_pattern.write(str(len(left)) + "\t" + "\t".join(str(element) for element in rule) + "\n")

                        print "# LEFT1: %d" % (len(left1_candidates))
                        print "# RIGHT1: %d" % (len(right1_candidate))

                        # all_unique_rule_list = list()
                        print "MCPP 1 after processing final: %f" % (time.time() - bbb)
                        print "MCPP 1 Other Time: %f" % (time.time() - tatatata)

                        # Save support information of each pattern(item)
                        next_pattern_to_support = dict()
                        for item in items:
                            pattern = (item[0],)
                            support = item[1]
                            next_pattern_to_support[(pattern, tuple(list()))] = support
                        
                        k = 1
                        while True:
                            k += 1

                            if time.time() - ALL_STARTTIME > 1000:
                                print "Too long execution time!"
                                break


                            print "START %d-MCPP" % k

                            MCPP_N_STARTTIME = time.time()

                            able_candidate = dict()
                            pruned_candidate = dict()
                            # new_rules = list()
                            start_ts = time.time()
                            sorted_left1 = sorted(left1_candidates.keys(), key=lambda x: len(left_right_dict[x]), reverse=True)
                            all_rules = list()
                            prev_rules = list(next_rules)
                            next_rules = list()
                            valid_rules = list()
    
                            prev_pattern_to_support = dict(next_pattern_to_support)
                            next_pattern_to_support = dict()

                            pattern_to_upper_bound_support = dict() # Saved upper bound pattern information globally
                            if k > 2:
                                pattern_to_upper_bound_support = UpdateUpperBoundSupportPairSets(sorted_left1, sorted_left1, pattern_to_upper_bound_support)
                                '''
                                endUpdate = True
                                updateCnt = 0
                                while endUpdate:
                                    # Update upper bound support until upper bound is not changed.
                                    print "Ancedent Upper Bound Update: %d" % updateCnt
                                    for left1 in sorted_left1:
                                        for left2 in sorted_left1:
                                            new_pattern_info, upper_bound_info, upper_bound_support = calculateUpperBoundSupport(left1, left2)
                                            if new_pattern_info is not None and upper_bound_info is not None:
                                                pattern_to_upper_bound_support[new_pattern_info] = upper_bound_info
                                                next_pattern_to_support[new_pattern_info] = upper_bound_support #Key contain gap list (new_pattern, gap_list)
                                                if new_pattern_info not in prev_pattern_to_support or prev_pattern_to_support[new_pattern_info] != upper_bound_support:
                                                    print "-----------------------"
                                                    print new_pattern_info
                                                    print upper_bound_info
                                                    print upper_bound_support
                                                    if new_pattern_info in prev_pattern_to_support:
                                                        print "Old", prev_pattern_to_support[new_pattern_info]
                                                    print "-----------------------"
                                    # Update upper bound support for each pattern
                                    print "Upper bound update information", len(next_pattern_to_support.keys())

                                    # Update prev_pattern_to_support based on next_pattern_to_support (upper bound pattern)
                                    for new_pattern_info in next_pattern_to_support:
                                        prev_pattern_to_support[new_pattern_info] = next_pattern_to_support[new_pattern_info]

                                    # Update superset pattern support based on subset pattern
                                    for subset_pattern_info in prev_pattern_to_support:
                                        subset_pattern = subset_pattern_info[0]
                                        subset_pattern_gap_list = subset_pattern_info[1]
                                        subset_pattern_support = prev_pattern_to_support[subset_pattern_info]
                                        for superset_pattern_info in prev_pattern_to_support:
                                            superset_pattern = superset_pattern_info[0]
                                            superset_pattern_gap_list = superset_pattern_info[1]
                                            superset_pattern_support = prev_pattern_to_support[superset_pattern_info]
                                            is_superset_of_pattern1, added_pattern_of_pattern1 = getIsSupersetPatternWithAddedPattern(subset_pattern, superset_pattern, subset_pattern_gap_list, superset_pattern_gap_list)
                                            if is_superset_of_pattern1 and superset_pattern_support > subset_pattern_support:
                                                print subset_pattern_info
                                                print superset_pattern_info
                                                print subset_pattern_support
                                                print superset_pattern_support
                                                prev_pattern_to_support[superset_pattern_info] = subset_pattern_support

                                    if len(next_pattern_to_support.keys()) == 0:
                                        endUpdate = False
                                    next_pattern_to_support = dict()
                                '''
                            ttt1 = time.time()
                            for i, left1 in enumerate(sorted_left1):
                                concatLeftCandidate(k, sorted_left1, left1, itemTimestampIndexDict, min_sup, min_conf, granularityTimestampList, next_rules, valid_rules, daily_time_idx_dict, max_partial_len=max_partial_len, pattern_to_upper_bound_support = pattern_to_upper_bound_support)
                                if i % 100 == 0:
                                    ttt2 = time.time()
                                    print "LEFT1: %d / %d" % (i, len(sorted_left1))
                                    print "TIMER per LEFT1: %f" % (ttt2 - ttt1)
                                    ttt1 = time.time()
                                    if time.time() - ALL_STARTTIME > 1000:
                                        print "Too long execution time!"
                                        break

                            end_ts = time.time()
                            MCPP_N_ENDTIME = time.time()
                            F_PERFORM.write('%s\t%s\t%d\t%f\t%f\t%d\t%s\t%s\t%f\n' % (file_info, performance_type, k, min_sup, min_conf, max_partial_len, PRUNING_RULE1_ON, PRUNING_RULE2_ON, MCPP_N_ENDTIME - MCPP_N_STARTTIME))
                            print "MCPP N Time: %f" % (MCPP_N_ENDTIME - MCPP_N_STARTTIME)
                            '''
                            for key in unique_candidate_rules:
                                rule = unique_candidate_rules[key]
                                all_unique_rule_list.append(rule)
                                rule = (rule[0], rule[1], rule[4], rule[5])
                                # f_unique_pattern.write("\t".join(str(element) for element in rule) + "\n")
                            '''
                            # print "# Stop Growth Rules (%d sequences): %d" % (k - 1, len(unique_candidate_rules))

                            tatatata = time.time()

                            left_right_dict = dict()
                            right_left_dict = dict()
                            left1_candidates = dict()
                            right1_candidate = dict()
                            # unique_candidate_rules = dict()

                            for rule in next_rules:
                                left = rule[0]
                                right = rule[1]
                                # unique_candidate_rules[(left, right)] = rule
                                all_rule_list.append(rule)

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
                                # f_pattern.write(str(len(left)) + "\t" + "\t".join(str(element) for element in rule) + "\n")

                            print "MCPP N Other Time: %f" % (time.time() - tatatata)

                            print "# Rules (%d sequences): " % k + str(len(next_rules))
                            print "# Valid Rules (%d sequences): " % k + str(len(valid_rules))
                            print end_ts - start_ts
                            print "Combination Test Cnt: " + str(test_combination_cnt)

                            if time.time() - ALL_STARTTIME > 1000:
                                print "Too long execution time!"
                                break

                            if k == 24 or len(next_rules) == 0:  # len(timestampList):
                                print "Complete"
                                break

                        '''
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
                        '''

                    ''' Complete gettting MCPP. After this line, calculating entropy in fully pattern and unfully pattern '''
                    '''
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
                    print feature_set
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
                    '''
                    ALL_ENDTIME = time.time()
                    F_PERFORM.write('%s\t%s\t%s\t%f\t%f\t%d\t%s\t%s\t%f\n' % (file_info, performance_type, 'ALL', min_sup, min_conf, max_partial_len, PRUNING_RULE1_ON, PRUNING_RULE2_ON, ALL_ENDTIME - ALL_STARTTIME))
                    print "All Time: %f" % (ALL_ENDTIME - ALL_STARTTIME)
                is_stop = True # Only once
                if performance_type == 'conf':
                    if min_conf >= (60 / 100.0):
                        is_stop = True
                    min_conf = min_conf + (10 / 100.0)
                    print min_conf
                elif performance_type == 'sup':
                    if min_sup >= (20 / 100.0):
                        is_stop = True
                    min_sup = min_sup + (25 / 1000.0)
                    print min_sup
                elif performance_type == 'partial':
                    if max_partial_len >= 5:
                        is_stop = True
                    max_partial_len = max_partial_len + 1
                    print min_sup
F_UPPER.close()
# log.close()
