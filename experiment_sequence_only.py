import os.path
import operator
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams.update({'font.size': 9})


def GetLift(support, left_support, right_support):
    return support / float(left_support * right_support)


def PlotPatternCountAsMinconf(rules_dir_path, output_dir):
    if not os.path.exists("%s/%s" % (output_dir, 'pattern_len_boxplot_data_dict.pickle')):
        getall = [[files, os.path.getsize(rules_dir_path + "/" + files)] for files in os.listdir(rules_dir_path)]
        file_info_list = list()
        for file_name, file_size in sorted(getall, key=operator.itemgetter(1)):
            if file_name[-30:] == 'value_MCPP_60_0.20_0.20.pickle': # We check confidence as each length of MCPP. Thus, we have to use all patterns not unique patterns.
                file_info_list.append(file_name.split('.pickle')[0])
        # file_info_list = ["051eed5b393ce23820369ccaf6bc053403029b9d_timeseries_dict_reindexed_2013-05-02_00-00-00_2015-08-02_23-55-00_reduced_feature_removed_default_value_UNIQUE_MCPP_60"]
        print file_info_list
        print len(file_info_list)


        min_sup = 0.20
        min_conf = 0.20

        valid_user_cnt = 0
        file_info_list = file_info_list
        for i, file_info in enumerate(file_info_list):
            # print file_info
            try:
                print "Processing %d/%d" % (i, len(file_info_list))
                rules = pickle.load(open("%s/%s.pickle" % (rules_dir_path, file_info)))
                valid_user_cnt += 1
                pattern_len_rules = dict()
                for rule in rules:
                    left_pattern, right_pattern, _, _, support, confidence, _, left_support, right_support = rule
                    if support < min_sup or confidence < min_conf:
                        continue
                    lift = GetLift(support, left_support, right_support)
                    pattern_len = len(left_pattern)
                    if pattern_len not in pattern_len_rules:
                        pattern_len_rules[pattern_len] = list()
                    pattern_len_rules[pattern_len].append((left_pattern, right_pattern, support, confidence, lift))
            except EOFError, e:
                print e
                print file_info

        print "# Users: %d" % valid_user_cnt

        def GetFrequentPattern(left_pattern, right_pattern):
            frequent_pattern = list()
            for mcpp_idx in range(len(left_pattern)):
                mcpp_frequent_set = left_pattern[mcpp_idx] + right_pattern[mcpp_idx]
                frequent_pattern.append(tuple(sorted(list(mcpp_frequent_set))))
            return tuple(frequent_pattern)

        pattern_len_boxplot_data_dict = dict()
        for pattern_len_idx, pattern_len in enumerate(pattern_len_rules.keys()):
            freq_pattern_mcpp_list_dict = dict()
            for rule in pattern_len_rules[pattern_len]:
                left_pattern, right_pattern, support, confidence, lift = rule
                frequent_pattern = GetFrequentPattern(left_pattern, right_pattern)
                if frequent_pattern not in freq_pattern_mcpp_list_dict:
                    freq_pattern_mcpp_list_dict[frequent_pattern] = list()
                freq_pattern_mcpp_list_dict[frequent_pattern].append((left_pattern, right_pattern, support, confidence, lift))

            # print len(freq_pattern_mcpp_list_dict)
            multivariate_freq_pattern_cnt = 0
            boxplot_data = list()
            for frequent_pattern_list in freq_pattern_mcpp_list_dict:
                is_multivariate = False
                for frequent_pattern in frequent_pattern_list:
                    if len(frequent_pattern) > 2:
                        is_multivariate = True
                if is_multivariate:
                    multivariate_freq_pattern_cnt += 1
                    # print frequent_pattern_list
                    conf_sup_ratio_list = list()
                    for pattern_info in freq_pattern_mcpp_list_dict[frequent_pattern_list]:
                        left_pattern = pattern_info[0]
                        is_condition = False
                        for mcpp_1 in left_pattern:
                            if len(mcpp_1) > 1:
                                is_condition = True
                                break
                        if not is_condition:
                            continue
                        print "%s\t%s\t%f\t%f\t%f" % (pattern_info[0], pattern_info[1], pattern_info[2], pattern_info[3], pattern_info[3] / pattern_info[2])
                        support = pattern_info[2]
                        confidence = pattern_info[3]
                        conf_sup_ratio = confidence / support
                        conf_sup_ratio_list.append(conf_sup_ratio)
                        # print conf_sup_ratio
                    boxplot_data.append(conf_sup_ratio_list)
            # print multivariate_freq_pattern_cnt
            # print boxplot_data
            if len(boxplot_data) > 0:
                pattern_len_boxplot_data_dict[pattern_len] = boxplot_data
            print "Pattern Len: %d\tAll FP: %d\tAll Conditional MCPP:%d" % (pattern_len, len(freq_pattern_mcpp_list_dict), multivariate_freq_pattern_cnt)
        # print pattern_len_boxplot_data_dict


        f = open('%s/%s'%(output_dir, 'pattern_len_boxplot_data_dict.pickle'), 'w')
        pickle.dump(pattern_len_boxplot_data_dict, f)
        f.close()
    else:
        pattern_len_boxplot_data_dict = pickle.load(open('%s/%s' % (output_dir, 'pattern_len_boxplot_data_dict.pickle')))

    fig, ax = plt.subplots(1, 1)
    overall_boxplot_data = list()
    pattern_len_list = sorted(pattern_len_boxplot_data_dict.keys())
    f = open("%s/%s" % (output_dir, 'sequence_only_result.txt'),'w')
    for pattern_len in pattern_len_list:
        boxplot_data = pattern_len_boxplot_data_dict[pattern_len]
        length_boxplot_data = list()
        for conf_sup_ratio_list in boxplot_data:
            # print conf_sup_ratio_list
            length_boxplot_data += conf_sup_ratio_list
        overall_boxplot_data.append(length_boxplot_data)
    ax.boxplot(overall_boxplot_data)
    for i, boxplot_data in enumerate(overall_boxplot_data):
        f.write("%d\t%f\n" % (i+1, np.median(boxplot_data)))
        ax.scatter([i+1]*len(boxplot_data), boxplot_data, facecolors='none', edgecolors='c')
    f.close()
    ax.margins(0.1, 0.1)
    # ax.get_xaxis().set_ticks([])
    ax.set_xticklabels(pattern_len_list)
    ax.set_ylim(bottom=1)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Usefulness')
    # fig.tight_layout()
    fig.set_size_inches(2, 1.5)
    plt.savefig(output_dir + "/confidence_support_ratio.pdf", bbox_inches='tight')
    '''
    pattern_len_list = sorted(pattern_len_boxplot_data_dict.keys())
    print pattern_len_list
    fig, axarr = plt.subplots(1, len(pattern_len_list))
    for pattern_len_idx, pattern_len in enumerate(pattern_len_list):
        boxplot_data = pattern_len_boxplot_data_dict[pattern_len]
        if len(pattern_len_list) == 1:
            ax = axarr
        else:
            ax = axarr[pattern_len_idx]
        overall_boxplot_data = list()
        for conf_sup_ratio_list in boxplot_data:
            # print conf_sup_ratio_list
            overall_boxplot_data += conf_sup_ratio_list
        print overall_boxplot_data
        # overall_ax.scatter([1] * len(overall_boxplot_data), overall_boxplot_data, facecolors='none', edgecolors='c')
        ax.boxplot(overall_boxplot_data)
        ax.margins(0.0, 0.1)
    fig.tight_layout()
    fig.set_size_inches(9, 9)
    plt.savefig(output_dir + "/confidence_support_ratio.pdf")
    # plt.show()
    '''


if __name__ == '__main__':
    # rules_dir_path = 'Paper_Dataset/Output/rules/rules'
    rules_dir_path = 'rules'
    output_dir = 'output'
    PlotPatternCountAsMinconf(rules_dir_path, output_dir)
