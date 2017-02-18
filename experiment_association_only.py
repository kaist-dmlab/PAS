import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import ast
import operator
import pickle
import pandas as pd
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches


# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams.update({'font.size': 8})



def GetCategory(class_label):
    category = None
    '''
    if class_label in ['volume_ring[disc]', 'volume_system[disc]', 'volume_notification[disc]', 'volume_music[disc]']:
        category = 'VOLUME'
    '''
    if class_label in ['display_orientation[cat]']:
        category = 'APP'
    elif class_label in ['class']:
        category = 'Availability'
    '''
    elif 'battery' in class_label:
        category = 'BATTERY'
    '''
    return category


def Entropy(values):
    value_cnt_dict = dict()
    for value in values:
        if value not in value_cnt_dict:
            value_cnt_dict[value] = 0
        value_cnt_dict[value] += 1
    value_ratio_list = list()
    for value in value_cnt_dict:
        value_ratio_list.append(value_cnt_dict[value] / float(sum(value_cnt_dict.values())))
    # return max(value_ratio_list), value_cnt_dict

    if len(value_ratio_list) == 1:
        entropy = 1.0
    else:
        entropy = stats.entropy(value_ratio_list) / np.log(len(value_ratio_list)) # Normalized
        entropy = 1 - entropy
    return entropy, value_cnt_dict


def GetEntropyEverage(entropy_info_list, granularity_min):
    time_key_class_values_dict = dict()
    for class_info in entropy_info_list:
        class_timestamp = class_info[0]
        class_value = class_info[1]
        granularity_timestamp = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([class_timestamp]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
        daily_time = granularity_timestamp.strftime("%H:%M:%S")
        time_key = "time_daily:" + str(daily_time)
        if time_key not in time_key_class_values_dict:
            time_key_class_values_dict[time_key] = list()
        time_key_class_values_dict[time_key].append(class_value)
    entropy_list = list()
    instance_num_list = list()
    for time_key in time_key_class_values_dict:
        # print Entropy(time_key_class_values_dict[time_key])[0]
        entropy_list.append(Entropy(time_key_class_values_dict[time_key])[0])
        instance_num_list.append(len(time_key_class_values_dict[time_key]))
    return np.mean(entropy_list), instance_num_list

def GetTimesFromPattern(patterns):
    time_list = list()
    for pattern in patterns:
        for item in pattern:
            # print item
            if "time_daily" in item:
                time_list.append(item)
    return tuple(sorted(time_list))

def PlotEntropyForPatternPair(entropy_result_path, output_dir, granularity_min):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if True or not os.path.exists(output_dir + '/sequence_only_result_class.pickle'):
        getall = [[files, os.path.getsize(entropy_result_path + "/" + files)] for files in os.listdir(entropy_result_path)]
        file_info_list = list()
        for file_name, file_size in sorted(getall, key=operator.itemgetter(1)):
            if file_name[-7:] == '.pickle' and 'U8' not in file_name:
                file_info_list.append(file_name)
        print "# All Processed Users: %d" % len(file_info_list)
        print file_info_list
        class_label_user_pattern_pair_info = dict()
        for i, user_result_path in enumerate(file_info_list):
            user_result_path = "%s/%s" % (entropy_result_path, user_result_path)
            print "%d/%d - %s" % (i, len(file_info_list), user_result_path)
            user_result = pickle.load(open(user_result_path))
            print "Loaded"
            for line_info in user_result:
                user = line_info[0]
                class_label = line_info[1]
                if class_label != 'class':#'class':
                    continue
                mcpp_len = line_info[2]
                condition_len = line_info[3]
                fully_pattern = str(line_info[4])
                unfully_pattern = str(line_info[5])
                fully_class_list = line_info[6]
                unfully_class_list = line_info[7]
                fully_class_cnt = len(fully_class_list)
                unfully_class_cnt = len(unfully_class_list)
                # fully_entropy = line_info[8]
                # unfully_entropy = line_info[9]
                fully_entropy, fully_instance_num_list = GetEntropyEverage(fully_class_list, granularity_min)
                unfully_entropy, unfully_instance_num_list = GetEntropyEverage(unfully_class_list, granularity_min)

                print (user, class_label, mcpp_len, condition_len, str(fully_instance_num_list), str(unfully_instance_num_list), fully_entropy, unfully_entropy)

                '''
                if condition_len <= 1:
                    continue
                '''
                if mcpp_len != len(fully_instance_num_list) or mcpp_len != len(unfully_instance_num_list):
                    continue

                for instance_num in fully_instance_num_list + unfully_instance_num_list:
                    if instance_num < 5:
                        continue

                if class_label not in class_label_user_pattern_pair_info:
                    class_label_user_pattern_pair_info[class_label] = dict()

                if user not in class_label_user_pattern_pair_info[class_label]:
                    class_label_user_pattern_pair_info[class_label][user] = list()
                class_label_user_pattern_pair_info[class_label][user].append((user, class_label, mcpp_len, condition_len, fully_pattern, unfully_pattern, fully_class_cnt, unfully_class_cnt, fully_entropy, unfully_entropy))
        user_feature_pattern_pair_info_list_dict = dict()
        user_fully_unfully_time_class_cnt = dict()
        for class_label in class_label_user_pattern_pair_info:
            user_plot_data = dict()
            print class_label_user_pattern_pair_info[class_label].keys()
            for user in class_label_user_pattern_pair_info[class_label]:
                pattern_pair_info_list = class_label_user_pattern_pair_info[class_label][user]
                fully_entropy_list = list()
                unfully_entropy_list = list()
                for pattern_pair_info in pattern_pair_info_list:
                    condition_len = pattern_pair_info[3]
                    fully_class_cnt = pattern_pair_info[-4]
                    fully_entropy = pattern_pair_info[-2]
                    unfully_class_cnt = pattern_pair_info[-3]
                    unfully_entropy = pattern_pair_info[-1]

                    '''
                    if condition_len <= 1 or fully_class_cnt < 10 or unfully_class_cnt < 10:
                        continue
                    '''
                    # Store the number of class in fully time and unfully time
                    if user not in user_fully_unfully_time_class_cnt:
                        user_fully_unfully_time_class_cnt[user] = dict()

                    if class_label not in user_fully_unfully_time_class_cnt[user]:
                        user_fully_unfully_time_class_cnt[user][class_label] = dict()
                        user_fully_unfully_time_class_cnt[user][class_label]["fully"] = dict()
                        user_fully_unfully_time_class_cnt[user][class_label]["unfully"] = dict()

                    fully_left = ast.literal_eval(pattern_pair_info[4].split('-')[0])
                    unfully_left = tuple([ast.literal_eval(unfully_pattern.split('-')[0])[0] for unfully_pattern in ast.literal_eval(pattern_pair_info[5])])

                    fully_timelist = GetTimesFromPattern(fully_left)
                    unfully_timelist = GetTimesFromPattern(unfully_left)

                    user_fully_unfully_time_class_cnt[user][class_label]["fully"][fully_timelist] = fully_class_cnt
                    user_fully_unfully_time_class_cnt[user][class_label]["unfully"][unfully_timelist] = unfully_class_cnt

                    # print pattern_pair_info
                    fully_entropy_list.append(fully_entropy)
                    unfully_entropy_list.append(unfully_entropy)
                    if user not in user_feature_pattern_pair_info_list_dict:
                        user_feature_pattern_pair_info_list_dict[user] = dict()
                    if class_label not in user_feature_pattern_pair_info_list_dict[user]:
                        user_feature_pattern_pair_info_list_dict[user][class_label] = list()
                    # DONE Store fully, unfully of user for the class label
                    user_feature_pattern_pair_info_list_dict[user][class_label].append(pattern_pair_info)

                if len(fully_entropy_list) == 0:
                    continue
                # print len(fully_entropy_list)
                user_plot_data[user] = (fully_entropy_list, unfully_entropy_list)

        ### Result Analysis ###
        # 1. Select feature for user as increasing entropy difference.
        # 2. User grouping based on selected features. (3~4 features)
        print "# Users: %d" % (len(user_feature_pattern_pair_info_list_dict.keys()))
        user_category_pair_avg_entropy_diff_dict = dict()
        class_label_user_list_dict = dict() # Plotting distribution features in the top 3 entorpy diff features of a each user.
        for user in user_feature_pattern_pair_info_list_dict:
            class_label_entropy_diff_list = list()
            for class_label in user_feature_pattern_pair_info_list_dict[user]:
                pattern_pair_info_list = user_feature_pattern_pair_info_list_dict[user][class_label]
                entropy_diff_list = [pattern_pair_info[-2] - pattern_pair_info[-1] for pattern_pair_info in pattern_pair_info_list]
                entropy_diff_mean = np.mean(entropy_diff_list)
                class_label_entropy_diff_list.append((user, class_label, entropy_diff_mean))
            class_label_entropy_diff_list = sorted(class_label_entropy_diff_list, key=lambda x: x[2])
            for class_label_entropy_diff in class_label_entropy_diff_list[:3]:
                print "%s\t%s\t%f" % class_label_entropy_diff
                class_label = class_label_entropy_diff[1]
                if class_label not in class_label_user_list_dict:
                    class_label_user_list_dict[class_label] = list()
                class_label_user_list_dict[class_label].append(user)
            # DONE Calculating average entropy of subfeatures in the feature category(App, Volume, Battery).
            class_category_feature_entropy_diff_list_dict = dict()
            for class_label_entropy_diff in class_label_entropy_diff_list:
                class_label = class_label_entropy_diff[1]
                category = GetCategory(class_label)
                if category is not None:
                    if category not in class_category_feature_entropy_diff_list_dict:
                        class_category_feature_entropy_diff_list_dict[category] = list()
                    class_category_feature_entropy_diff_list_dict[category].append(class_label_entropy_diff)
            for category in class_category_feature_entropy_diff_list_dict:
                category_feature_entropy_diff_list = class_category_feature_entropy_diff_list_dict[category]
                # for category_feature_entropy_diff in category_feature_entropy_diff_list:
                #     print "%s\t%s\t%f" % category_feature_entropy_diff
                category_feature_entropy_diff_list = [entropy_info[2] for entropy_info in category_feature_entropy_diff_list]
                category_feature_entropy_diff_mean = np.mean(category_feature_entropy_diff_list) # Mean of entropy diff between pattern pair on subfeatures of the category feature
                print ">> %s\t%s\t%f" % (user, category, category_feature_entropy_diff_mean)
                if user not in user_category_pair_avg_entropy_diff_dict:
                    user_category_pair_avg_entropy_diff_dict[user] = dict()
                user_category_pair_avg_entropy_diff_dict[user][category] = category_feature_entropy_diff_mean

        ax = plt.subplot(1, 1, 1)
        x_list = class_label_user_list_dict.keys()
        y_list = [len(class_label_user_list_dict[class_label]) for class_label in x_list]
        ind = np.arange(len(x_list))
        width = 0.35
        ax.bar(ind, y_list, width, align='center')
        ax.set_ylabel('# Users')
        ax.set_xlabel('Features of Top 3 in a each user')
        ax.set_xticks(ind)
        ax.set_xticklabels(x_list, rotation=270)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(0.02, 0.0)

        # plt.show()
        fig = plt.gcf()
        fig.set_size_inches(3, 3)
        fig.tight_layout()
        plt.savefig(output_dir + "/feature_distribution.pdf")
        plt.clf()

        # Plotting distribution of entropy in the sequential and nonsequential pattern
        user_class_category_pattern_type_feature_pattern_info_list_dict = dict()
        for user in user_feature_pattern_pair_info_list_dict:
            class_label_entropy_diff_list = list()
            for class_label in user_feature_pattern_pair_info_list_dict[user]:
                category = GetCategory(class_label)
                if category is not None:
                    if user not in user_class_category_pattern_type_feature_pattern_info_list_dict:
                        user_class_category_pattern_type_feature_pattern_info_list_dict[user] = dict()
                    if category not in user_class_category_pattern_type_feature_pattern_info_list_dict[user]:
                        user_class_category_pattern_type_feature_pattern_info_list_dict[user][category] = dict()
                        user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["FULLY"] = dict()
                        user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["UNFULLY"] = dict()
                    pattern_pair_info_list = user_feature_pattern_pair_info_list_dict[user][class_label]
                    for pattern_pair_info in pattern_pair_info_list:
                        fully_pattern = pattern_pair_info[4]
                        fully_entropy = pattern_pair_info[-2]
                        unfully_pattern = pattern_pair_info[5]
                        unfully_entropy = pattern_pair_info[-1]
                        if fully_pattern not in user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["FULLY"]:
                            user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["FULLY"][fully_pattern] = dict()
                        if unfully_pattern not in user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["UNFULLY"]:
                            user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["UNFULLY"][unfully_pattern] = dict()

                        user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["FULLY"][fully_pattern][class_label] = fully_entropy
                        user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["UNFULLY"][unfully_pattern][class_label] = unfully_entropy
                        # Complete to check with manual result


        def GetAllTimesFromTimeKeyList(time_key_list):
            all_time_key_list = set()
            for time_key in time_key_list:
                for time in time_key:
                    all_time_key_list.add(time)
            return all_time_key_list

        # Availability Instance Counts
        # class_label = 'class'
        # category = 'Availability'
        class_label = 'class'
        category = 'Availability'
        category_class_count_list = dict()
        for user in user_fully_unfully_time_class_cnt:
            if class_label in user_fully_unfully_time_class_cnt[user]:
                fully_class_count_list = list()
                unfully_class_count_list = list()
                for time_key in user_fully_unfully_time_class_cnt[user][class_label]['fully']:
                    time_len = len(time_key)
                    class_cnt = user_fully_unfully_time_class_cnt[user][class_label]['fully'][time_key]
                    class_cnt_mean = class_cnt # / float(time_len)
                    # print class_cnt_mean
                    fully_class_count_list.append(class_cnt_mean)
                for time_key in user_fully_unfully_time_class_cnt[user][class_label]['unfully']:
                    time_len = len(time_key)
                    class_cnt = user_fully_unfully_time_class_cnt[user][class_label]['unfully'][time_key]
                    class_cnt_mean = class_cnt # / float(time_len)
                    # print class_cnt_mean
                    unfully_class_count_list.append(class_cnt_mean)
                fully_class_count_mean = np.mean(fully_class_count_list)
                unfully_class_count_mean = np.mean(unfully_class_count_list)
                all_class_count_mean = (fully_class_count_mean + unfully_class_count_mean)
                # all_class_count_mean = fully_pattern_count
                if category not in category_class_count_list:
                    category_class_count_list[category] = list()
                category_class_count_list[category].append((user, class_label, all_class_count_mean))

        f = open(output_dir + '/sequence_only_result_class.pickle', 'w')
        pickle.dump(user_class_category_pattern_type_feature_pattern_info_list_dict, f)
        f.close()
        f = open(output_dir + '/sequence_only_category_class_count_list.pickle', 'w')
        pickle.dump(category_class_count_list, f)
        f.close()

    else:
        user_class_category_pattern_type_feature_pattern_info_list_dict = pickle.load(open(output_dir + '/sequence_only_result_class.pickle'))
        category_class_count_list = pickle.load(open(output_dir + '/sequence_only_category_class_count_list.pickle'))

    category = 'Availability'
    class_top_k_list = [len(category_class_count_list[category])]
    x_label = ["PAS", "A-Only"]
    ind = [1, 2]
    fig, axarr = plt.subplots(1, 1)
    # category = 'Availability'
    category = 'Availability'
    for class_top_k_idx, class_top_k in enumerate(class_top_k_list):
        user_class_cnt_list = category_class_count_list[category] # user_fully_unfully_time_cnt_dict[category]
        user_class_cnt_list = sorted(user_class_cnt_list, key=lambda x:x[2], reverse=True)
        user_class_cnt_list = user_class_cnt_list[:class_top_k]

        entropy_diff_list = list()
        overall_fully_entropy_list = list()
        overall_unfully_entropy_list = list()
        for user_idx, user_class_count in enumerate(user_class_cnt_list):
            user = user_class_count[0]
            fully_x_list = list()
            fully_y_list = list()
            unfully_x_list = list()
            unfully_y_list = list()
            fully_pattern_time_list = list()
            unfully_pattern_time_list = list()
            for fully_pattern in user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["FULLY"]:
                fully_left = ast.literal_eval(fully_pattern.split('-')[0])
                fully_timelist = GetTimesFromPattern(fully_left)
                fully_pattern_entropy_list = user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["FULLY"][fully_pattern].values()
                fully_pattern_entropy_mean = np.mean(fully_pattern_entropy_list)
                if fully_timelist not in fully_pattern_time_list:
                    fully_x_list.append(ind[0])
                    fully_y_list.append(fully_pattern_entropy_mean)
                    fully_pattern_time_list.append(fully_timelist)
            for unfully_pattern in user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["UNFULLY"]:
                unfully_left = tuple([ast.literal_eval(unfully_pattern_item.split('-')[0])[0] for unfully_pattern_item in ast.literal_eval(unfully_pattern)])
                unfully_timelist = GetTimesFromPattern(unfully_left)
                unfully_pattern_entropy_list = user_class_category_pattern_type_feature_pattern_info_list_dict[user][category]["UNFULLY"][unfully_pattern].values()
                unfully_pattern_entropy_mean = np.mean(unfully_pattern_entropy_list)
                if unfully_timelist not in unfully_pattern_time_list:
                    unfully_x_list.append(ind[1])
                    unfully_y_list.append(unfully_pattern_entropy_mean)
                    unfully_pattern_time_list.append(unfully_timelist)
                    # print unfully_timelist
            # print fully_y_list
            # print unfully_y_list
            # print "%s\t%d\t%d\t%d\t%d\t%f" % (user, len(fully_pattern_time_list), len(unfully_pattern_time_list), len(fully_pattern_time_list) + len(unfully_pattern_time_list), user_class_count[2], np.median(fully_y_list) - np.median(unfully_y_list))
            entropy_diff_list.append(np.median(fully_y_list) - np.median(unfully_y_list))
            # overall_fully_entropy_list += fully_y_list
            # overall_unfully_entropy_list += unfully_y_list
            overall_fully_entropy_list.append(np.median(fully_y_list))
            overall_unfully_entropy_list.append(np.median(unfully_y_list))

        overall_ax = axarr#[class_top_k_idx]
        overall_ax.scatter([ind[0]] * len(overall_fully_entropy_list), overall_fully_entropy_list, facecolors='none', edgecolors='c')
        overall_ax.scatter([ind[1]] * len(overall_unfully_entropy_list), overall_unfully_entropy_list, facecolors='none', edgecolors='c')
        bp = overall_ax.boxplot([overall_fully_entropy_list, overall_unfully_entropy_list], widths=(0.3, 0.3))
        print "User Cnt: %d" % len(user_class_cnt_list)
        print "Consistency in PAS: %f" % bp['medians'][0].get_ydata()[0]
        print "Consistency in A-Only: %f" % bp['medians'][1].get_ydata()[0]
        '''
        print "-----"
        print np.mean(overall_fully_entropy_list) - np.mean(overall_unfully_entropy_list)
        print np.median(overall_fully_entropy_list) - np.median(overall_unfully_entropy_list)
        print np.mean(entropy_diff_list)
        print np.median(entropy_diff_list)
        print "-----"
        print np.median(overall_fully_entropy_list)
        print np.median(overall_unfully_entropy_list)
        print bp['medians'][0].get_ydata()[0]
        print bp['medians'][1].get_ydata()[0]
        print bp['medians'][0].get_ydata()[0] - bp['medians'][1].get_ydata()[0] # Median in boxplot
        '''
        if class_top_k_idx == 0:
            overall_ax.set_ylabel('Consistency')
            # overall_ax.set_xlabel('Top %d Users' % (class_top_k))
        # else:
            # overall_ax.set_xlabel('All Users')
        # overall_ax.text(0.8, 0.8, )
        # result_text = "%f" % (bp['medians'][0].get_ydata()[0] - bp['medians'][1].get_ydata()[0])
        # ax.text(0.5, 0.5, result_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15)
        # ax.set_xticks(ind)
        # overall_ax.xaxis.tick_top()
        overall_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        overall_ax.set_xticklabels(x_label)
        overall_ax.margins(0.1, 0.1)
        overall_ax.set_xlabel('1')
        overall_ax.xaxis.label.set_color('white')
        diff_text = "%.3f" % (bp['medians'][0].get_ydata()[0] - bp['medians'][1].get_ydata()[0])
        diff_percen = "%.1f" % ((bp['medians'][0].get_ydata()[0] - bp['medians'][1].get_ydata()[0]) / bp['medians'][1].get_ydata()[0])

        # overall_ax.text(1.27,0.25, diff_percen, fontsize=7)
        overall_ax.arrow(1.85,0.200,-0.65,0.035)
    fig.set_size_inches(2, 1.5)
    fig.tight_layout()
    # matplotlib.rcParams.update({'font.size': 30})
    plt.savefig(output_dir + "/entropy_distribution.pdf", bbox_inches='tight')
    # plt.show()

    print len(overall_fully_entropy_list)
    print len(overall_unfully_entropy_list)
    user_class_cnt_list = user_class_cnt_list[:class_top_k]
    print len(user_class_cnt_list)

    fig, ax = plt.subplots(1)
    ind = np.arange(len(user_class_cnt_list))

    for x in ind:
        entropy_diff = overall_fully_entropy_list[x] - overall_unfully_entropy_list[x]
        if entropy_diff > 0:
            # ax.plot([x, x], [overall_fully_entropy_list[x], overall_unfully_entropy_list[x]], c='r', linewidth=1.5)
            width = 0.4
            p = patches.Rectangle(
                (x-width/2.0, overall_fully_entropy_list[x]),
                width, overall_unfully_entropy_list[x]-overall_fully_entropy_list[x],
                hatch='////',
                fill=False,
                edgecolor="red"
            )
            ax.add_patch(p)
        else:
            # ax.plot([x, x], [overall_unfully_entropy_list[x], overall_fully_entropy_list[x]], c='b', linewidth=1.5)
            p = patches.Rectangle(
                (x-width/2.0, overall_unfully_entropy_list[x]),
                width, overall_fully_entropy_list[x]-overall_unfully_entropy_list[x],
                fill=False,
                edgecolor="blue"
            )
            ax.add_patch(p)

    ax.plot(ind, overall_fully_entropy_list, 'ro', c='k', marker='o', markerfacecolor='black', label='1', markersize=5)
    ax.plot(ind, overall_unfully_entropy_list, 'ro', c='k', marker='o', markerfacecolor='None', label='2', markersize=5)


    ax.set_xticks(ind)
    ax.set_xticklabels([str(x+1) for x in ind])
    ax.set_xlabel('User Index', labelpad=1)
    ax.set_ylabel('Consistency')
    ax.margins(0.1, 0.1)
    # ax.set_position([0.2,0.2,0.5,0.8])
    lgd = ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2), fontsize=7, frameon=False)
    fig.set_size_inches(2, 1.5)
    fig.tight_layout()
    # matplotlib.rcParams.update({'font.size': 30})
    plt.savefig(output_dir + "/entropy_distribution_all_users.pdf", bbox_inches='tight')

    # plt.show()








if __name__ == '__main__':
    entropy_result_path = 'entropy_result'
    output_dir = 'output'
    PlotEntropyForPatternPair(entropy_result_path, output_dir, granularity_min=60)
