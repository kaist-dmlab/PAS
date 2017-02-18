import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "serif"
# matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.size': 8})


def PlotScalabilityResult(origin_performance_check_type_ruletype_time):
    result_file_path = 'output/performance_result_scalability.txt'
    f = open(result_file_path)
    setting_time_list = dict()
    for line in f:
        line = line.split('\n')[0]

        user = line.split('\t')[0]
        performance_check_type = line.split('\t')[1]
        mcpp_len = line.split('\t')[2]
        support = float(line.split('\t')[3])
        confidence = float(line.split('\t')[4])
        max_partial_len = int(line.split('\t')[5])
        rule1_on = line.split('\t')[6]
        rule2_on = line.split('\t')[7]
        # rule3_on = line.split('\t')[8]
        time = float(line.split('\t')[8])

        if support <= 0.5 and confidence <= 0.5:
            setting_key = (user, performance_check_type, support, confidence, max_partial_len, rule1_on, rule2_on)
            if setting_key not in setting_time_list:
                setting_time_list[setting_key] = dict()
            if mcpp_len not in setting_time_list[setting_key]:
                setting_time_list[setting_key][mcpp_len] = list()
            setting_time_list[setting_key][mcpp_len].append(time)

    setting_time_sum = dict()
    for setting_key in setting_time_list:
        user, performance_check_type, support, confidence, max_partial_len, rule1_on, rule2_on = setting_key
        all_time = 0
        for mcpp_len in setting_time_list[setting_key]:
            if mcpp_len in ['ALL']:
                setting_time_mean = np.mean(setting_time_list[setting_key][mcpp_len])
                all_time += setting_time_mean
        setting_time_sum[setting_key] = all_time# - setting_time_list[setting_key]['1']

    performance_check_type_ruletype_time = dict()
    for setting_key in setting_time_sum:
        user, performance_check_type, support, confidence, max_partial_len, rule1_on, rule2_on = setting_key

        setting_time = setting_time_sum[setting_key]
        print str(setting_key) + "\t" + str(setting_time)
        rule_type = (rule1_on, rule2_on)
        if performance_check_type not in performance_check_type_ruletype_time:
            performance_check_type_ruletype_time[performance_check_type] = dict()
        if rule_type not in performance_check_type_ruletype_time[performance_check_type]:
            performance_check_type_ruletype_time[performance_check_type][rule_type] = list()
        multiple_num = int(user.split('_')[1])
        performance_check_type_ruletype_time[performance_check_type][rule_type].append((support, confidence, max_partial_len, setting_time, multiple_num))

    # plt.clf()
    # fig = plt.figure()
    fig, ax = plt.subplots(1,1)

    line_list = list()
    label_list = list()
    for perform_idx, performance_check_type in enumerate(performance_check_type_ruletype_time):
        print performance_check_type
        for rule_type in performance_check_type_ruletype_time[performance_check_type]:
            print rule_type
            if performance_check_type == 'conf':
                # label = "All RULES"
                marker = 's'
                color = 'r'
            elif performance_check_type == 'sup':
                # label = "RULE_LEFT"
                marker = 'D'
                color = 'b'
            elif performance_check_type == 'partial':
                # label = "RULE_RIGHT"
                marker = 'o'
                color = 'g'

            if performance_check_type == 'conf':
                for target_value in [0.3]:
                    # Origin size data (x1)
                    linestyle = '-'
                    if target_value == 0.2:
                        linestyle = '--'
                    performance_info_list = origin_performance_check_type_ruletype_time[performance_check_type][rule_type]
                    x_y_list = list()
                    for performance_info in performance_info_list:
                        support, confidence, max_partial_len, setting_time = performance_info
                        if confidence == target_value:
                            x_y_list.append((1, setting_time))

                    # Big size data (xN)
                    performance_info_list = performance_check_type_ruletype_time[performance_check_type][rule_type]
                    for performance_info in performance_info_list:
                        support, confidence, max_partial_len, setting_time, multiple_num = performance_info
                        if confidence == target_value:
                            x_y_list.append((multiple_num, setting_time))
                    x_y_list = sorted(x_y_list, key=lambda x: x[0])
                    print x_y_list

                    x_list = [x_y[0] for x_y in x_y_list]
                    y_list = [x_y[1] for x_y in x_y_list]
                    # label = "SUP:%.2f, CONF:%.2f, GAP:%d" % (support, target_value, max_partial_len-1)
                    # label = "conf:%.1f" % (target_value)
                    # label = "(%.2f,%.2f,%d)" % (target_value,support,max_partial_len)
                    label = "(%.2f,%.2f)" % (support, target_value)

                    line, = ax.plot(x_list, y_list, linestyle=linestyle, label=label, marker=marker, c=color)
                    line_list.append(line)
                    label_list.append(label)
            elif performance_check_type == 'sup':
                for target_value in [0.15]:
                    linestyle = '-'
                    if target_value == 0.2:
                        linestyle = '--'
                    # Origin size data (x1)
                    performance_info_list = origin_performance_check_type_ruletype_time[performance_check_type][rule_type]
                    x_y_list = list()
                    for performance_info in performance_info_list:
                        support, confidence, max_partial_len, setting_time = performance_info
                        if support == target_value:
                            x_y_list.append((1, setting_time))

                    # Big size data (xN)
                    performance_info_list = performance_check_type_ruletype_time[performance_check_type][rule_type]
                    for performance_info in performance_info_list:
                        support, confidence, max_partial_len, setting_time, multiple_num = performance_info
                        if support == target_value:
                            print "%f\t%f\t%d\t%f"%(support, confidence, multiple_num, setting_time)
                            x_y_list.append((multiple_num, setting_time))
                    x_y_list = sorted(x_y_list, key=lambda x: x[0])
                    print x_y_list

                    x_list = [x_y[0] for x_y in x_y_list]
                    y_list = [x_y[1] for x_y in x_y_list]
                    # label = "(%.2f,%.2f,%d)" % (confidence,target_value,max_partial_len)
                    label = "(%.2f,%.2f)" % (target_value, confidence)

                    line, = ax.plot(x_list, y_list, linestyle=linestyle, label=label, marker=marker, c=color)
                    line_list.append(line)
                    label_list.append(label)
            elif performance_check_type == 'partial':
                for target_value in [1]:
                    linestyle = '-'
                    '''
                    if target_value == 1:
                        linestyle = '--'
                    '''
                    # Origin size data (x1)
                    performance_info_list = origin_performance_check_type_ruletype_time[performance_check_type][rule_type]
                    x_y_list = list()
                    for performance_info in performance_info_list:
                        support, confidence, max_partial_len, setting_time = performance_info
                        if max_partial_len == target_value:
                            x_y_list.append((1, setting_time))

                    # Big size data (xN)
                    performance_info_list = performance_check_type_ruletype_time[performance_check_type][rule_type]
                    for performance_info in performance_info_list:
                        support, confidence, max_partial_len, setting_time, multiple_num = performance_info
                        if max_partial_len == target_value:
                            x_y_list.append((multiple_num, setting_time))
                    x_y_list = sorted(x_y_list, key=lambda x: x[0])
                    print x_y_list

                    x_list = [x_y[0] for x_y in x_y_list]
                    y_list = [x_y[1] for x_y in x_y_list]
                    # label = "gap:%d" % (target_value)
                    label = "(%.2f,%.2f)" % (support ,confidence)
                    print label
                    print color
                    print x_list
                    print y_list
                    line, = ax.plot(x_list, y_list, linestyle=linestyle, label=label, marker=marker, c=color)
                    line_list.append(line)
                    label_list.append(label)
    ax.set_xlabel('Data Size', labelpad=1)
    ax.set_ylim([0, 1000])
    ax.set_xlim([1, 5])
    ax.set_xticklabels(['1x','2x','3x','4x','5x'])
    ax.set_ylabel('Running Time (sec)')
    # ax.annotate('min_supp', xy=(0.5,1.2), xytext=(0.1,1.1), fontsize=7, xycoords='axes fraction',  arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=0.4))
    # ax.annotate('min_conf', xy=(0.0,1.0), xytext=(0.6,1.1), fontsize=7, xycoords='axes fraction',  arrowprops=dict(facecolor='black', shrink=0.05, width=0.5))


    ax.text(1.9,1060, 'min_supp', fontsize=6)
    ax.text(3.4,1060, 'min_conf', fontsize=6)
    ax.annotate('', xy=(0.5,1.2), xytext=(0.5,1.15), fontsize=7, xycoords='axes fraction',  arrowprops=dict(facecolor='black', shrink=0.01, width=0.01, headwidth=2))
    ax.annotate('', xy=(0.65,1.2), xytext=(0.65,1.15), fontsize=7, xycoords='axes fraction',  arrowprops=dict(facecolor='black', shrink=0.01, width=0.01, headwidth=2))

    # ax.arrow(0, 0, 3,1000, head_width=0.05, head_length=0.1, xycoords='axes fraction')
    # ax.margins(0.2, 0.2)
    # ax.xaxis.set_data_interval(0.1)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    
    fig.set_size_inches(1.95,1.5)
    fig.tight_layout()
    '''
    new_label_list = ['No Rule', 'Backward Prune', 'Forward Stop + Prune', 'All Rules']
    new_line_list = list()
    for label in new_label_list:
        line_idx = label_list.index(label)
        line = line_list[line_idx]
        new_line_list.append(line)
    '''
    # lgd = ax.legend(line_list, label_list, loc='upper center', ncol=3, bbox_to_anchor=(1.0, 1.1),fontsize=7, frameon=False)
    lgd = plt.figlegend(line_list, label_list, loc='upper center', ncol=3, bbox_to_anchor=(0.65, 1.1),fontsize=7, frameon=False, columnspacing=0.5)
    plt.savefig("scalability_result_36647_3.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


def PlotPerformanceResult(result_file_path):
    f = open(result_file_path)
    setting_time_list = dict()
    for line in f:
        line = line.split('\n')[0]

        user = line.split('\t')[0]
        performance_check_type = line.split('\t')[1]
        mcpp_len = line.split('\t')[2]
        support = float(line.split('\t')[3])
        confidence = float(line.split('\t')[4])
        max_partial_len = int(line.split('\t')[5])
        rule1_on = line.split('\t')[6]
        rule2_on = line.split('\t')[7]
        # rule3_on = line.split('\t')[8]
        time = float(line.split('\t')[8])

        if support <= 0.5 and confidence <= 0.5:
            setting_key = (user, performance_check_type, support, confidence, max_partial_len, rule1_on, rule2_on)
            if setting_key not in setting_time_list:
                setting_time_list[setting_key] = dict()
            if mcpp_len not in setting_time_list[setting_key]:
                setting_time_list[setting_key][mcpp_len] = list()
            setting_time_list[setting_key][mcpp_len].append(time)

    setting_time_sum = dict()
    for setting_key in setting_time_list:
        user, performance_check_type, support, confidence, max_partial_len, rule1_on, rule2_on = setting_key
        all_time = 0
        for mcpp_len in setting_time_list[setting_key]:
            if mcpp_len in ['ALL']:
                setting_time_mean = np.mean(setting_time_list[setting_key][mcpp_len])
                all_time += setting_time_mean
        setting_time_sum[setting_key] = all_time# - setting_time_list[setting_key]['1']

    performance_check_type_ruletype_time = dict()
    for setting_key in setting_time_sum:
        user, performance_check_type, support, confidence, max_partial_len, rule1_on, rule2_on = setting_key

        setting_time = setting_time_sum[setting_key]
        print str(setting_key) + "\t" + str(setting_time)
        rule_type = (rule1_on, rule2_on)
        if performance_check_type not in performance_check_type_ruletype_time:
            performance_check_type_ruletype_time[performance_check_type] = dict()
        if rule_type not in performance_check_type_ruletype_time[performance_check_type]:
            performance_check_type_ruletype_time[performance_check_type][rule_type] = list()
        performance_check_type_ruletype_time[performance_check_type][rule_type].append((support, confidence, max_partial_len, setting_time))

    # PlotScalabilityResult(origin_performance_check_type_ruletype_time=performance_check_type_ruletype_time)
    fig, axarr = plt.subplots(1, len(performance_check_type_ruletype_time.keys()))
    line_list = list()
    label_list = list()
    for perform_idx, performance_check_type in enumerate(performance_check_type_ruletype_time):
        print performance_check_type
        if performance_check_type == 'sup':
            ax = axarr[0]
        elif performance_check_type == 'conf':
            ax = axarr[1]
        elif performance_check_type == 'partial':
            ax = axarr[2]
        # ax = axarr
        # ax = axarr[perform_idx]
        for rule_type in performance_check_type_ruletype_time[performance_check_type]:
            if rule_type == ('True', 'True'):
                label = "All Rules"
                marker = 's'
                color = 'r'
                zorder = 4
            elif rule_type == ('True', 'False'):
                label = "Forward Stop + Prune"
                marker = 'D'
                color = 'b'
                zorder = 3
            elif rule_type == ('False', 'True'):
                label = "Backward Prune"
                marker = 'o'
                color = 'g'
                zorder = 2
            elif rule_type == ('False', 'False'):
                label = "No Rule"
                marker = 's'
                color = 'k'
                zorder = 1
            else:
                continue
            performance_info_list = performance_check_type_ruletype_time[performance_check_type][rule_type]
            if performance_check_type == 'conf':
                performance_info_list = sorted(performance_info_list, key=lambda x: x[1])
                x_list = [p_info[1] for p_info in performance_info_list]
                y_list = [p_info[3] for p_info in performance_info_list]
            elif performance_check_type == 'sup':
                performance_info_list = sorted(performance_info_list, key=lambda x: x[0])
                x_list = [p_info[0] for p_info in performance_info_list]
                y_list = [p_info[3] for p_info in performance_info_list]
            elif performance_check_type == 'partial':
                performance_info_list = sorted(performance_info_list, key=lambda x: x[2])
                x_list = [p_info[2] for p_info in performance_info_list]
                y_list = [p_info[3] for p_info in performance_info_list]
            if rule_type == ('False', 'False', 'False'):
                line, = ax.plot(x_list, y_list, label=label, marker=marker, markerfacecolor='none', c=color, zorder=zorder)
            else:#if rule_type == ('True', 'True'):
                line, = ax.plot(x_list, y_list, label=label, marker=marker, c=color, zorder=zorder)
            if perform_idx == 0:
                line_list.append(line)
                label_list.append(label)
        # ax.legend(loc='best')
        if performance_check_type == 'conf':
            ax.set_xlabel('Minimum Confidence', labelpad=1)
            ax.set_ylim([0, 150])
            ax.set_xlim([0.1, 0.5])
        elif performance_check_type == 'sup':
            ax.set_xlabel('Minimum Support', labelpad=1)
            ax.set_ylim([0, 150])
            ax.set_xlim([0.1, 0.2])
        elif performance_check_type == 'partial':
            ax.set_xlabel('Maximum Gap', labelpad=1)
            ax.set_ylim([0, 600])
            ax.set_xlim([1, 5])
        if perform_idx == 2:
            ax.set_ylabel('Running Time (sec)')
            ax.set_xticks([0.1,0.125,0.15,0.175,0.2])
            ax.set_xticklabels([0.10,0.125,0.15,0.175,0.2])
        else:
            ax.locator_params(axis='x', nbins=5)
            ax.yaxis.labelpad=0.1
    
        ax.margins(0.2, 0.2)
        # ax.xaxis.set_data_interval(0.1)
        # ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)
    fig.set_size_inches(5.0, 1.5)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    new_label_list = ['No Rule', 'Backward Prune', 'Forward Stop + Prune', 'All Rules']
    new_line_list = list()
    for label in new_label_list:
        line_idx = label_list.index(label)
        line = line_list[line_idx]
        new_line_list.append(line)
    
    lgd = plt.figlegend(new_line_list, new_label_list, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=7, frameon=False)
    plt.savefig("output/performance_result.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    result_file_path = 'output/performance_result.txt'
    PlotPerformanceResult(result_file_path)
    # PlotScalabilityResult()
