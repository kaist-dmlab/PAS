Discovering Periodic Association Sequences from Multivariate Time Series
=========================
## Overview
Because periodicity is commonly observed in the physical world, a number of research efforts have been devoted to detect periodic patterns in time series data. However, existing approaches have limitations in describing complex repeating behavior, such as human routine behavior, because they focus on either temporal sequence of repeating events or multivariate association in repeating events. We contend that supporting both properties plays a pivotal role in enriching the expressive power of periodic patterns. Toward this goal, in this paper, we propose a novel periodic pattern called periodic association sequence (PAS), which is represented by a sequence of periodic (or cyclic) association rules rather than typical frequent itemsets. Hence, its primary advantage is the capability of describing repeating consequences in both temporal and multivariate dimensions. Then, we present an efficient algorithm and mechanism for detecting PASs from a large-scale multivariate time-series database. Last, using two real-world smartphone log data sets, we conduct extensive experiments to demonstrate the benefits of the PAS. The evaluation results confirm that PASs are more coherent to human behavior and achieve higher confidence than state-of-the-art periodic patterns. The source code and data are available at https://github.com/jaegil/PAS.

## Code
- The binary code used in the paper in available at https://github.com/jaegil/PAS.
- Language: Python 2.7

## Data Sets
| Name            | Size          | Link                 | Description                          |
| :-------------- | :-----------: | -------------------: |-------------------------------------:|
| Device Analyzer | xxx GB        | Link1                | 27 users smartphone usage data       |
| KAIST           | xx MB         | Link2                | 541 users user smartphone usage data |

## How to run
1. Install: python2.7, pandas, numpy
2. Set configuration in pas_mining.py
  - dir_path: a path of dataset
  - output_dir: a path of result of preprocessing dataset
  - rules_dir: a path of result of PAS mining
  - granularity_min: time span size (in minutes)
  - MIN_DATE_LENGTH: the minimum number of days users have
3. Set paramters in pas_mining.py
  - min_sup: the minimum support
  - min_conf: the minimum confidence
  - max_gap: the maximum gap
4. Run 'python pas_mining.py'
  
