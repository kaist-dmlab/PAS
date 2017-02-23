Discovering Periodic Association Sequences from Multivariate Time Series
=========================
## Overview
Because periodicity is commonly observed in the physical world, a number of research efforts have been devoted to detect periodic patterns in time series data. However, existing approaches have limitations in describing complex repeating behavior, such as human routine behavior, because they focus on either temporal sequence of repeating events or multivariate association in repeating events. We contend that supporting both properties plays a pivotal role in enriching the expressive power of periodic patterns. Toward this goal, in this paper, we propose a novel periodic pattern called the periodic association sequence (PAS), which is represented by a sequence of periodic (or cyclic) association rules rather than typical frequent itemsets. Hence, its primary advantage is the capability of describing repeating consequences in both temporal and multivariate dimensions. Then, we present an efficient algorithm and mechanism for detecting PASs from a large-scale multivariate time-series database. Last, using two real-world smartphone log data sets, we conduct extensive experiments to demonstrate the benefits of the PAS. The evaluation results confirm that PASs are more coherent to human behavior and achieve higher confidence than state-of-the-art periodic patterns.

## Algorithm
| Approach                 | Method          | Paper                |
| :----------------------- | :-------------- | :------------------- |
| Assocation-only approach | MobileMiner     | [Srinivasan et al.](https://ai2-s2-pdfs.s3.amazonaws.com/b379/bc081918226f5f92f1919bceebbb76a1d027.pdf)|
| Sequence-only approach   | SMCA            | [Huang et al.](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1423978) |
| **Our apprach**          | **PAS_MINING**  |                      |
- Language: Python 2.7
- We incorporated the competitors into our single implementation. The implementation of the competitors guarantees the correct output, but we did not fully implement the efficiency optimization techniques for the competitors because efficiency comparison with the competitors is beyond the scope of the paper.

## Data Sets
| Name            | Link                                         | Description                                    |
| :-------------- | :------------------------------------------- |:-----------------------------------------------|
| Device Analyzer | [Link](https://deviceanalyzer.cl.cam.ac.uk/) | 541 users' smartphone usage data   |
| KAIST           | [Link](http://dmserver6.kaist.ac.kr/PAS/kaist_dataset.tar.gz) | 27 users' smartphone usage data |

- Three days of a sample user in the Device Analyzer data set
![Data example](http://dmserver6.kaist.ac.kr/PAS/img/data_example.png "Data example")

- Periodicity visualization using our proposed **PERIODICITY SPECTRUM**
<img height="200" src="http://dmserver6.kaist.ac.kr/PAS/img/periodicity_distribution.png"></img>

- Variables used in the Device Analyzer data set

| Variable          | Description          | Type    | # Instances   |
| :---------------: | :------------------: | :-----: |:-------------:|
| appcat            | Application category | nominal |	18,676,555   |
| celllocation\_lac | Location name (LAC)  | nominal |	25,104,298   |
| charge\_state     | Charge status        | nominal |	5,293,941    |
| headset           | Headset on/off       | binary  |	1,004,573    |
| ringermode        | Ringtone mode        | nominal |	4,728,265    |
| wifi\_conn\_state | Wifi connectivity    | binary  |	2,350,560    |

- Variables used in the KAIST data set

| Variable           | Description              | Type    | # Instances |
| :----------------: | :----------------------: | :-----: |:-----------:|
| appcat             | Application category     | nominal |	191,537     |
| location\_district | Location name (district) | nominal |	56,719      |
| location\_bssid    | Wifi BSSID               | nominal |	40,788      |
| location\_ssid     | Wifi SSID                | nominal |	35,217      |
| charge\_state      | Charge status            | nominal |	81,110      |
| headset            | Headset on/off           | binary  |	73,106      |
| ringermode         | Ringtone mode            | nominal |	73,106      |
| wifi\_conn\_state  | Wifi connectivity        | binary  |	41,007      |

## How to Run
1. Install: python2.7, pandas, numpy
2. Set the preprocessing parameters in pas_mining.py
  - dir_path: the path of input data sets
  - output_dir: the path of preprocessed data sets
  - rules_dir: the path of final results (patterns)
  - granularity_min: time span size (in minutes)
  - MIN_DATE_LENGTH: the minimum number of days which valid users should have
3. Set the configuration paramters in pas_mining.py
  - min_sup: the minimum support
  - min_conf: the minimum confidence
  - max_gap: the maximum gap
4. Run '**python pas_mining.py**'
  
## Experiment
All results are saved in the 'output' directory. Please make sure the 'output' directory existing and accessible.

1. Comparision with Association-Only
  - Run '**python experiment_association_only.py**'
  - <img height="360" src="http://dmserver6.kaist.ac.kr/PAS/img/compared_association.png"></img>
2. Comparision with Sequence-Only <br/>
  - Run '**python experiment_sequence_only.py**'
  - <img height="210" src="http://dmserver6.kaist.ac.kr/PAS/img/compared_sequence.png"></img>
3. Performance and Scalability Result
  - Run '**python pas_mining_performance.py & python experiment_performance.py**'
  - <img height="210" src="http://dmserver6.kaist.ac.kr/PAS/img/performance.png"></img>
