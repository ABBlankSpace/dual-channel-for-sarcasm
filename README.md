# Overview
Here is the source code for our NAACL 2022 paper A Dual-Channel Framework for Sarcasm Recognition by Detecting Sentiment Conflict.

## Run the scripts

Please modify the parameters in `train.sh` and run `sh train.sh`. 

## Experiments

### \* Datasets

We evaluate our model on three benchmark datasets, IAC-V1, IAC-V2, Twitter. The split datasets are in `/data`. 

The official websites for datasets:
- IAC-V1: https://nlds.soe.ucsc.edu/sarcasm1
- IAC-V2: https://nlds.soe.ucsc.edu/sarcasm2
- Twitter: https://github.com/Cyvhee/SemEval2018-Task3

### \* Main results

<table>
    <tr>
        <td rowspan="2">Model</td>
        <td colspan="2">IAC-V1</td>
        <td colspan="2">IAC-V2</td>
        <td colspan="2">Tweets</td>
    </tr>
    <tr>
        <td>F1</td>
        <td>Acc.</td>
        <td>F1</td>
        <td>Acc.</td>
        <td>F1</td>
        <td>Acc.</td>
    </tr>
    <tr>
        <td>UCDCC</td>
        <td>58.5</td>
        <td>58.5</td>
        <td>67.0</td>
        <td>67.0</td>
        <td>72.4</td>
        <td>79.7</td>
    </tr>
    <tr>
        <td>THU-NGN</td>
        <td>64.2</td>
        <td>64.3</td>
        <td>73.3</td>
        <td>73.3</td>
        <td>70.5</td>
        <td>73.5</td>
    </tr>
    <tr>
        <td>Bi-LSTM</td>
        <td>64.6</td>
        <td>64.6</td>
        <td>79.7</td>
        <td>79.7</td>
        <td>71.7</td>
        <td>73.0</td>
    </tr>
    <tr>
        <td>At-LSTM</td>
        <td>65.3</td>
        <td>65.5</td>
        <td>76.1</td>
        <td>76.2</td>
        <td>70.0</td>
        <td>70.2</td>
    </tr>
    <tr>
        <td>CNN-LSTM-DNN</td>
        <td>60.9</td>
        <td>61.1</td>
        <td>75.2</td>
        <td>75.3</td>
        <td>71.9</td>
        <td>72.3</td>
    </tr>
    <tr>
        <td>MIARN</td>
        <td>64.9</td>
        <td>65.2</td>
        <td>75.2</td>
        <td>75.3</td>
        <td>68.8</td>
        <td>70.2</td>
    </tr>
    <tr>
        <td>ADGCN</td>
        <td>64.3</td>
        <td>64.3</td>
        <td>80.9</td>
        <td>80.9</td>
        <td>72.8</td>
        <td>73.6</td>
    </tr>
    <tr>
        <td>DC-Net (Ours)</td>
        <td>66.4</td>
        <td>66.5</td>
        <td>82.1</td>
        <td>82.1</td>
        <td>76.3</td>
        <td>76.7</td>
    </tr>
</table>


## Citation
```
@article{DBLP:journals/corr/abs-2109-03587,
  author    = {Yiyi Liu and
               Yequan Wang and
               Aixin Sun and
               Zheng Zhang and
               Jiafeng Guo and
               Xuying Meng},
  title     = {A Dual-Channel Framework for Sarcasm Recognition by Detecting Sentiment
               Conflict},
  journal   = {CoRR},
  volume    = {abs/2109.03587},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.03587}
}
```