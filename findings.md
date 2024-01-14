This content is under construction.

<h2>Statistics of Datasets</h2>

| Dataset | #Users | #Items | #Interactions | #Density |
| --- | --- | --- | --- | --- |
| ML-1M | 6,038 | 3,492 | 575,281 | 2.728% |
| DouBan | 2,831 | 36,821 | 805,611 | 0.772% |
| Epinions | 75,887 | 75,881 | 508,837 | 0.009% |

<h2>Evaluation Protocol</h2>

We split the datasets into three parts (training set, validation set, and test set) in a ratio of 7:1:2. The poisoning data only affects the model training process; therefore, we exclusively utilize the training set data as the known data. Four metrics are used for measuring attack efficiency, i.e., Hit Ratio@50, Precision@50, Recall@50 ,and NDCG@50. Each experiment in this section is conducted 10 times, and then we report the average results.

<h2>Experimental Results on ML-1M</h2>
Victim Model: LightGCN <br>
Target Item: '371', '3637', '3053', '3334', '158' <br><br>

| **Method**          | **Attack Ratio** | **HitRate@50** | **Precision@50** | **Recall@50** | **NDCG@50**   |
|-----------------|------------|---------|-----------|--------|--------|
| RandomAttack    | 0.01       | 0.0106  | 0.0019    | 0.0185 | 0.0112 |
| BandwagonAttack | 0.01       | 0.0133  | 0.0021    | 0.0206 | 0.0131 |
| AUSH            | 0.01       | 0.0122  | 0.0022    | 0.0217 | 0.0169 |
| PoisonRec       | 0.01       | 0.0130  | **0.0023**    | 0.0225 | **0.0177** |
| GOAT            | 0.01       | 0.0094  | 0.0017    | 0.0173 | 0.0104 |
| A_ra            | 0.01       | 0.0133  | 0.0019    | 0.0192 | 0.0097 |
| GTA             | 0.01       | 0.0145  | **0.0023**    | **0.0227** | 0.0098 |
| CLeaR           | 0.01       | **0.0146**  | 0.0019    | 0.0188 | 0.0092 |
| DLAttack        | 0.01       | 0.0096  | 0.0018    | 0.0183 | 0.0142 |
| FedRecAttack    | 0.01       | 0.0136  | 0.0021    | 0.0207 | 0.0113 |



<h2>Experimental Results on DouBan</h2>
Victim Model: LightGCN <br>
Target Item: '31232', '35591', '31660', '26924', '28069' <br><br>

| **Method**          | **Attack Ratio** | **HitRate@50** | **Precision@50** | **Recall@50** | **NDCG@50**   |
|-----------------|------------|---------|-----------|--------|--------|
| RandomAttack    | 0.01       | 0.0010  | 0.0002    | 0.0021 | 0.0008 |
| BandwagonAttack | 0.01       | 0.0020  | 0.0010    | 0.0096 | 0.0088 |
| AUSH            | 0.01       | 0.0022  | 0.0010    | **0.0104** | **0.0102** |
| PoisonRec       | 0.01       | 0.0022  | **0.0011**    | 0.0103 | 0.0101 |
| GOAT            | 0.01       | 0.0016  | 0.0007    | 0.0075 | 0.0048 |
| A_ra            | 0.01       | 0.0012  | 0.0002    | 0.0024 | 0.0013 |
| GTA             | 0.01       | 0.0020  | 0.0003    | 0.0033 | 0.0012 |
| CLeaR           | 0.01       | **0.0024**  | 0.0005    | 0.0047 | 0.0020 |
| DLAttack        | 0.01       | 0.0021  | 0.0010    | 0.0099 | 0.0097 |
| FedRecAttack    | 0.01       | 0.0012  | 0.0003    | 0.0027 | 0.0013 |



