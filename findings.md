<h2>Statistics of Datasets</h2>

| Dataset | #Users | #Items | #Interactions | #Density |
| :---: | :---: | :---: | :---: | :---: |
| ML-1M | 6,038 | 3,492 | 575,281 | 2.728% |
| DouBan | 2,831 | 36,821 | 805,611 | 0.772% |
| Epinions | 75,887 | 75,881 | 508,837 | 0.009% |

<h2>Evaluation Protocol</h2>

We split the datasets into three parts (training set, validation set, and test set) in a ratio of 7:1:2. The poisoning data only affects the model training process; therefore, we exclusively utilize the training set data as the known data. Four metrics are used for measuring attack efficiency, i.e., Hit Ratio@50, Precision@50, Recall@50 ,and NDCG@50. Each experiment in this section is conducted 10 times, and then we report the average results. The victim Model is choosen as LightGCN.

<h2>Findings</h2>

We begin by comparing existing poisoning attack methods on three distinct datasets. The results are presented in the following content. In these tables, values shown in bold represent the best performing indices. Upon examining the tables, we can derive the following observations and conclusions: 

1. **Effectiveness of Attack Methods**: In environments free from attacks, the probability of targeted items appearing in user recommendations is generally low. Traditional attack methods, such as Random-Attack and Bandwagon-Attack, often do not match the effectiveness of more sophisticated intelligent attack methods. This observation highlights the evolving nature of attack strategies and the need for robust countermeasures.

2. **Diverse Impact of Poisoning Attacks**: Diverse poisoning attack methods exhibit varying degrees of efficacy across different evaluation metrics. While CLeaR exhibits its formidable threats in the Hit Ratio metric across all cases, it does not consistently lead in other measures. The Hit Ratio indicates the extent to which the target item is visible to users, whereas Precision, Recall, and NDCG offer more nuanced insights into the attack's impact. Depending on their objectives, attackers may opt for specific methods that align with their desired outcomes, whether it be widespread visibility or more targeted influence.

3. **Variability Across Datasets**: The performance of poisoning attack methods varies across different datasets. This variability underscores the difficulty in predicting which poisoning attack method might be deployed in real-world scenarios, as the choice heavily depends on specific circumstances.

Our findings reveal a complex landscape of poisoning attack methods, each with unique strengths and weaknesses across different datasets and evaluation criteria. This complexity necessitates a multi-faceted approach to understanding and mitigating such attacks in recommendation systems.


<h2>Experimental Results on ML-1M</h2>

**Target Items**: '371', '3637', '3053', '3334', '158' <br><br>

| **Method**          | **Attack Ratio** | **HitRate@50** | **Precision@50** | **Recall@50** | **NDCG@50**   |
|:-----------------:|:------------:|:---------:|:-----------:|:--------:|:--------:|
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

**Target Items**: '31232', '35591', '31660', '26924', '28069' <br><br>

| **Method**          | **Attack Ratio** | **HitRate@50** | **Precision@50** | **Recall@50** | **NDCG@50**   |
|:-----------------:|:------------:|:---------:|:-----------:|:--------:|:--------:|
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


<h2>Experimental Results on Epinions</h2>

**Target Items**: '38063', '41186', '71296', '35566', '68017' <br><br>

| **Method**          | **Attack Ratio** | **HitRate@50** | **Precision@50** | **Recall@50** | **NDCG@50**   |
|:-----------------:|:------------:|:---------:|:-----------:|:--------:|:--------:|
| RandomAttack   | 0.01       | 0.0074  | 0.0035    | 0.0346 | 0.0226|
| BandwagonAttack| 0.01       | 0.0063  | 0.0029    | 0.0291 | 0.0168|
| AUSH           | 0.01       | 0.0022  | 0.0010    | 0.0107 | 0.0102|
| PoisonRec      | 0.01       | 0.0021  | 0.0011    | 0.0106 | 0.0102|
| GOAT           | 0.01       | 0.0034  | 0.0004    | 0.0041 | 0.0021|
| A_ra           | 0.01       | 0.0123  | **0.0056**    | **0.0558** | **0.0311**|
| GTA            | 0.01       | 0.0044  | 0.0020    | 0.0203 | 0.0126|
| CLeaR          | 0.01       | **0.0131**  | 0.0055    | 0.0552 | 0.0295|
| DLAttack       | 0.01       | 0.0021  | 0.0010    | 0.0104 | 0.0101|
| FedRecAttack   | 0.01       | 0.0118  | 0.0054    | 0.0536 | 0.0299|




