---
title: Rank Aggregation
---

`main.py` will load the configuration and save running result in when done. One can setup comment/uncomment for different configurations in main.py and invoke the script sequentially. They can be run simultaneously, the saving process will have a lock to prevent each process writing over existing data.

# Algorithm

Assume we have a series of items and each of them have a score conceptually. The score is denoted as $\mathbf{s}$ and sometimes called the utility of item, in some literature it is referred as $\mathbf{w}$ or $\pi$. In this document, we refer $w$ as $w=e^s$ and don't use $\pi$.

## Models - Methods for calculating likelihood

Let $c_{ji,k}$ be the numbers of times $i$ is preferred over item $j$ reported by judge $k$.

### btl
`btl`: Simple BT-model

$$\sum{c_{ji} * \log( {\exp{(s_j - s_i)} + 1}} )$$

### gbtl
`gbtl`: Generalized BT-model, consider confidence of each judge

$$\sum{c_{ji,k} * \log({\exp{((s_j - s_i)/\beta_k)} + 1}})$$

In order to cast the restriction that $\beta > 0$, in actually implementation $\beta = \epsilon^2$ is used, the formula becomes $\sum{c_{ji,k} * \log({\exp{((s_j - s_i)/\epsilon^2)} + 1}})$

Note that the likehood can be affected by the number of pairs, so it is normalized by dividing the result by $\sum{c_{ji}}$.

### gbtl-negative

`gbtl-negative`: The denominator part in `gbtl` can be negative.
Ideally, this algorithm can handle those adversarial judges which know that $s_i > s_j$ but cast a vote that $s_i < s_j$.

$$\sum{c_{ji,k} * \log({\exp{((s_j - s_i)/\beta_k)} + 1}})$$

However, because if $\beta$ is near zero, it will be a perfect judge, when noisy data is seen, in order to compensate the likelihood, the $\beta$ tends to be larger and larger, and have no possible way to become an adversarial so that to have a negative value of $\beta$.


### gbtl-inverse

`gbtl-inverse`: Let $inv_k = 1/\beta_k$ the denominator part in `gbtl-negative` is retreated as a multiplier for easy optimization. 

This is used to compensate the effect of previous method, so the good judge will have a very large $inv$, while bad judge can flip sign to become adversarial. However, this behavior is not observed in the experiments.

$$\sum{c_{ji,k} * \log({\exp{((s_j - s_i)*inv_k)} + 1}})$$

## Initialization
The spectral method described in "Rank Centrality" is used to provide a near estimate of the true utility.

+ Count the number of times item $j$ is preferred over $i$ and form a matrix.
+ Figure out the transition matrix, note there will be a self loop for each node.
+ Compute the stationary distribution for the matrix treat it as $w$.
+ Take log then get $s$.

For unified optimization answer value are normalized.

+ $s$ is shifted to have minimum value to be 0. $\mathbf{s} = \mathbf{s} - \min{s_i}$.
+ $s$ is then nomalized to be summed to 1. $\mathbf{s}  = \mathbf{s}  / \sum{s_i}$.


+ For `gbtl-*`, after calculate $s$ for all judges, we assume $\beta_1 = 1$, the $\beta_k$ for each judge will be $\beta_k = 1 / \#items  *\sum{s_{1, i}/s_{k, i} }$. Because it is hard to analytically make solution for the $\beta$, this approximation is used.

## Optimization

+ Calculate the gradient using likelihood function mentioned above. Update $s$ and $\beta$ simultaneously. It is also possible to do alternating update.

Because in the likelihood function, only difference between $s$ matters, so there will be infinate number of solutions if $s$ is not restricted. Without loss of generality we put $\min(s) = 0$. In `gbtl-*` model, consider we have fixed `s`, but the 
+ $s$ is shifted to have minimum value to be 0. $\mathbf{s} = \mathbf{s} - \min{s_i}$.
+ $\beta$ is scaled to prevent the change of likelihood for next step $\beta = \beta / \sum{s_i}$.
+ $s$ is then nomalized/scaled to be summed to 1. $\mathbf{s} = \mathbf{s} / \sum{s_i}$.


# Code Implementation

Currently, the source code reflect three generations of changes.

+ explicit loop for single element optimization, single pair. Code was removed because of inefficiency.
+ explicit loop for single element optimization, counting pair. `opt_sparse = True`.
+ implicit loop using matrix computation, counting pair. `opt_spares = False`.

The third method can finish 800 iterations of tranining for 100 items in one second. However the second method would cost half an hour.

However, data is generated pair by pair, for some runs including more judges and more comparison for each individual judge, it will take serveral seconds to generate. This should be improved later.
The model will be load into GPU once for each configuration, however, for each configuration, only data matrix is changed. It is possible to make the model persistent in memory to prevent this overhead.

Planned: share data generation result to prevent regenerate.

## experiments

After the third revision of the code base is done. It is able to do more large scale computation. Each setup is runned over 12 predefined random seeds. 

The final result of ranking is compared and provided using Spearman Dist. More comparison metric will be impletemented in the future.
The accuracy of estimated $s$ is not compared.


## naming convention

`random`: random initializations\
`spectral`: Using rank centrality to provide initialization\

`do`: just output after initialization\
`mle`: use MLE, optimized by SGD to find result\

`j`: number of judges\
`i`: number of items\
`p`: number of comparisons provided by a single judge

Subscripts i,j,k will be used for indexing first item in pair, second item in pair and specific judge repectively.

### single judge
compare `btl` and `gbtl` with different $\beta$, it confirms that when $\beta$ ranging from 0 to 1, the ranking accuracy for the two algorithm will drop. And the recovery of the true $s$ in `gbtl` seems to be better. 

### several judges
we want to know what exactly the deviation from good judge and bad judge that our model performs well.

To make things simple, we assume half of the judges are good, half of them are bad.
It seems that $1.0, 0.1$ or $1.0, 0.05$  works best for our algorithm.

#### 0.5
|data|algo|acc|
|----|----|----|
|ma-b1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5-j8-i10-p800|btl-spectral-mle|0.8090909091|
|ma-b1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5-j8-i10-p800|gbtl-spectral_all-mle|0.8|
|ma-b1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5-j8-i100-p8000|btl-spectral-mle|0.1201350135|
|ma-b1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5-j8-i100-p8000|gbtl-spectral_all-mle|0.1292939294|
|ma-b1.0,0.5,1.0,0.5-j4-i10-p800|btl-spectral-mle|0.7686868687|
|ma-b1.0,0.5,1.0,0.5-j4-i10-p800|gbtl-spectral_all-mle|0.7727272727|
|ma-b1.0,0.5-j2-i10-p800|btl-spectral-mle|0.6838383838|
|ma-b1.0,0.5-j2-i10-p800|gbtl-spectral_all-mle|0.6878787879|

#### 0.1
|data|algo|acc|
|----|----|----|
|ma-b1.0,0.1,1.0,0.1-j4-i10-p800|btl-spectral-mle|0.9353535354|
|ma-b1.0,0.1,1.0,0.1-j4-i10-p800|gbtl-spectral_all-mle|0.9525252525|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p8000|btl-spectral-mle|0.2715981598|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p8000|gbtl-spectral_all-mle|0.289009901|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p80000|btl-spectral-mle|0.6671327133|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p80000|gbtl-spectral_all-mle|0.6681918192|
|ma-b1.0,0.1-j2-i10-p800|btl-spectral-mle|0.9404040404|
|ma-b1.0,0.1-j2-i10-p800|gbtl-spectral_all-mle|0.9393939394|

#### 0.05
|data|algo|acc|
|----|----|----|
|ma-b1.0,0.05,1.0,0.05,1.0,0.05,1.0,0.05-j8-i10-p800|btl-spectral-mle|0.9737373737|
|ma-b1.0,0.05,1.0,0.05,1.0,0.05,1.0,0.05-j8-i10-p800|gbtl-spectral_all-mle|0.9878787879|
|ma-b1.0,0.05,1.0,0.05-j4-i10-p800|btl-spectral-mle|0.9666666667|
|ma-b1.0,0.05,1.0,0.05-j4-i10-p800|gbtl-spectral_all-mle|0.9767676768|
|ma-b1.0,0.05-j2-i10-p800|btl-spectral-mle|0.9686868687|
|ma-b1.0,0.05-j2-i10-p800|gbtl-spectral_all-mle|0.9757575758|

#### other worse result (truncated)
|data|algo|acc|
|----|----|----|
|ma-b1.0,0.001,0,0.001-j4-i100-p8000|btl-spectral-mle|0.9955955596|
|ma-b1.0,0.001,0,0.001-j4-i100-p8000|gbtl-spectral_all-mle|0.5013961396|
|ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i10-p200|btl-spectral-mle|0.9929292929|
|ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i10-p200|gbtl-spectral_all-mle|0.4159779614|
|ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i10-p800|btl-spectral-mle|1|
|ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i10-p800|gbtl-spectral_all-mle|0.6151515152|
|ma-b1.0,0.001,1.0,0.001,1.0,0.001,1.0,0.001-j8-i100-p8000|btl-spectral-mle|0.9943934393|

|data|algo|acc|
|----|----|----|
|ma-b1.0,0.005,1.0,0.005,1.0,0.005,1.0,0.005-j8-i10-p800|btl-spectral-mle|0.996969697|
|ma-b1.0,0.005,1.0,0.005,1.0,0.005,1.0,0.005-j8-i10-p800|gbtl-spectral_all-mle|0.7444444444|
|ma-b1.0,0.005,1.0,0.005-j4-i10-p800|btl-spectral-mle|0.9919191919|
|ma-b1.0,0.005,1.0,0.005-j4-i10-p800|gbtl-spectral_all-mle|0.6585858586|
|ma-b1.0,0.005-j2-i10-p800|btl-spectral-mle|0.9888888889|
|ma-b1.0,0.005-j2-i10-p800|gbtl-spectral_all-mle|0.4981818182|


#### effect of number of judges

For different size of repeated comparisons

|data|algo|acc|
|----|----|----|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i10-p800|btl-spectral-mle|0.9575757576|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i10-p800|gbtl-spectral_all-mle|0.9696969697|
|ma-b1.0,0.1,1.0,0.1-j4-i10-p800|btl-spectral-mle|0.9353535354|
|ma-b1.0,0.1,1.0,0.1-j4-i10-p800|gbtl-spectral_all-mle|0.9525252525|
|ma-b1.0,0.1-j2-i10-p800|btl-spectral-mle|0.9404040404|
|ma-b1.0,0.1-j2-i10-p800|gbtl-spectral_all-mle|0.9393939394|

|data|algo|acc|
|----|----|----|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p8000|btl-spectral-mle|0.4824192419|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p8000|gbtl-spectral_all-mle|0.4846394639|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p8000|btl-spectral-mle|0.3701830183|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p8000|gbtl-spectral_all-mle|0.3776047605|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p8000|btl-spectral-mle|0.2715981598|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p8000|gbtl-spectral_all-mle|0.289009901|

|data|algo|acc|
|----|----|----|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p80000|btl-spectral-mle|0.8575687569|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j16-i100-p80000|gbtl-spectral_all-mle|0.8588678868|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p80000|btl-spectral-mle|0.7859305931|
|ma-b1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1-j8-i100-p80000|gbtl-spectral_all-mle|0.7851465147|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p80000|btl-spectral-mle|0.6671327133|
|ma-b1.0,0.1,1.0,0.1-j4-i100-p80000|gbtl-spectral_all-mle|0.6681918192|

#### if the number of total comparisons stays the same


### sample beta from beta dist.

This is trying to replicate result from "Pair wise comparison for crowd sourcing"

For almost noisy judge data, replicate the beta(0.5, 0.5) setting in paper.

|data|algo|acc|
|----|----|----|
|be-b1,1-j8-i100-p800|btl-spectral-mle|0.08789578958|
|be-b1,1-j8-i100-p800|gbtl-spectral_all-mle|0.08652965297|
|be-b1,1-j8-i100-p8000|btl-spectral-mle|0.3267266727|
|be-b1,1-j8-i100-p8000|gbtl-spectral_all-mle|0.3415711571|
|be-b1,1-j8-i100-p80000|btl-spectral-mle|0.6691479148|
|be-b1,1-j8-i100-p80000|gbtl-spectral_all-mle|0.6653045305|

Most of the judges are good judge, test the effect of increasing number of comparisons.

|data|algo|acc|
|----|----|----|
|be-b1,10-j8-i100-p800|btl-spectral-mle|0.5451905191|
|be-b1,10-j8-i100-p800|gbtl-spectral_all-mle|0.464360436|
|be-b1,10-j8-i100-p8000|btl-spectral-mle|0.8952985299|
|be-b1,10-j8-i100-p8000|gbtl-spectral_all-mle|0.8870107011|
|be-b1,10-j8-i100-p80000|btl-spectral-mle|0.9789818982|
|be-b1,10-j8-i100-p80000|gbtl-spectral_all-mle|0.9818841884|
|be-b10,100-j8-i100-p800|btl-spectral-mle|0.2448734873|
|be-b10,100-j8-i100-p800|gbtl-spectral_all-mle|0.2081188119|
|be-b10,100-j8-i100-p8000|btl-spectral-mle|0.6454565457|
|be-b10,100-j8-i100-p8000|gbtl-spectral_all-mle|0.6555425543|
|be-b10,100-j8-i100-p80000|btl-spectral-mle|0.9328562856|
|be-b10,100-j8-i100-p80000|gbtl-spectral_all-mle|0.933360336|

### gbtl-neg

```
============ progress 0.075 ETA 279.58250617980957 Elpased 22.66885280609131
ground truth s [0.         0.00020163 0.00041065 0.00062733 0.00085195 0.00108479
 0.00132616 0.00157638 0.00183576 0.00210464 0.00238337 0.00267232
 0.00297184 0.00328235 0.00360422 0.00393789 0.00428378 0.00464234
 0.00501404 0.00539935 0.00579878 0.00621284 0.00664206 0.00708702
 0.00754827 0.00802642 0.00852208 0.0090359  0.00956855 0.01012071
 0.01069309 0.01128644 0.01190153 0.01253915 0.01320013 0.01388533
 0.01459562 0.01533194 0.01609522 0.01688647 0.01770671 0.01855699
 0.01943842 0.02035214 0.02129933 0.02228122 0.02329908 0.02435423
 0.02544803 0.0265819  0.0277573  0.02897577 0.03023887 0.03154824
 0.03290558 0.03431264 0.03577125 0.03728329 0.03885072 0.04047557
 0.04215994 0.04390601 0.04571605 0.0475924 ]
ground truth beta [ 0.01  0.01  0.01 -0.01]
initial ranking result [ 3  7  6  5  8 10  0 11  4 12  2 13  1 19 14 16  9 15 18 17 29 23 21 22
 25 24 20 28 32 27 26 31 33 34 36 37 30 35 38 41 45 39 40 43 44 42 47 49
 46 52 48 50 53 57 51 55 56 54 60 58 61 59 63 62]
initial: s, beta [-5.2720213 -5.1716332 -5.2129464 -5.444453  -5.2164335 -5.368752
 -5.3729267 -5.3734903 -5.3445864 -5.135718  -5.335087  -5.2179203
 -5.214886  -5.1861053 -5.146576  -5.1004915 -5.1431556 -4.987467
 -5.0673037 -5.15859   -4.80821   -4.872526  -4.863924  -4.908108
 -4.8143506 -4.857876  -4.7040515 -4.7425213 -4.761551  -4.9777923
 -4.4764986 -4.692269  -4.744742  -4.682644  -4.6659007 -4.46313
 -4.625089  -4.616323  -4.4551554 -4.363563  -4.328222  -4.4227347
 -4.272601  -4.327117  -4.2889585 -4.388178  -4.1075854 -4.176772
 -4.0739956 -4.1465154 -4.040415  -3.9391983 -4.094847  -3.9865823
 -3.8557765 -3.9005246 -3.891471  -3.9393566 -3.7179568 -3.6568007
 -3.7950552 -3.6620479 -3.5163724 -3.6215727] [1.        1.0002658 0.9995459 1.1390154]


{'res_s': array([0.00124053, 0.00040423, 0.00151192, 0.00128391, 0.00158678,
       0.00082832, 0.        , 0.00318954, 0.00109345, 0.00241894,
       0.00208388, 0.00181347, 0.00263239, 0.0033853 , 0.00300745,
       0.00432503, 0.00486559, 0.00475302, 0.00503233, 0.00445427,
       0.00703741, 0.00695946, 0.00651139, 0.00769732, 0.00819746,
       0.00874015, 0.00946085, 0.00893357, 0.01014537, 0.00983707,
       0.01191817, 0.01097003, 0.01102478, 0.01298092, 0.01320732,
       0.01593317, 0.01443559, 0.01500799, 0.01573041, 0.01768097,
       0.01755599, 0.01891693, 0.01924198, 0.02010392, 0.01983309,
       0.02065322, 0.02342728, 0.02284312, 0.02449871, 0.0255978 ,
       0.02778998, 0.0287823 , 0.02884273, 0.03269271, 0.03338232,
       0.03350034, 0.03440319, 0.03572588, 0.03953067, 0.04090979,
       0.04199352, 0.04351607, 0.04731778, 0.04662086], dtype=float32), 'res_beta': array([0.01293905, 0.01244986, 0.01229717, 0.28041995], dtype=float32)}
ne-b0.01-j4-i64-p12800+gbtlneg-spectral_all-mle 0.9949633699633698
```

### ground truth compared to spectral

```
481	be-b1,1-j8-i100-p800	btl-spectral-mle	0.235675
489	be-b1,1-j8-i100-p800	gbtlneg-disturb-mle	0.252450

491	be-b1,1-j8-i100-p8000	btl-spectral-mle	0.578940
499	be-b1,1-j8-i100-p8000	gbtlneg-disturb-mle	0.731426

501	be-b1,1-j8-i100-p80000	btl-spectral-mle	0.875639
509	be-b1,1-j8-i100-p80000	gbtlneg-disturb-mle	0.958753

*511	be-b1,10-j8-i100-p800	btl-spectral-mle	0.799607
*519	be-b1,10-j8-i100-p800	gbtlneg-disturb-mle	0.651792

*521	be-b1,10-j8-i100-p8000	btl-spectral-mle	0.966131
*529	be-b1,10-j8-i100-p8000	gbtlneg-disturb-mle	0.947896

531	be-b1,10-j8-i100-p80000	btl-spectral-mle	0.994547
539	be-b1,10-j8-i100-p80000	gbtlneg-disturb-mle	0.992609

*311	be-b10,100-j4-i100-p800	btl-spectral-mle	0.313435
*319	be-b10,100-j4-i100-p800	gbtlneg-disturb-mle	0.294446

321	be-b10,100-j4-i15-p200	btl-spectral-mle	0.916964
329	be-b10,100-j4-i15-p200	gbtlneg-disturb-mle	0.924702

*541	be-b10,100-j8-i100-p800	btl-spectral-mle	0.460158
*549	be-b10,100-j8-i100-p800	gbtlneg-disturb-mle	0.421968

551	be-b10,100-j8-i100-p8000	btl-spectral-mle	0.842468
559	be-b10,100-j8-i100-p8000	gbtlneg-disturb-mle	0.850073

561	be-b10,100-j8-i100-p80000	btl-spectral-mle	0.973359
569	be-b10,100-j8-i100-p80000	gbtlneg-disturb-mle	0.974697

331	be-b10,100-j8-i15-p400	btl-spectral-mle	0.976488
339	be-b10,100-j8-i15-p400	gbtlneg-disturb-mle	0.977976

341	be-b2,10-j4-i15-p800	btl-spectral-mle	0.945833
349	be-b2,10-j4-i15-p800	gbtlneg-disturb-mle	0.965584

351	be-b3,20-j4-i15-p800	btl-spectral-mle	0.971131
359	be-b3,20-j4-i15-p800	gbtlneg-disturb-mle	0.975893

361	be-b5,100-j4-i15-p800	btl-spectral-mle	0.989286
369	be-b5,100-j4-i15-p800	gbtlneg-disturb-mle	0.991369
```