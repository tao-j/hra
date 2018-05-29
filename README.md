# Rank Aggregation
main.py will load the configuration and save running result in when done. One can setup comment/uncomment for different configurations in main.py and invoke the script sequentially. They can be run simultaneously, the saving process will have a lock to prevent each process writing over existing data.


## naming convention

`btl`: Simple BT-model\
`gbtl`: Generalized BT-model, consider confidence of each judge

`random`: random initializations\
`spectral`: Using rank centrality to provide initialization\

`do`: just output after initialization\
`mle`: use MLE, optimized by SGD to find result\

`j`: number of judges\
`i`: number of items\
`p`: number of comparisons provided by a single judge

## algorithm

$s$ is the actual utility of item, in some literature it is referred as $w$ or $\pi$. Usually, we refer $w$ as  $w=e^s$ and don't use $\pi$.

$c_{ji,k}$ is the count of item $i$ is preferred over item $j$ for judge $k$

`btl`: $\sum{c_{ji} * \log( {\exp{(s_j - s_i)} + 1}} )$\
`gbtl`: $\sum{c_{ji,k} * \log({\exp{(s_j - s_i)/\beta_k} + 1}})$

### initialization
The spectral method described in "Rank Centrality" is used to provide a near estimate of the true utility.

+ Count the number of times item $j$ is preferred over $i$ and form a matrix.
+ Figure out the transition matrix, note there will be a self loop for each node.
+ Compute the stationary distribution for the matrix treat it as $w$.
+ Take log then get $s$.


+ $s$ is shifted to have minimum value to be 0. $\mathbf{s} = \mathbf{s} - \min{s_i}$.
+ $s$ is then nomalized to be summed to 1. $\mathbf{s}  = \mathbf{s}  / \sum{s_i}$.


+ For `gbtl`, after calculate $s$ for all judges, we assume $\beta_1 = 1$, the $\beta_k$ for each judge will be $\beta_k = 1 / \#items  *\sum{s_{1, i}/s_{k, i} }$

### optimization

+ Calculate the gradient using likelihood function mentioned above. Update $s$ and $\beta$ simultaneously. 
+ $s$ is shifted to have minimum value to be 0. $\mathbf{s} = \mathbf{s} - \min{s_i}$.
+ $\beta$ is scaled to prevent the change of likelihood for next step $\beta = \beta / \sum{s_i}$.
+ $s$ is then nomalized/scaled to be summed to 1. $\mathbf{s} = \mathbf{s} / \sum{s_i}$.


## code base 

Currently, the source code reflect three generations of changes.

+ explicit loop for single element optimization, single pair. Code was removed because of inefficiency.
+ explicit loop for single element optimization, counting pair. `opt_sparse = True`.
+ implicit loop using matrix computation, counting pair. `opt_spares = False`.

The third method can finish 800 iterations of tranining for 100 items in one second. However the second method would cost half an hour.

However, data is generated pair by pair, for some runs including more judges and more comparison for each individual judge, it will take serveral seconds to generate. This should be improved later.
The model will be load into GPU once for each configuration, however, for each configuration, only data matrix is changed. It is possible to make the model persistent in memory to prevent this overhead.

## experiments

After the third revision of the code base is done. It is able to do more large scale computation. Each setup is runned over 12 predefined random seeds. 

The final result of ranking is compared and provided using Spearman Dist. More comparison metric will be impletemented in the future.
The accuracy of estimated $s$ is not compared.

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
