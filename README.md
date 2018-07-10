---
title: Rank Aggregation from Pairwise Comparision
---

# Algorithm

Given a set of items and each of them have a score conceptually, which may or may not exist in reality. The score is denoted as $\mathbf{s}$ and sometimes called the utility of item. In some literature it is referred as $\mathbf{w}$ or $\pi$. In this document, we refer $w$ as $w=e^s$ and assume $\sum{\pi_i} = 1$, which is a normalized version of $w$.

## Models - Methods for calculating likelihood

Let $c_{ij,k}$ be the numbers of times $i$ is preferred over item $j$ reported by judge $k$. Note that $i \neq j$.

In following methods, each kind of likelihood function corredponds to a model with various intentions. 

### btl
`btl`: Simple BT-model

We assume each user have a random noise $\epsilon \sim Gumble(\mu, \beta)$ added to the conceptual score $s_i$ while looking at item $i$. So the probability of $s_i > s_j$ is actually $\Pr(s_i + \epsilon_i > s_j + \epsilon_j)$. Here, we assume $\beta = 1$. Detailed induction process omited here. The result is as follows.

$$-\sum{c_{ij} \cdot \ln( {e^{s_j - s_i} + 1}} )$$

Since the number of comparisons can vary case by case, the actual likelihood is normalized by of total number of comparisons.

$$-(\sum{c_{ij}})^{-1} \sum{c_{ij} \cdot \ln( {e^{s_j - s_i} + 1}} )$$

### gbtl
`gbtl`: Generalized BT-model
Since each judge can have different judgement abilities, by adjusting the $\beta$ parameter noise generation Gumble distribution, we can also estimate the quality of that judge.

$$-(\sum{c_{ij, k}})^{-1} \sum{c_{ij,k} \cdot \ln({e^{\frac{s_j - s_i}{\beta_k}} + 1}})$$

By definition, the Gumble distribution must have $\beta > 0$ . In order to cast the restriction that $\beta > 0$, in actual implementation $\beta = \epsilon^2$ is used, the formula becomes $-(\sum{c_{ij, k}})^{-1} \sum{c_{ij,k} \cdot \ln({e^{\frac{s_j - s_i}{\epsilon_k^2}} + 1}})$

### gbtl-negative

`gbtl-negative`: the restrcition $\beta > 0$ is lifted in `gbtl`.

Suppose a judge have good knowlege of the items, however he tends to provide ratings at the opposite direction to his best knowledge. In such adversarial setting, the fomula is quite similar to previous case. Assume $\beta'_k  = - \beta_k < 0$

$$-(\sum{c_{ij, k}})^{-1} \sum{c_{ij,k} \cdot \ln({e^{\frac{s_j - s_i}{-\beta_k}} + 1}})$$
we can see that it is still calculating the probability of items i better than item j.

Ideally, this algorithm can handle those adversarial judges which know that $s_i > s_j$ but cast a vote that $s_i < s_j$.

$$-(\sum{c_{ij, k}})^{-1} \sum{c_{ij,k} \cdot \ln({e^{\frac{s_j - s_i}{\beta'_k}} + 1}})$$

When $\beta$ is near zero, it will correspond to a perfect judge. When noisy comparison is seen, in order to compensate the likelihood, the $\beta$ tends to be larger and larger. It would be very hard for the optimization algorithm which is a likelihood maximizer to produce gradient that to make the value of $\beta$ to become negative. For this knid of judge, when $\beta$ is positive and very small, it will have smaller likelihood than $\beta$ is positive and have larger value. However, when $\beta$ is negative and very small, it will also have larger likelihood. To overcome this next algorithm is used.

### gbtl-inverse

`gbtl-inverse`: Let $\gamma_k = 1/\beta_k$ to replace the denominator part in `gbtl-negative` as a multiplier for easy optimization. 

This is used to compensate the effect of previous method, so the good judge will have a very large $\gamma_k$, while bad judge can flip sign to become adversarial while the gradient will always in the right direction to make the likelihood larger.

$$-(\sum{c_{ij, k}})^{-1} \sum{c_{ij,k} \cdot \ln({e^{(s_j - s_i) \cdot \gamma_k} + 1}})$$

## Initialization

Instead of put initial value of each parameter randomly, the spectral method described in "Rank Centrality" is used to provide a near estimate of the true utility, so that the gradient descent algorithm may produce better result.

### About Generated Data
The data is generated according to the `gbtl` model. All $w$ is generated by a `spacing` mechanism, adjacent items have the same ratio of $10^{1/n}$, according to "Rank Centrality" paper. So in the implementation, we also need to take $ln$ after generate this power series.

### 1. Count Matrix and Probability Matrix
Count the number of times item $i$ is preferred over $j$ and form a matrix of $c_{ij}$. Denote the matrix derived from all data regardless of distinction of judge as $C$, and for each judge the matrix is denoted as $C_k$.

From these matrix, the ratio $p_{ij} = \frac{c_{ij}}{c_{ij}+c_{ji}}$ is actually an unbiased estimate of $\Pr(i > j)$. Assume $p_{ii} = 0$. We have a matrix of probability for the whole collection $P$ and for each judge $P_k$ is also available.

Since $p_{ij}$ is an unbiases estimate of $\frac{w_{i}}{w_{j} + w_{i}}$, we can use matrix $P$ to derive $n(n-1)/2$ equations for $w_i, i \in [n]$. This will result in a overdetermined system with each equation looks like $w_j = \frac{1-p_{ij}}{p_{ij}} w_i$. In order to make denominator not to be zero, each pair $p_{ij}, p_{ji}$ in $P$ matrix should contain at least one none zero item, which is trivally true. For a certain pair that $p_ij = 1$, then $w_j = 0$. So if each judge give us the perfect judgement then this method would only give us the top item to be $1$. It would be interesting to iteratively remove the top item from the pairwise comparison pool and then solve for the top item for the rest of the items  to figure out the second top item.

Consider the case when there is a perfect judge that will produce correct answer for each pair, however with `btl` model, the judge must have a chance to produce error. Then it is possible to have two items have utilities that is against order with ground truth.

### 2. Transition Matrix
Compute the transition matrix $D$ according to given formula described in "Rank Centrality", note there will be a self loop for each node. Higher the score higher the possibility for the self loop for that item.

We want to study the relationship between three matrices, count probability and transition. First of all, how could the distribution of $s$ or $w$ affect the actual visual representation of the matrix. It is possible to try out arithmetic progression (sum to 1.) and normal distribution with different mean (0.25 0.5 0.75) as well as power progression described in the paper. The effect of $\beta$ is not linear, $\beta = \tanh(b)$ to set the value of $\beta$ when $b = 0.1, 0.3, 0.5, 0.7, 0.9$. It is also preferrable to test the result with $s$ in random order, so the result matrix would look more irregular. This experiment tends to find any visual trace for various configurations. If possible, it would be preferable to just find $\beta$ from these configuration regardless of $s$. The report is attached.

Conceptually, the true value of $s$ is easier to recover if $\beta$ is within certain range, either too small or too large $\beta$ will hamper the algorithm from producing correct value.


### 3. Stationary Distribution
Compute the stationary distribution for matrix $D$ treat it as $w$.
Since the stationary distribution assumes $\sum{w_i} = 1$. When the data is noisy, which is equivalent to have more error comparisons, each element in resulting vector will be close to $1/n$. However, the if all participating judge have make perfect verdict, then the resulting vector will be $[0, 0, \cdots, 1]$, which means it is only possible to distinguish the top item.

### 4. Solving equations

#### Step One
For `gbtl*` algorithms, we first solve for the stationary distribution $\mathbf{w}$ for each judge, let $w_{i,k}$ denote the estimation of $i$th item provided by judge $k$. Suppose there are $n$ items, and $m$ judges. The second equality sign holds because the summation of the stationary distribution is 1.

$$\frac{e^{s_i/\beta_k}}{\sum_{i \in [m]}{e^{s_i/\beta_k}}} = \frac{w_{i,k}}{\sum_{i \in [m]}{w_{i,k}}} = w_{i,k}$$

$$\frac{e^{s_i/\beta_k}/  w_{i,k}}{\sum_{i \in [m]}{e^{s_i/\beta_k}}}  = 1$$

Multiply each side by the denominator (note the summation over $e^x$ should be non-zero. Let $u_{i, k} = e^{s_i/\beta_k}$ then we have:

$${e^{s_i/\beta_k}/  w_{i,k}} = {\sum_{i \in [m]}{e^{s_i/\beta_k}}} $$
$${u_{i,k}/  w_{i,k}} = \sum_{i \in [m]}{u_{i,k}} $$

Let the subscripts start from $1$.

There are $n$ such equation for $i \in [n]$.
Then the coeffficient matrix is:
```
A=
[[1-1/w_{1,k} 1.           1.          ....       1.         ]
 [1.          1-1/w_{2,k}  1.          ....       1.         ]
 [1.          1.           1-1/w_{3,k} ....       1.         ]
 ....
 [1.          1.           1.          1.         1-1/w_{n,k}]]

b = [0., 0., 0., ...., 0.] 
```
Solve for $\mathbf{u}$, in $A\mathbf{u} = b$. The matrix $A$ have a very good change to be full rank, and the solution for the system is $\mathbf{u} = [0, 0, \cdots, 0]$.

The above holds because all we have in comparison is pairwise difference. By fixing one value, it is possible to get other value.
Assume $s_1 = 0$, then $u_{1,k} = 1$. Plug into the equation:

```
A'=
[[1.           1.          ....       1.         ]
 [1-1/w_{2,k}  1.          ....       1.         ]
 [1.           1-1/w_{3,k} ....       1.         ]
 ....
 [1.           1.          1.         1-1/w_{n,k}]]

b' = [1/w_{1,k}-1, -1., -1., ...., -1.] 
```
Solve for $\mathbf{u}'$, in $A'\mathbf{u}' = b'$.
This system of equation have a property that if $s_1$ is shifted by amount of $\delta_s$, then the solution vector will also shift by $\delta_s$, keeping the pair-wise difference the same.

A toy example for pratical understanding, for a specific judge with $\beta = 1$, $\mathbf{s} = [0.7,0.2,0.3,0.5]$, a simple calculation will give $\mathbf{w} = [0.32304109, 0.19593432, 0.21654092, 0.26448367]$

the system $Au = b$ will be overdetermined with n equations and $n-1$ unknown. Directly solve the overdetemined system. $u' = (A^TA)^{-1}A^Tb'$
```python
s = np.array([0.7,0.2,0.3,0.5])
print('s', s)
beta = 1.
w = np.exp(s/beta) / np.sum(np.exp(s/beta))
print('w', w)
A = np.ones(s.shape[0]-1) - np.diag(1. / w[1:])
b = np.array([-1.] * s.shape[0])
b[0] += 1. / w[0]
print('b', b)
A = np.vstack([np.ones((1, s.shape[0] -1)), A])
print('A', A)
u = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
print('u', u)
print('\hat{s}', np.log(u))
```
```
s [0.7 0.2 0.3 0.5]
w [0.32304109 0.19593432 0.21654092 0.26448367]
b [ 2.09558146 -1.         -1.         -1.        ]
A [[ 1.          1.          1.        ]
   [-4.103751    1.          1.        ]
   [ 1.         -3.61806487  1.        ]
   [ 1.          1.         -2.78095173]]
u [0.60653066 0.67032005 0.81873075]  
\hat{s} [-0.5 -0.4 -0.2]
```

#### Step Two
From previous step, we have estimated $s_i / \beta_k$. In this step, by using all $n*m$ such ratios, actual $s_i, \beta_k$ can be derived. One $\beta_k$ is set to $1.$ since only ratio dependency is provided, otherwise there will be infinite number of solutions.

First we want to check whether the *first step* above can provide precise estimation when $\beta$ is involved.
```python
s = np.array([0.3, 0.7, 0.2, 0.5])
betas = np.array([1.0, 2.0, 0.5])
print('s', s)
qs = []

for beta in betas:
    print(s/beta)
    w = np.exp(s/beta) / np.sum(np.exp(s/beta))
    print('w', w)
    A = np.ones(s.shape[0]-1) - np.diag(1. / w[1:])
    b = np.array([-1.] * s.shape[0])
    b[0] += 1. / w[0]
    print('b', b)
    A = np.vstack([np.ones((1, s.shape[0] -1)), A])
    print('A', A)
    u = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
    print('u', u)
    q = np.hstack([np.log(u)])
    print('q=\hat{s/\\beta}', q)
    print('-------------------')
    qs.append(q)
qs = np.vstack(qs)
print(qs)
```

```
s [0.3 0.7 0.2 0.5]
[0.3 0.7 0.2 0.5]
w [0.21654092 0.32304109 0.19593432 0.26448367]
b [ 3.61806487 -1.         -1.         -1.        ]
A [[ 1.          1.          1.        ]
 [-2.09558146  1.          1.        ]
 [ 1.         -4.103751    1.        ]
 [ 1.          1.         -2.78095173]]
u [1.4918247  0.90483742 1.22140276]
q=\hat{s/\beta} [ 0.4 -0.1  0.2]
-------------------
[0.15 0.35 0.1  0.25]
w [0.23376485 0.28552103 0.222364   0.25835011]
b [ 3.2778031 -1.        -1.        -1.       ]
A [[ 1.          1.          1.        ]
 [-2.50236895  1.          1.        ]
 [ 1.         -3.49713076  1.        ]
 [ 1.          1.         -2.87071631]]
u [1.22140276 0.95122942 1.10517092]
q=\hat{s/\beta} [ 0.2  -0.05  0.1 ]
-------------------
[0.6 1.4 0.4 1. ]
w [0.18063269 0.40200545 0.14788954 0.26947231]
b [ 4.53609638 -1.         -1.         -1.        ]
A [[ 1.          1.          1.        ]
 [-1.48752845  1.          1.        ]
 [ 1.         -5.76180339  1.        ]
 [ 1.          1.         -2.71095638]]
u [2.22554093 0.81873075 1.4918247 ]
q=\hat{s/\beta} [ 0.8 -0.2  0.4]
-------------------
[[ 0.4  -0.1   0.2 ]
 [ 0.2  -0.05  0.1 ]
 [ 0.8  -0.2   0.4 ]]
```
The result is promising, by looking at the $s$ estimated, they all differs by a certain ratio which is corresponding $\beta_k$.

Next, assemble all these ratios and assume $\beta_1 = 1.0$, recall we also have $s_1= 0.$, there would be $(n-1)*m$ equations and $m-1 + n_1$ unknown.

```python
n_items = s.shape[0] - 1
# we can either restrict beta_0 or not 
n_judges = betas.shape[0]
A = np.zeros((n_items*n_judges, n_items+n_judges-1))
b = np.ones((n_items*n_judges,)) * 0.

for c_i in range(n_judges):        
        for s_i in range(n_items):
            if c_i != 0:
                A[n_items*c_i+s_i][n_items + c_i - 1] = -qs[c_i][s_i]
            A[n_items*c_i+s_i][s_i] = 1

for s_i in range(n_items):
    b[s_i] = qs[0][s_i]


u = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
print(u)

import scipy.sparse.linalg
import pandas as pd
print(pd.DataFrame(A))
x = scipy.sparse.linalg.lsqr(A, b, show=True)[0]
print(x)
```

```
[ 0.4 -0.1  0.2  2.   0.5]
     0    1    2     3    4
0  1.0  0.0  0.0  0.00  0.0
1  0.0  1.0  0.0  0.00  0.0
2  0.0  0.0  1.0  0.00  0.0
3  1.0  0.0  0.0 -0.20  0.0
4  0.0  1.0  0.0  0.05  0.0
5  0.0  0.0  1.0 -0.10  0.0
6  1.0  0.0  0.0  0.00 -0.8
7  0.0  1.0  0.0  0.00  0.2
8  0.0  0.0  1.0  0.00 -0.4
 
LSQR            Least-squares solution of  Ax = b
The matrix A has        9 rows  and        5 cols
damp = 0.00000000000000e+00   calc_var =        0
atol = 1.00e-08                 conlim = 1.00e+08
btol = 1.00e-08               iter_lim =       10
 
   Itn      x[0]       r1norm     r2norm   Compatible    LS      Norm A   Cond A
     0  0.00000e+00   4.583e-01  4.583e-01    1.0e+00  2.2e+00
     1  1.33333e-01   3.742e-01  3.742e-01    8.2e-01  2.2e-01   1.7e+00  1.0e+00
     2  2.13278e-01   3.131e-01  3.131e-01    6.8e-01  8.8e-02   1.9e+00  3.1e+00
     3  4.00000e-01   1.796e-14  1.796e-14    3.9e-14  9.0e-01   2.0e+00  1.3e+01
 
LSQR finished
Ax - b is small enough, given atol, btol                  
 
istop =       1   r1norm = 1.8e-14   anorm = 2.0e+00   arnorm = 3.2e-14
itn   =       3   r2norm = 1.8e-14   acond = 1.3e+01   xnorm  = 2.1e+00
 
[ 0.4 -0.1  0.2  2.   0.5]
```

Breakdown the last line of the output, which should read as `s = [0.4 -0.1  0.2 ] \beta = [2.   0.5]`, which is exactly the ground truth. 

This experiment showed that if the $\mathbf{w_k}$ obtained from whatever process is indeed an unbiased estimate of above mentioned difference and ratio, then solution should be exactly the ground truth or very close.
**Note that in "Spetral Method", the transition matrix was a derivation of the unbiased estimate.** It is very hard to construct unbiased estimator, "Spectral Method" turned out to be a rather robust one, but we may need a better one.


### 5. Post process
Compute $s_i = log(w_i)$.

+ $s$ is shifted to have minimum value to be 0. $\mathbf{s} = \mathbf{s} - \min{s_i}$.
+ $s$ is then rescaled to be summed to 1. $\mathbf{s}  = \mathbf{s}  / \sum{s_i}$.
+ $\beta$ is also rescaled accordingly with the same ratio as $s$.

<!-- + For `gbtl-*`, after calculate $s$ for all judges, we assume $\beta_1 = 1$, the $\beta_k$ for each judge will be $\beta_k = 1 / \#items  *\sum{s_{1, i}/s_{k, i} }$. Because it is hard to analytically make solution for the $\beta$, this approximation is used. -->


## Optimization

+ Calculate the gradient using likelihood function mentioned above. Update $s$ and $\beta$ simultaneously. It is also possible to do alternating update.

Because in the likelihood function, only difference between $s$ matters, so there will be infinate number of solutions if $s$ is not restricted. Without loss of generality we put $\min(s) = 0$. In `gbtl-*` model, consider we have fixed `s`, but the 
+ $s$ is shifted to have minimum value to be 0. $\mathbf{s} = \mathbf{s} - \min{s_i}$.
+ $\beta$ is scaled to prevent the change of likelihood for next step $\beta = \beta / \sum{s_i}$.
+ $s$ is then nomalized/scaled to be summed to 1. $\mathbf{s} = \mathbf{s} / \sum{s_i}$.


# Code Implementation

`main.py` will load the configuration and save running result in when done. One can setup comment/uncomment for different configurations in main.py and invoke the script sequentially. They can be run simultaneously, the saving process will have a lock to prevent each process writing over existing data.

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


### new initialization method 
result of spectral method followed by 2-step solving method.

Some successful result is shown below.
```
data_name	algo_name	acc
0	be-b1,10-j8-i100-p800	btl-spectral-mle	0.8424
1	be-b1,10-j8-i100-p800	gbtl-spectral_all-do	0.3339
2	be-b1,10-j8-i100-p800	gbtl-spectral_all-mle	0.9160
4	be-b1,10-j8-i100-p800	gbtlinv-spectral_all-mle	0.3811
3	be-b1,10-j8-i100-p800	gbtlneg-spectral_all-mle	0.3969

5	be-b1,10-j8-i100-p8000	btl-spectral-mle	0.9790
6	be-b1,10-j8-i100-p8000	gbtl-spectral_all-do	0.6665
7	be-b1,10-j8-i100-p8000	gbtl-spectral_all-mle	0.9639
9	be-b1,10-j8-i100-p8000	gbtlinv-spectral_all-mle	0.5813
8	be-b1,10-j8-i100-p8000	gbtlneg-spectral_all-mle	0.5814
```

Some times, `gbtl` will give initialization point which is in reverse order.

```
data_name	algo_name	acc
0	po-b5-j4-i10-p800	btl-spectral-mle	1.0000
1	po-b5-j4-i10-p800	gbtl-spectral_all-do	0.9212
2	po-b5-j4-i10-p800	gbtl-spectral_all-mle	1.0000
6	po-b5-j4-i10-p800	gbtlinv-disturb-mle	1.0000
4	po-b5-j4-i10-p800	gbtlinv-spectral_all-mle	1.0000
5	po-b5-j4-i10-p800	gbtlneg-disturb-mle	1.0000
3	po-b5-j4-i10-p800	gbtlneg-spectral_all-mle	1.0000

7	po-b5-j4-i100-p8000	btl-spectral-mle	0.9224
8	po-b5-j4-i100-p8000	gbtl-spectral_all-do	-0.2044
9	po-b5-j4-i100-p8000	gbtl-spectral_all-mle	nan
13	po-b5-j4-i100-p8000	gbtlinv-disturb-mle	0.9837
11	po-b5-j4-i100-p8000	gbtlinv-spectral_all-mle	-0.6483
12	po-b5-j4-i100-p8000	gbtlneg-disturb-mle	0.9833
10	po-b5-j4-i100-p8000	gbtlneg-spectral_all-mle	-0.6499

14	po-b5-j4-i100-p80000	btl-spectral-mle	0.9920
15	po-b5-j4-i100-p80000	gbtl-spectral_all-do	-0.1351
16	po-b5-j4-i100-p80000	gbtl-spectral_all-mle	nan
20	po-b5-j4-i100-p80000	gbtlinv-disturb-mle	0.9985
18	po-b5-j4-i100-p80000	gbtlinv-spectral_all-mle	-0.6650
19	po-b5-j4-i100-p80000	gbtlneg-disturb-mle	0.9984
17	po-b5-j4-i100-p80000	gbtlneg-spectral_all-mle	-0.4681
```

To fix this, in addition to seperate transition matrix for each judge, a grand total transition matrix is added while performing the "Two step" initialization described above.


```
data_name	algo_name	acc
5	be-b1,10-j8-i100-p800	btl-spectral-mle	0.8317
6	be-b1,10-j8-i100-p800	gbtl-spectral_all-do	0.8269
7	be-b1,10-j8-i100-p800	gbtl-spectral_all-mle	0.9044
9	be-b1,10-j8-i100-p800	gbtlinv-spectral_all-mle	0.8342
8	be-b1,10-j8-i100-p800	gbtlneg-spectral_all-mle	0.8304

10	be-b1,10-j8-i100-p8000	btl-spectral-mle	0.9780
11	be-b1,10-j8-i100-p8000	gbtl-spectral_all-do	0.9602
12	be-b1,10-j8-i100-p8000	gbtl-spectral_all-mle	0.9657
14	be-b1,10-j8-i100-p8000	gbtlinv-spectral_all-mle	0.5779
13	be-b1,10-j8-i100-p8000	gbtlneg-spectral_all-mle	0.5779

0	be-b5,100-j4-i15-p800	btl-spectral-mle	0.9923
1	be-b5,100-j4-i15-p800	gbtl-spectral_all-do	0.9899
2	be-b5,100-j4-i15-p800	gbtl-spectral_all-mle	0.9940
4	be-b5,100-j4-i15-p800	gbtlinv-spectral_all-mle	0.9952
3	be-b5,100-j4-i15-p800	gbtlneg-spectral_all-mle	0.9952
```


```
0	po-b5-j4-i10-p800	btl-spectral-mle	1.0000
1	po-b5-j4-i10-p800	gbtl-spectral_all-do	0.9899
2	po-b5-j4-i10-p800	gbtl-spectral_all-mle	1.0000
4	po-b5-j4-i10-p800	gbtlinv-spectral_all-mle	1.0000
3	po-b5-j4-i10-p800	gbtlneg-spectral_all-mle	1.0000

5	po-b5-j4-i100-p8000	btl-spectral-mle	0.9190
6	po-b5-j4-i100-p8000	gbtl-spectral_all-do	0.9080
7	po-b5-j4-i100-p8000	gbtl-spectral_all-mle	nan
9	po-b5-j4-i100-p8000	gbtlinv-spectral_all-mle	-0.3160
8	po-b5-j4-i100-p8000	gbtlneg-spectral_all-mle	-0.3190

10	po-b5-j4-i100-p80000	btl-spectral-mle	0.9920
11	po-b5-j4-i100-p80000	gbtl-spectral_all-do	0.9734
12	po-b5-j4-i100-p80000	gbtl-spectral_all-mle	nan
14	po-b5-j4-i100-p80000	gbtlinv-spectral_all-mle	-0.6623
13	po-b5-j4-i100-p80000	gbtlneg-spectral_all-mle	-0.6625
```

For 3 good judge 1 adversarial judge

```
10	ne-b0.005-j4-i16-p12800	btl-spectral-mle	1.0000
11	ne-b0.005-j4-i16-p12800	gbtl-spectral_all-do	0.9613
12	ne-b0.005-j4-i16-p12800	gbtl-spectral_all-mle	nan
14	ne-b0.005-j4-i16-p12800	gbtlinv-spectral_all-mle	-0.0132
13	ne-b0.005-j4-i16-p12800	gbtlneg-spectral_all-mle	-0.0020

5	ne-b0.005-j4-i16-p800	btl-spectral-mle	0.9824
6	ne-b0.005-j4-i16-p800	gbtl-spectral_all-do	0.9926
7	ne-b0.005-j4-i16-p800	gbtl-spectral_all-mle	nan
9	ne-b0.005-j4-i16-p800	gbtlinv-spectral_all-mle	0.6637
8	ne-b0.005-j4-i16-p800	gbtlneg-spectral_all-mle	0.9961

0	ne-b0.005-j4-i4-p800	btl-spectral-mle	1.0000
1	ne-b0.005-j4-i4-p800	gbtl-spectral_all-do	1.0000
2	ne-b0.005-j4-i4-p800	gbtl-spectral_all-mle	nan
4	ne-b0.005-j4-i4-p800	gbtlinv-spectral_all-mle	1.0000
3	ne-b0.005-j4-i4-p800	gbtlneg-spectral_all-mle	1.0000

20	ne-b0.01-j4-i64-p12800	btl-spectral-mle	0.9900
21	ne-b0.01-j4-i64-p12800	gbtl-spectral_all-do	0.9966
22	ne-b0.01-j4-i64-p12800	gbtl-spectral_all-mle	nan
24	ne-b0.01-j4-i64-p12800	gbtlinv-spectral_all-mle	0.9967
23	ne-b0.01-j4-i64-p12800	gbtlneg-spectral_all-mle	0.9967

15	ne-b0.01-j4-i64-p800	btl-spectral-mle	0.8504
16	ne-b0.01-j4-i64-p800	gbtl-spectral_all-do	0.9380
17	ne-b0.01-j4-i64-p800	gbtl-spectral_all-mle	nan
19	ne-b0.01-j4-i64-p800	gbtlinv-spectral_all-mle	0.9402
18	ne-b0.01-j4-i64-p800	gbtlneg-spectral_all-mle	0.9402
```

More Settings using popular information
```
	data_name	algo_name	acc
4	be-b10,100-j4-i100-p800	btl-spectral-do	0.2923
5	be-b10,100-j4-i100-p800	gbtl-spectral_all-do	0.2494
6	be-b10,100-j4-i100-p800	gbtl-spectral_all-mle	nan
7	be-b10,100-j4-i100-p800	gbtlneg-spectral_all-mle	0.2509

8	be-b3,20-j4-i100-p800	btl-spectral-do	0.4829
9	be-b3,20-j4-i100-p800	gbtl-spectral_all-do	0.4599
10	be-b3,20-j4-i100-p800	gbtl-spectral_all-mle	nan
11	be-b3,20-j4-i100-p800	gbtlneg-spectral_all-mle	0.4655
```