# 🎲 Probability

## [Home](https://github.com/lqminhhh/data-science-interview-prep/blob/main/README.md)

## A. Conditional Probability

**Bayes' Rule:** The conditional probability of A given B

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

- $P(A)$: prior
- $P(B|A)$: likelihood
- $P(B)$: posterior

Note: 
- If $P(A) = P(A|B)$, then $A$ and $B$ are independent, since knowing about $B$ tells us nothing about the probability of $A$ having also occured.
- In ML, the goal of Bayes' Rule is to identify the best conditional distribution for a given variable given the data that is available

## B. Law of Total Probability 

Assume we have serveral disjoint event within $B$ having occured; we can then break down the probability of an event $A$ having also occured thanks to the law of total probability, which is stated as follows:

$$
P(A) = P(A|B_1)P(B_1) + ... + P(A|B_n)P(B_n)
$$

Applications:
- Partitioning events: If we want to model the probability of an event $A$ happening, it can be decomposed into the weighted sum of conditional probabilities based on each possible scenario having occured.
- If asked to assess a probability involving a "tree of outcomes" upon which the probability depends

## C. Counting

If the order of selection of the $n$ items being counted $k$ at a time matters, then the method for counting possible permutations is employed:

$$
n * (n - 1) * ... * (n - k + 1) = \frac{n!}{(n - k)!}
$$

If order of selection does not matter, then the technique to count possible number of combinations is relevant:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

## D. Random Variables

A **random variable (RV)** maps outcomes of a random experiment to real numbers:

$$
X: \Omega \rightarrow \mathbb{R}
$$

Examples: number of clicks, session revenue, time to churn.

### 1. Types
- Discrete
    - Countable values  
    - Defined by **PMF** $P(X=x)$

- Continuous
    - Continuous range of values  
    - Defined by **PDF** $f(x)$  
    - $P(X=x)=0$

### 2. CDF

$$
F(x) = P(X \le x)
$$

- Non-decreasing  
- Fully defines a distribution  

### 3. Expectation & Variance

#### Expectation

$$
E[X] = \sum x p(x) \quad \text{or} \quad \int x f(x)\,dx
$$

**Linearity of expectation**:

$$
E[aX+bY]=aE[X]+bE[Y]
$$

(holds even if dependent)

#### Variance
$$
\text{Var}(X)=E[X^2]-(E[X])^2
$$

Measures spread / uncertainty.

### 4. Independence

$X, Y$ independent if:

$$
P(X,Y)=P(X)P(Y)
$$

If independent:

$$
E[XY]=E[X]E[Y]
$$

Zero correlation $\neq$ independence.

### 5. Common Distributions

| Distribution | Use Case | Parameters | Mean | Variance |
|-------------|---------|-----------|------|----------|
| Bernoulli | Success / failure | $p$ | $p$ | $p(1-p)$ |
| Binomial | # successes in $n$ trials | $n, p$ | $np$ | $np(1-p)$ |
| Poisson | Event count per interval | $\lambda$ | $\lambda$ | $\lambda$ |
| Uniform | Equal probability | $a, b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Normal | Natural variation | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| Exponential | Time until next event | $\lambda$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| Gamma | Time until $k$ events | $k, \theta$ | $k\theta$ | $k\theta^2$ |
| Chi-squared | Variance / goodness-of-fit | $k$ | $k$ | $2k$ |
