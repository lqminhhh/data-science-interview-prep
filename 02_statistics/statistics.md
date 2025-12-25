# 📊 Statistics

## [Home](https://github.com/lqminhhh/data-science-interview-prep/blob/main/README.md)

## A. Properties of Random Variables

The expectation of (average value, or mean) of a random variable is given by the integral of the value of $X$ with its probability density function (PDF) $f_x(x)$:

$$
\mu = E[X] = \int_{-\infty}^{\infty}xf_x(x)dx
$$

and the variance is given by:

$$
Var(X) = E[(X - E[X]) ^ 2] = E[X^2] - (E[X])^2
$$

The variance is always non-negative, and its square root is called the standard deviation, which is heavily used in statistics.

$$
\sigma = \sqrt{Var(X)} = \sqrt{E[(X - E[X]) ^ 2]} = \sqrt{E[X^2] - (E[X])^2}
$$

The conditional values of both the expectation and variance are as follows. For example, consider the case for the conditional expectation of $X$, given that $Y = y$:

$$
E[X|Y = y] = \int_{-\infty}^{\infty}xf_{X|Y}(x|y)dx
$$

For any given random variables $X$ and $Y$, the covariance, a linear measure of relationship between the two variables, is defined by the following:

$$
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
$$

and the normalization of covariance, represented by the Greek letter $\rho$, is the correlation between $X$ and $Y$:

$$
\rho(X, Y) = \frac{Cov(X, Y)}{\sqrt{Var(X)Var(Y)}}
$$

## B. Law of Large Numbers

The Law of Large Numbers (LLN) states that if you sample a random variable independently a large number of times, the measured average value should converge to the random variable's true expectation. Stated more formally,

$$
\overline{X_n} = \frac{X_1 + ... + X_n}{n} \rightarrow \mu, \text{as } n \rightarrow \infty
$$

## C. Central Limit Theorem

The Central Limit Theorem (CLT) states that if you repeatedly sample a random variable a large number of times, the distribution of the sample mean will approach a normal distribution regardless of the initial distribution of the random variable. Formally, the CLT states that:

$$
\overline{X_n} = \frac{X_1 + ... + X_n}{n} \rightarrow \sim N\left(\mu, \frac{\sigma^2}{n}\right); \text{hence } \frac{\overline{X_n} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

## D. Hypothesis Testing

### General Setup

Generally, hypotheses concern particular properties of interest for a given population, such as its parameters, like $\mu$. The steps in testing a hypothesis are as follows:

1. State a null hypothesis $H_0$ and alternative hypothesis $H_1$.
2. Choose significance level $\alpha$
3. Compute test statistic
4. Calculate p-value
5. Reject or fail to reject $H_0$

Hypothesis tests are either one- or two-tailed tests. A one-tailed test has the following types of null and alternative hypotheses:

$$
H_0: \mu = \mu_0 \text{ versus } H_1: \mu < \mu_0 \text{ or } H_1: \mu > \mu_0
$$

whereas a two-tailed test has these types: $H_0: \mu = \mu_0 \text{ versus } H_1: \mu \neq \mu_0$

### Test Statistics

A test statistic is a numerical summary designed for the purpose of determining whether the null hypothesis or the alternative hypothesis should be accepted as correct. More specifically, it assumes that the parameter of interest follows a particular sampling distribution under the null hypothesis.

| Test | Use Case |
|----|---------|
| Z-test | Mean with known variance, large $n$ |
| T-test | Mean with unknown variance |
| Chi-squared | Categorical data, independence |
| ANOVA | Compare means across groups |
| KS test | Distribution comparison |

