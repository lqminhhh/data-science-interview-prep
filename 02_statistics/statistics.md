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

### p-values and Confidence Intervals

- **p-value:** the probability of observing the value of the calculated test statistic under the null hypothesis assumptions. Common level of significance: 0.05

- **Confidence Interval:** A range of values that, if a large sample were taken, would contain the parameter value of interest $(1 - \alpha)\%$ of the time. The general form for the confidence interval around the population mean looks like:

$$
\mu \pm z_{a/1}\frac{\sigma}{\sqrt{n}}
$$

### Type I and II Errors

- **Type I ("False Positive")**: When one rejects the null hypothesis when it is correct
- **Type II ("False Negative")**: When the null hypothesis is not rejected when it is incorrect.

Usually, $1-\alpha$ is referred to as the confidence level, whereas $1 - \beta$ is referred to as the power. If you plot the sample size versus power, generally you should see a larger sample size corresponding to a larger power. It can be useful to look at power in order to gauge the sample size needed for detecting a significant effect. Generally, tests are set up in such a way as to have both $1 - \alpha$ and $1 - \beta$ relatively high (at $0.95$ and $0.8$, respectively)

**Bonferroni correction:** A method to make sure that the overall rate of false positives is controlled within a multiple testing framework. In testing multiple hypotheses, it is possible that if you ran many experiments, you would see a statistically significant outcome at least one. However, a more desirable outcome is to have the overall $\alpha$  of the $100$ tests be $0.05$, and this can be done by setting the new $\alpha$ to $\alpha / n$, where $n$ is the number of hypothesis tests.

## E. MLE and MAP

In maximum likelihood estimation (MLE), the goal is to estimate the most likely parameters given a likelihood function: $\theta_{MLE} = \text{arg max }L(\theta)$, where $L(\theta) = f_n(x_1...x_n|\theta)$.

Since the values of $X$ are assumed to be i.i.d., then the likelihood function becomes the following:

$$
L(\theta) = \prod_{i = 1}^{n}f(x_i|\theta)
$$

The natural log of $L(\theta)$ is then taken prior to calculating the maximum; since log is monotonically increasing function, maximizing the log-likelihood log $L(\theta)$ is equivalent to maximizing the likelihood:

$$
\text{log }L(\theta) = \prod_{i = 1}^{n}\text{log }f(x_i|\theta)
$$

Anotheer way of fitting paramters is through maximum a posteriori estimation (MAP), which assumes a "prior distribution:"

$$
\theta_{MAP} = \text{arg max }g(\theta) f(x_1...x_n|\theta)
$$

where the similar log-likelihood is again employed, and $g(\theta)$ is a density function of $\theta$.