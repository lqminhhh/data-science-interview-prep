# 🤖 Machine Learning

## A. What to Expect for ML Interview Questions:

- **Conceptual:** Do you have a strong theoretical ML background?
    - E.g.: "How does PCA work?", "What is your favorite ML model?", etc.
- **Resume-driven:** Have you actively applied ML before?
    - Be sure to explain every ML you mentioned in your resume
- **End-to-end modeling:** Can you apply ML to hypothetical business problem
    - E.g.: "How would you match Uber drivers to riders?", "How would you build a search autocomplete feature for Pinterest?" etc.

## B. Linear Algebra

### Eigenvalues and Eigenvectors

For some $n \times n$ matrix $A$, $x$ is an eigenvector of $A$ if: $Ax = \lambda x$, where $\lambda$ is a scalar. A matrix can represent a linear transformation and, when applied to a vector $x$, results in another vector called an *eigenvector*, which has the same direction as $x$ and in fact $x$ multiplied by a scaling factor $\lambda$ known as an *eigenvalue*.

### Eigendecomposition

The decomposition of a square matrix into its eigenvectors is called *eigendecomposition*. However, not all matrices are square. Non-square matrices are decomposed inusing a method called singular value decomposition (SVD). A matrix to which SVD is applied has a decomposition of the form: $A = U \sum V^T$, where $U$ is an $m \times n$ matrix, and $V$ is an $n \times n$ matrix.

## C. Gradient Descent

One popular optimization method is *gradient descent*, which takes small steps in the direction of steepest descent for a particular objective function. It's akin to racing down a hill. To win, you always take a "next step" in the steepest direction downhill.

![alt text](../images/image.png)

For convex functions, the gradient descent algorithm eventually finds the optimal point by updating the below equation until the value at the next iteration is very close to the current iteration (convergence):

$$
x_{t + 1} = x_t - \alpha_t \nabla f(x_t)
$$

that is, it calculates the negative of the gradient of the cost function and scales that by some constant $\alpha_t$, which is known as the learning rate, and then moves in that direction at each iteration of the algorithm.

**Note**: We can use a version of gradient descent called *stochastic gradient descent (SGD)* which adds an element of randomness so that the gradient does not get stuck. SGD uses one data point at a time for a single step and uses a much smaller subset of data points at any given stepo, but is nonetheless able to obtain an unbiased estimate of the true gradient. Alternatively, we can use *batch gradient descent (BGD)*, which uses a fixed, small number (a mini-batch) of data points per step.

## D. Model Evaluation and Selection

- Model evaluation: The process of evaluating how well a model performs on the *test set* after it's been trained on the *train set* (usually 80-20)

- Model selection: The process of selecting which model to implement after each model has been evaluated.

| Metric | Definition | Use Cases |
|------|-----------|-----------|
| MAE (Mean Absolute Error) | $\frac{1}{n}\sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | Regression when all errors are equally important and interpretability matters |
| MSE (Mean Squared Error) | $\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | Penalizes large errors more; useful when big mistakes are costly |
| RMSE (Root MSE) | $\sqrt{MSE}$ | Same units as target; common in regression benchmarking |
| $R^2$ (Coefficient of Determination) | $1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}$ | Explains variance captured by the model; not ideal alone for prediction quality |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Classification with balanced classes |
| Precision | $\frac{TP}{TP + FP}$ | When false positives are costly (e.g., fraud alerts) |
| Recall | $\frac{TP}{TP + FN}$ | When false negatives are costly (e.g., disease detection) |
| F1-score | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Imbalanced classification requiring balance between precision and recall |
| ROC–AUC | Area under ROC curve | Measures ranking ability across thresholds; robust to class imbalance |
| Log Loss | $-\frac{1}{n}\sum [y\log(p)+(1-y)\log(1-p)]$ | Probabilistic classification; penalizes overconfident wrong predictions |

### Bias-Variance Trade-off

With any model, we are usuallying trying to estimate a function $f(x)$, which predicts our target variable $y$ based on our input $x$:

$$
y = f(x) + \omega
$$

where $\omega$ is noise, not captured by $f(x)$, and is assumed to be distributed as a zero-mean Gaussian random variable for certain regression problems. We can decompose the error of $y$ into the following:

1. **Bias:** how close the model's predicted values come to the true underlying $f(x)$ values, with smaller being better

2. **Variance:** the extent to which model prediction error changes based on training inputs, with smaller being better

3. **Irreducible error:** variation due to inherently noisy observation processes

Example:
- Linear regression: high bias, low variance
- Neural networks: low bias, high variance

![alt text](../images/image-1.png)

### Model Complexity and Overfitting

![alt text](../images/image-2.png)

### Regularization

*L1 (LASSO):* The Lasso coefficients minimize:

$$
RSS + \lambda \sum_{j=1}^{p} |\beta_j|
$$

where $\lambda$ is the tuning parameter.

<img width="313" alt="Screenshot 2025-02-02 at 17 25 46" src="https://github.com/user-attachments/assets/81f32807-a6f4-44ba-bd38-2535b9fee032" />

- Lasso has the effect of forcing some of the coefficients to be exactly equal to zero when $\lambda$ is sufficently large.
- It performs variable selection. We say that the lasso yields sparse model.

*L2 (Ridge):* The ridge regression coefficients are the values that minimize: 

$$
RSS + \lambda \sum_{j=1}^{p} \beta_j^2
$$

where $\lambda > 0$ is the tuning parameter, to be determined seperately.

<img width="324" alt="Screenshot 2025-02-02 at 17 21 48" src="https://github.com/user-attachments/assets/8c29fb56-637f-4afc-b804-65d51d7e11d5" />

- The ridge regression coefficients can change substantially when multiplying a given predictor by a constant. Therefore, it is best to apply ridge regression after standardizing the predictors.
- Select the best $\lambda$ by selection criteria.
- However, the ridge regression will include all $p$ predictors in the final model (disadvantages).

**Note**: The L1 and L2 penalties can also be linearly combined, resulting in a popular form of regularization called *elastic net*.

### Interpretability and Explainability

- Linear models: have weights which can be visualized and analyzed to interpret the decision making
- SHAPP (SHapley Additive exPlanation): uses "Shapley" values to denote the average marginal contribution of a feature over all possible combinations of inputs.
- LIME (Local Interpretable Model-agnostic Explanations): uses sparse linear models built around various predictions to understand how any model performs in that local vicinity

## E. Model Training

### Cross-Validation

Idea: Running the algorithm on subsamples of the training data and evaluating model performance on the portion of the data that was excluded from the subsample. This process is repeated many times for the different subsamples, and the results are combined at the end.

One popular way to do cross-validation is called *k-fold cross-validation*. The process is as follows:
1. Randomly shuffle data into equally-sized blocks (folds).
2. For each fold $k$, train the model on all data except for fold $i$, and evaluate the validation error using block $i$.
3. Average the $k$ validation errors from step 2 to get an estimate of the true error.

![alt text](../images/image3.png)

Another form is *leave-one-out cross-validation*. LOOCV is a special case of *k-fold* cross validation where $k$ is equal to the size of the dataset ($n$). That is, it is where the model is testing on every single data point during the cross-validation.

### Bootstrapping and Bagging

- Bootstrapping: drawing observations from a large data sample repeatedly (sampling with replacement) and estimating some quantity of a population by average estimates from multiple smaller samples. Use cases: small dataset, class imbalance.

- Ensemble learning: the process of averaging estimates from many smaller models into a main model. Each individual model is produced using a particular sample from the process. This process of bootstrap aggregation is also known as *bagging*.

### Hyperparameter Tuning

- Grid search: forming a grid that is the Cartesian product of those parameters and then sequentially trying all such combinations and seeing which yields the best results.

- Random search: define a distribution for each parameter and randomly sample from the joint distribution over all parameters.

### Training Times and Learning Curves

- Training times: Another facter to consider when it comes to model selection. While you can always train more complex models that might achieve marginally higher model performance metrics, the trade-off versus increased resource usage and training time might make such a decision suboptimal.

- Learning curvse: plots of model learning performance over time. The y-axis is some metric of learning (e.g. classification accuracy), and the x-axis is experience (time).

![alt text](../images/image4.png)

## F. Linear Regression

Linear regression is a form of *supervised learning*, where a model is trained on labeled input data. The goal is to estimate a function $f(x)$, such that each feature has a linear relationship to the target variable $y$, or:

$$
y = X \beta
$$

where $X$ is a matrix of predictor variables and $\beta$ is a vector of parameters that determines the weight of each variable in predicting the target variable.

### Evaluating Linear Regression

Evaluation of this model is built on the concept of *residual*: the distance between what the model predicted versus the actual value. Linear regression estimates $\beta$ by minimizing the residual sum of squares (RSS), which is given by the following:

$$
RSS(\beta) = (y - X \beta)^T(y - X \beta)
$$

Two other sum of squares concepts to know besides the RSS are the total sum of squares (TSS) and eplained sum of squares (ESS). The total sum of squares is the combined variation in the data (ESS + RSS). $R^2$, a popular metric for assessing good-ness-of-fit, is given by $R^2 = 1 - \frac{RSS}{TSS}$. It ramges between 0 and 1m and represents the proportion of variability in the data explained by the model.

![alt text](../images/image5.png)

Other prominent error metrics:
- MSE (Mean squared error): mesaures the *variance* of the residuals -> penalizes larger errors more than MAE, making it more sensitive to outliers
- MAE (Mean absolute error): measures the *average* of the residuals.

### Subset Selection

- Best subset selection: try eaach model with $k$ predictors, out of $p$ possible ones, where $k<p$. Then, you choose the best subset model using a regression metric like $R^2$ -> **Can be computationally infeasible as $p$ increases**
- Stepwise selection: We aim to find a model with high $R^2$ and low RSS, while considering the number of predictors using metrics like AIC or adjusted $R^2$.
    - Forward selection: start with an empty model and iteratively add the most useful predictor
    - Backward selection: start with the full model and iteratively remove the least useful predictor.

### Assumptions

Four main assumptions to prevent erroneous results:
- **Linearity:** The relationship between the feature set and the target variable is linear
- **Homoscedasticity:** The variance of the residuals is constant.
- **Independence:** All observations are independent of one another.
- **Normality:** The distribution of $Y$ is assumed to be normal.

### Avoid Linear Regression Pitfalls

**Heteroscedasticity:** If the residuals of the residuals is not constant, then *heteroscedasticity* is most likely presented, meaning that the residuals are not identically distributed

![alt text](../images/image6.png)

Another useful diagnostic plot is the scale-location plot, which plots standardized residuals versus the fitted values. If the data shows heteroscedasticity, then you will not see a horizontal line with equally spread points.

![alt text](../images/image7.png)

**Normality:** We can test if the residuals are normally distributed through a QQ plot.

![alt text](../images/image8.png)

![alt text](../images/image9.png)

**Outliers:** this can have an outsized impact on regression results. One of the popular method to check this is examining *Cook's distance*, which is the estimate of the influence of any given data point. It takes into account the residual and leverage (how far away the $X$ value differs from that of other observations) of every point.

**Multicollinearity:** We can check this by examining the variance inflation factor (VIF), which quantifies how much the estimated coefficients are inflated when multicollinearity exists. Methods to address multicollinearity include removing the correlated variables, linearly combining the variables, or using PCA/PLS (partial least squares).

**Confounding Variables**: Multicollinearity is an extreme case of *confounding*, which occurs when a variable (but not the main independent or dependent variables) affects the relationship between the independent and dependent variables. This can cause invalid correlations. Confounding can occur in many ways:
- Selection bias: where the data are biased due to the way they were collected (e.g. group imbalance)
- Omitted variable bias: occurs when important variables are omitted, resulting in a linear regression model that is biased and inconsistent. Omitted variables can stem from dataset generation issues or choices made during modeling.

## G. Generalized Linear Models (GLMs)

The GLM is a generalization of linear regression that allows for the residuals to not just be normally distributed. The three common components to any GLM are:

|**Link Function**|**Systematic Component**|**Random Component**|
|-|-|-|
|$ln \lambda_i$|$=b_0 + b_1 x_i$|$+\epsilon$|
|$y_i$|$\sim \text{Poisson}(\lambda_i)$||

## H. Classification