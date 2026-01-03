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

## E. Model Training