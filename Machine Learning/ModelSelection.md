# Model Selection
## 1. Subset Selection
### 1.1. Best Subset Selection
<img width="648" alt="Screenshot 2025-02-02 at 17 07 39" src="https://github.com/user-attachments/assets/e33473ed-7129-4c76-8aa7-2c6b57b62341" />

<b> Note: </b> For computational reasons, best subset selection cannot be applied with large p.

### 1.2. Forward Stepwise Selection
<img width="644" alt="Screenshot 2025-02-02 at 17 08 20" src="https://github.com/user-attachments/assets/164fc1d5-438b-4b70-b7e1-331b8f24e316" />

### 1.3. Backward Stepwise Selection
<img width="637" alt="Screenshot 2025-02-02 at 17 08 55" src="https://github.com/user-attachments/assets/3e80acd2-77e5-453e-adf6-3e71a9fe89ab" />

<b> Note: </b> 
- Forward and backward selection searches through $1 + \frac{p(p+1)}{2}$ models, and are not guaranteed to yield the best model.
- Backward selection requires $n > p$ so that the full model can be fit. Forward selection can be used when $n < p$.

### 1.4. Selection Criteria
| Criteria        | Comments                               | Direction |
|-----------------|----------------------------------------|-----------|
| R<sup>2</sup>         | Portion of variation explained by the model | Large     |
| Adjusted R<sup>2</sup> | Adjusted for number of independent variables | Large     |
| Mallows's C<sub>p</sub> | Unbiased estimate of MSE              | Small     |
| AIC             | Measure the risk of overfitting and underfitting | Small     |
| BIC             | Heavier penalty on model with many variables | Small     |

## 2. Shrinkage
### 2.1. Ridge Regression
The ridge regression coefficients are the values that minimize: 

$$
RSS + \lambda \sum_{j=1}^{p} \beta_j^2
$$

where $\lambda > 0$ is the tuning parameter, to be determined seperately.

<img width="324" alt="Screenshot 2025-02-02 at 17 21 48" src="https://github.com/user-attachments/assets/8c29fb56-637f-4afc-b804-65d51d7e11d5" />

- The ridge regression coefficients can change substantially when multiplying a given predictor by a constant. Therefore, it is best to apply ridge regression after standardizing the predictors.
- Select the best $\lambda$ by selection criteria.
- However, the ridge regression will include all $p$ predictors in the final model (disadvantages).

### 2.2. LASSO
The Lasso coefficients minimize:

$$
RSS + \lambda \sum_{j=1}^{p} |\beta_j|
$$

where $\lambda$ is the tuning parameter.

<img width="313" alt="Screenshot 2025-02-02 at 17 25 46" src="https://github.com/user-attachments/assets/81f32807-a6f4-44ba-bd38-2535b9fee032" />

- Lasso has the effect of forcing some of the coefficients to be exactly equal to zero when $\lambda$ is sufficently large.
- It performs variable selection. We say that the lasso yields sparse model.
