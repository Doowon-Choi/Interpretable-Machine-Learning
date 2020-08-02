# Interpretable-Machine-Learning
This session contains several R codes for interpretable machine learning. In the file, there are useful other online references I have used.

---

### 1. Feature Importance plot 
Random Forest vs SVM   
![rf_svm_feature_imp](https://user-images.githubusercontent.com/69023373/89113993-6329d180-d43d-11ea-82f9-0bc13d7cb9ba.png)


### 2. Partial Dependence Plot (PDP)
Random Forest vs SVM   
![rf_svm_pdp](https://user-images.githubusercontent.com/69023373/89114002-86548100-d43d-11ea-9eb7-e5c7e8473634.png)

### 3. Individual Conditioal Expectation (ICE)
Random Forest vs SVM   
![rf_svm_ice](https://user-images.githubusercontent.com/69023373/89114124-a2a4ed80-d43e-11ea-8ce5-dd2d4b42532d.png)

### 4. 2d-Feature space over prediction   
Random Forest output   
![2d_feature_space](https://user-images.githubusercontent.com/69023373/89114048-0ed32180-d43e-11ea-9f81-c3bd4c092f2b.png)

### 5. Accumulated Local Effects (ALE)
Random Forest vs SVM (Compare ALE with PDP, say conditional distribution vs. marginal distribution)     
![rf_svm_ale_pdp_compare](https://user-images.githubusercontent.com/69023373/89114060-2a3e2c80-d43e-11ea-9f10-52243c2a005c.png)

### 6. Feature Interaction
SVM   
![Feature_interaction](https://user-images.githubusercontent.com/69023373/89114142-d122c880-d43e-11ea-8c09-75caa4201a1d.png)

### 7.Surrogate model
Decision tree model (CART) is used as a surrogate model of a black box model SVM.     
![Surrogate_tree](https://user-images.githubusercontent.com/69023373/89114158-f1eb1e00-d43e-11ea-84d5-6301de78fac7.png)

### 8. Local Interpretable Model-Agnostic Explanations (LIME)
Unlike global model, several particular observations are approximated by interpretable models like linear regression, LASSO, Ridge regression etc.   
Four instances of our interest is compared below under different values of bin.   
![Lime](https://user-images.githubusercontent.com/69023373/89114184-3d9dc780-d43f-11ea-82bb-7655d56279bc.png)

