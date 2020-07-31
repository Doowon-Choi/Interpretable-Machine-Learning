####### Useful online link for stuyding Explainable AI #############

## https://github.com/marcotcr/lime/tree/ce2db6f20f47c3330beb107bb17fd25840ca4606
## https://www.shirin-glander.de/2018/07/explaining_ml_models_code_caret_iml/
## https://github.com/christophM/interpretable-ml-book/blob/master/manuscript/05.8-agnostic-lime.Rmd
## https://bradleyboehmke.github.io/HOML/iml.html#feature-interactions

library('iml')
library("partykit")
library("ggplot2")

#### Load data
data("Boston", package = "MASS")
head(Boston)

## fit machine learning model using partykit ##

## The following decision tree model is already interpretable mode, but here I treat it as a black box model
## Later, SVM and RF model will be used as a black box model ##

mob_mod <- lmtree(medv ~ lstat + rm | zn + indus + chas + nox + age + 
                 dis + rad + tax + crim + ptratio, data = Boston, 
               minsize = 40)

### Using the iml Predictor() that holds the model and the data
X <- Boston[which(names(Boston) != "medv")]
predictor <- Predictor$new(mob_mod, data = X, y = Boston$medv)

### Feature Importance (a.k.a permutation approach)
imp <- FeatureImp$new(predictor, loss = "mae")
imp$results
plot(imp)

########### Feature effects, PDP, ALE, ICE #########
### Partial dependence plot ##
### in the folloiwng results, the yellow line shows PDP
pdp <- Partial$new(predictor, feature="lstat")
pdp$plot()

### PDP ###
pdp <- FeatureEffect$new(predictor, feature="lstat", method = "pdp")
pdp_plot1 = pdp$plot() +scale_x_continuous('lstat') + 
  scale_y_continuous('Predicted medv')

pdp2 <- FeatureEffect$new(predictor, feature="rm", method = "pdp")
pdp_plot2 = pdp2$plot() +scale_x_continuous('rm') + 
  scale_y_continuous('Predicted medv')


gridExtra::grid.arrange(pdp_plot1, pdp_plot2, ncol = 2)


Partial$new(predictor, c("lstat", "rm"))

### ICE ####
ice1 <- FeatureEffect$new(predictor, feature="lstat", method = "ice")
ice_plot1 = ice1$plot() +scale_x_continuous('lstat') + 
  scale_y_continuous('Predicted medv')

ice2 <- FeatureEffect$new(predictor, feature="rm", method = "ice")
ice_plot2 = ice2$plot() +scale_x_continuous('rm') + 
  scale_y_continuous('Predicted medv')

gridExtra::grid.arrange(ice_plot1, ice_plot2, ncol = 2)

#### pdp + ICE ### not recommended, use Partial$new to draw both pdp and ice
pdp_ice = FeatureEffect$new(predictor, feature="lstat", center.at = min(Boston$medv), method="pdp+ice")
pdp_ice$plot() + scale_color_discrete(guide='none')

### Accumulated local effect, ALE ###
ale1 <- FeatureEffect$new(predictor, feature="lstat", method = "ale")
ale1$plot() + ggtitle("ALE")

### compare PDP and ALE ####
pdp = FeatureEffect$new(predictor, feature = "lstat", method = "pdp")
pdp1 = pdp$plot() + ggtitle("PDP")
pdp = FeatureEffect$new(predictor, feature = "rm", method = "pdp")
pdp2 = pdp$plot() + ggtitle("PDP")
ale1 = FeatureEffect$new(predictor, feature = "lstat", method = "ale")$plot() + ggtitle("ALE")
ale2 = FeatureEffect$new(predictor, feature = "rm", method = "ale")$plot() + ggtitle("ALE")

gridExtra::grid.arrange(pdp1, pdp2, ale1, ale2)

## if pdp is similar to ale, that measn two features, lstat and rm, are not correlated ##
## both pdp and ale shows effect over the possible range of features, but some of feature values are unrealistic

#### the following one is interaction plot, before that let's compare the result between SVM and random forest
### train random forest
library("randomForest")
rf <- randomForest(medv ~ ., data = Boston, ntree = 50)

### train svm using radial kernel and determine cost and gamma by cross validation
library("e1071")
x_set <- subset(Boston, select=-medv)
y_set <- Boston$medv

### tuning hyper-parameters of svm by 10-fold cross validation
svm_tune <- tune(svm, train.x = x_set, train.y = y_set, kernel="radial",  ranges=list(cost=10^seq(-1,2,0.5), 
                                                                                      gamma=seq(.5,2,0.2)))
print(svm_tune)
svm_mod = svm(medv ~., data = Boston, kernel="radial", cost = 3.162, gamma = 0.5)

### create predictor from random forest and SVM ###
predictor <- Predictor$new(mob_mod, data = X, y = Boston$medv)

predictor.rf <- Predictor$new(rf, data = X, y=Boston$medv)
predictor.svm <- Predictor$new(svm_mod, data = X, y = Boston$medv)

### see the feature importance 
imp.rf <- FeatureImp$new(predictor.rf, loss="mse")
imp.svm <- FeatureImp$new(predictor.svm, loss="mse")

p1 <- plot(imp.rf) + ggtitle("random forest")
p2 <- plot(imp.svm) + ggtitle("SVM")

gridExtra::grid.arrange(p1, p2, nrow = 1)

pdp_obj2 <- FeatureEffect$new(predictor.rf, feature = c("lstat","rm"), method = "pdp")
#### response values on the 2D of feature space
pdp_obj2$plot()

### compare pdp
rf.pdp <- FeatureEffect$new(predictor.rf, feature = "rm", method = "pdp")
svm.pdp <- FeatureEffect$new(predictor.svm, feature = "rm", method = "pdp")

p1.pdp <- plot(rf.pdp) + ggtitle("random forest")
p2.pdp <- plot(svm.pdp) + ggtitle("SVM")

gridExtra::grid.arrange(p1.pdp, p2.pdp, nrow = 1)

##### Compare ice
rf.ice <- FeatureEffect$new(predictor.rf, feature = "rm", method = "ice")
svm.ice <- FeatureEffect$new(predictor.svm, feature = "rm", method = "ice")

p1.ice <- plot(rf.ice) + ggtitle("pdp-rf")
p2.ice <- plot(svm.ice) + ggtitle("pdp-SVM")

gridExtra::grid.arrange(p1.ice, p2.ice, nrow = 1)

### compare pdp and ale over two models with respect to variable rm

rf.ale = FeatureEffect$new(predictor.rf, feature = "rm", method = "ale")
svm.ale = FeatureEffect$new(predictor.svm, feature = "rm", method = "ale")

p1.ale <- plot(rf.ale) + ggtitle("ale-rf")
p2.ale <- plot(svm.ale) + ggtitle("ale-svm")


gridExtra::grid.arrange(p1.pdp, p2,pdp, p1.ale, p2.ale, nrow=1)

p <- list()
p[[1]] <- p1.pdp
p[[2]] <- p2.pdp
p[[3]] <- p1.ale
p[[4]] <- p2.ale

do.call("grid.arrange", c(p,ncol=2)) 

############ Measuring interactions by SVM model ###############

### interaction by H-statistic. the particular feature interacts with any other features. From this, we find the most
### influential variable interacted with others.
inter_all <- Interaction$new(predictor.svm)
inter_all$plot() + ggtitle("svm")

## rad is selected variable with high degree of interaction. The next step is to calculate two-way interaction w.r.t 'rad'
inter_rad <- Interaction$new(predictor.svm, feature="rad")
inter_rad$plot() + ggtitle("interaction with rad")


########### Surrogate model (interpretable model instead of black box model) using tree ##########
#### In this case, we use ctree in partykit invented by Achime Zeilies,who also invented nice theoretical decision tree MOB
tree <- TreeSurrogate$new(predictor.svm, maxdepth = 3)
tree$r.squared
tree$results
plot(tree)

######## Local interpretable model-agnostic, LIME #########
#### See the behavior of the particular single instance. Personally, this is similar as a locally weighted regression framework####

### Note that we have focused on global interpretability  #####
#The generalized algorithm LIME applies is: http://uc-r.github.io/lime

  #### Every complex model is linear on a local scale, like in traditional analysis, Taylor's expansion has
  #### similar philosophy, rougly.

## Lime procedure ##

# Given an observation, permute it to create replicated feature data with slight value modifications.
# Compute similarity distance measure between original observation and permuted observations.
# Apply selected machine learning model to predict outcomes of permuted data.
# Select m number of features to best describe predicted outcomes.
# Fit a simple model to the permuted data, explaining the complex model outcome with m features from the permuted data weighted by its similarity to the original observation .
# Use the resulting feature weights to explain local behavior.

### There is another package "lime" for the same task, but "iml" pacakge is used below

library(lime)
local_mod <- LocalModel$new(predictor.svm, x.interest = x_set[2,], k = 10) ### I guess LASSO is used here.
local_mod$explain(x_set[2,])
#local_mod$predict(x_set[2,])
local_mod$plot()


######## conduct LIME using "lime" package ###########
### I follow the exisiting online material https://www.data-imaginist.com/2017/announcing-lime/ ####
### http://uc-r.github.io/lime / good introduction to using lime package 
### However, I see the effect of bins that discretize the continuous features on the interpretation. This implies the
### possible limitation of the method


data(biopsy)

# First we'll clean up the data a bit
biopsy$ID <- NULL
biopsy <- na.omit(biopsy)
names(biopsy) <- c('clump thickness', 'uniformity of cell size', 
                   'uniformity of cell shape', 'marginal adhesion',
                   'single epithelial cell size', 'bare nuclei', 
                   'bland chromatin', 'normal nucleoli', 'mitoses',
                   'class')

# Now we'll fit a linear discriminant model on all but 4 cases
set.seed(4)
test_set <- sample(seq_len(nrow(biopsy)), 100)
class_value <- biopsy$class
biopsy$class <- NULL

train_x <- biopsy[-test_set, ]
train_y <- class_value[-test_set]

lda_mod <- lda(train_x, train_y)

#### Start line
### train the explainer // similar as predictor in package 'iml'
explainer_lda_4 <- lime(train_x, model = lda_mod, bin_continuous = TRUE, quantile_bins = FALSE,  n_bins =4)
explainer_lda_8 <- lime(train_x, model = lda_mod, bin_continuous = TRUE, quantile_bins = FALSE,  n_bins =8)

# Use the explainer on new observations
#n_lables = 2 explain the prob of 1 and 0, labels: which lable do you want to explain?. Either of them
# must be specified.
## ridge is used (i.e., highest weights)
explanation_lda_4 <- explain(biopsy[test_set[1:4], ], explainer_lda_4,labels='benign', n_features = 4
                       , feature_select = "highest_weights", kernel_width = 0.5)
explanation_lda_8 <- explain(biopsy[test_set[1:4], ], explainer_lda_8,labels='benign', n_features = 4
                           , feature_select = "highest_weights", kernel_width = 0.5)

### draw plot
plot_bin4 <- plot_features(explanation_lda_4, ncol = 1)
plot_bin8 <- plot_features(explanation_lda_8, ncol = 1)

gridExtra::grid.arrange(plot_bin4, plot_bin8, ncol=2)
plot_explanations(explanation_lda_4)


##### lime supports supervised models produced in caret, mlr, xgboost, h2o, keras, and MASS::lda.

##### Use lime with unspported pacage in R http://uc-r.github.io/lime

