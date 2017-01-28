## Section 4 solutions

1. The average effect on Y of a one unit increase in x3, holding all other predictors (x1 & x2) fixed, is -0.0174

1. We calculate the odds to get e^0.11 = 1.1163. So, all else being equal, a 1 year increase in years of marriage increases the odds of having an affair by a multiplicative factor of 1.1163.  

1. a. Bias is the difference between expected prediction of our model and the correct value we are trying to predict.  
      Variance is the amount by which our prediction would change if estimated using a different training dataset.  

    b. Variance increases, Bias decreases
    
    c. Bias increases, Variance decreases
    
    d. As the penalty term, lamda increases, the flexiblity of lasso/ridge decreases, leading to decreased variance but               increased bias.  Specifically for Ridge, a penalty is placed on the sum of the square of the beta coefficients (excluding       the intercept) in the case of Ridge.  For Lasso, a penalty is placed on the sum of the absolute value of the beta               coefficients. 
    
1. Since this has a non-linear decision boundary, an SVM or a Decision Tree or Random Forest would be a good model.    
    
1. a. and b.


    |      | Blue | Red | p_Blue | p_Red | Gini_Coefficient |
    | -----| -----| -----| -----| -----| -----| 
    |     Bag | 501 | 530 | 0.49 | 0.51 | 0.500 |
    |     Bag1  | 1 |  30 | 0.03 | 0.97 | 0.062 |
    |     Bag2   | 500 | 500 | 0.50 | 0.50 | 0.500 |

    Information Gain going from Bag --> Bag1 and Bag2:
    0.500 - [(31/1031)*0.062 + (1000/1031)*0.500] = 0.013
    
    c.
    
    
    |      | Blue | Red | p_Blue | p_Red | Gini_Coefficient |
    | -----| -----| -----| -----| -----| -----| 
    |     Bag | 501 | 530 | 0.49 | 0.51 | 0.500 |
    |     Bag1  | 100 |  400 | 0.20 | 0.80 | 0.320 |
    |     Bag2   | 401 | 130 | 0.76 | 0.24 | 0.370 |

    Information Gain going from Bag --> Bag1 and Bag2:
    0.500 - [(500/1031) * 0.320 + (531/1031) * 0.370] = 0.154

    d.  Bag1 in part (b) only has 31 marbles.  Splits that create purity in one node but don't have much n-size don't help us           with the classification very much, which is reflected in the weights in the information gain computation.  

    e.  Consider all predictors and all possible split points, then choose the predictor-split combination with greatest             information gain.  
    
1. a. Decision trees suffer from high variance.  For example, if we split the training data into two parts at random, the two         resulting trees could be quite different.  Bagging, or bootstrap aggregation, is a general purpose procedure for reducing       the variance of a statistical learning method.  It is particularly useful in averaging away the (usually high) variance of       individual decision trees.  

   b. Random forests provide improvement over bagged trees by way of a small tweak that *decorrelates* the trees.  In                 particular, each time a split in a tree is considered, a random sample of m < p total predictors is chosen.  Typically we       set m = sqrt(p)
    
    c. Like bagging, boosting is another general purpose procedure.  In contrast to random forests, boosted trees are grown            sequentially:  each tree is grown using information from previously grown trees.  In the case of gradient boosted               regression trees, we have way more tuning parameters:  max depth of each tree (often quite small), minimum samples per          leaf, number of trees grown, learning rate. 
    
       If doing stochastic gradient boosting, we can also take a random subsample of the features (similar to random forest), or        build the tree on a random subset of the training set.  
    

1. The 1st Principal Component is simply a linear combination of the **predictors** that produces the largest **variance**,        subject to **the sum of the square of the coefficients being equal to 1**.  The 2nd principal component is constructed          similarly, subject to the additional constraint that the direction must be *orthogonal* to the first principal component        direction.  The 3rd principal component is constructed similarly, subject to constraint that the direction must be              *orthoginal* to the first and second principal component directions.  And so forth.

   The first k principal components can be thought of as all of your data projected into a *lower*-dimensional space,              specifically k-dimensional.    

   We can tell how much of the variance in the data is explained by the first k principal components by examining the              *scree plot*.    

   To perform Principal Components Regression, we can take the *principal components* and use them simply as predictors in         Linear Regression.  

1.  Both kNN and k-means use distance-based similarity.  In particular, for kNN, we compute the K observations that are nearest to a given test observation which may be very far away in p-dimensional space when is p is large, resulting in poor predictions.

1. Naïve Bayes classifiers are based on applying Bayes’ theorem with strong (naive) independence assumptions between the features.

1.  The following are methods that would likely benefit from first scaling or standardizing the data.  
      * Lasso/Ridge -  Lasso and Ridge coefficient estimates change substantially when multiplying a given predictor by a               constant, therefore it's best to standardize the predictors.  
      * PCA - The first principal component is found by maximizing the variance of a linear combination of predictions, subject         to constraint (see above answer on PCA).  Therefore it's important to first scale and standardize the data.  
        Similar reasons for construction of other principal components.  
      * k-Means/kNN - Distance-based calculations naturally require scaling/standardization.  Otherwise predictors with bigger          spread would have more influence.  
      * SVM - The margin is calculated subject to the sum of the square of the coefficients being equal to 1, so we must                scale/standardize the predictors.  Alternatively, we can view the cost function as loss + penalty and follow a similar          line of reasoning as in the Lasso and Ridge cases. 

