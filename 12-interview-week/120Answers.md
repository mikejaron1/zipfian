**Compiled + ordered student answers**

**Ones checked or wrote up are initialed.  Not initialed still need checking.  Feel free to initial as well!**

##Table of Contents

* [Predictive Modeling (19)](#predictive-modeling-19)
* [Programming (14)](#programming-14)
* [Probability (19)  -  done](#probability-19)
* [Statistical Inference (15)](#statistical-inference-15)
* [Data Analysis (27)](#data-analysis-27)
* [Product Metrics (15)](#product-metrics-15)
* [Communication (11)](#communication-11)

##Predictive Modeling (19)

**Modeling: 1 - TL**

Things to consider when choosing a model:
http://scikit-learn.org/stable/tutorial/machine_learning_map/

**Modeling: 2  - TL**

-Dataset shift is a common problem in predictive modeling that occurs when the joint distribution of inputs and outputs differs between training and test stages. Covariate shift, a particular case of dataset shift, occurs when only the input distribution changes. Dataset shift is present in most practical applications, for reasons ranging from the bias introduced by experimental design to the irreproducibility of the testing conditions at training time.

[-Reference](http://mitpress.mit.edu/books/dataset-shift-machine-learning) 

**Modeling: 3 - TL**

What are some ways I can make my model more robust to outliers?

* Identify outlier influence (outliers defined as being outside of 2-3 standard deviations)

* If outlier has little influence, it will not affect the model much and thus can be ignored.  This really depends on method, and even how the model is being constructed.  Some methods are more robust to outliers than others.      

* If outlier is due to poor data collection/entry, can (with caution) replace with mean or median imputation.   

It's important in many supervised learning algorithms to look for outliers up front and decide how to handle them. In the case of unbalanced classes or anomaly detection, those data points may be valuable and are not truly outliers. Other times, those outliers might clearly be corrupt or irrelevant data and can be removed without harm. There is no clear cut rule for how to deal with outliers; you'll have to use your good judgment when it comes up.

**Modeling: 4 - TL**

Squared errors gives relatively high weight to large errors, so it is most useful when large errors are particularly undesirable.

Squared error is appropriate when results are sensitive to larger errors or outliers, while absolute error is best when larger errors or outliers aren't as big of a deal.

**Modeling: 5 - TL**

* For all supervised learning algorithms, we care most of all about generalization on unseen data. That means that during prototyping and model iteration, k-fold cross validation should be what we're aiming to optimize, and in the end, the evaluation metric we care about is performance on hold-out test data.

For a binary classifier, consider...

* F1 - great for a single performance evaluation metric

* Misclassification rate - though especially problematic with unbalanced classes

* Precision/Recall - A better metric than misclassification rate for unbalanced classes

* Receiver Operator Characteristic (ROC) plot - good for seeing sensitivity to threshold parameter.  Can use to observe Precision/Recall tradeoff for different thresholds.  

    * Area under the curve can be used for single performance metric

For error metric for more than 2 groups, [consider this](http://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics)

**Modeling: 6 - TL**

Some binary classification methods:  Logistic Regression, SVM, Decision Tree (and of course also Random Forest, Boosted Trees), kNN, Naive Bayes

#### Logistic regression

* Large data

    * Logistic regression can handle large datasets reasonably well if you train the algorithm with stochastic gradient descent (i.e. you can train the algorithm partially on small batches of training data at a time). In scikit-learn, you can use the SGDClassifier to do a logistic regression with stochastic gradient descent; use the partial_fit method.

    * For the analytical solution (normal equations), the problem becomes intractable either as the number of examples or features grows too large (~ >10k) because you have to take the inverse of a large matrix and solve a system of linear equations.

* Training speed

* Prediction speed

    * Very fast: threshold(sigmoid(dot_product(weights, input)))

* Interpretability

    * Similar to linear regression, the coefficients could be interpreted to indicate the importance of the factors, but you have to be careful about putting the data on a uniform scale and about the explaining away effect.

* Communication

    * Logistic regression is finding a decision boundary that's a linear combination of the features (i.e. linear hyperplane). Basically, it draws a line through the dataset and everything on one side is classified as class 1, and on the other side, class 2.

* Visualization

    * Visualization in high-dimensional space is always tricky, but if you use PCA to reduce the number of dimensions down to 2 or 3, train the logistic regression on the reduced feature space, and visualize the decision boundary (y = dot(weights, x)). If you reduced to 2 dimensions, the boundary will be a line, 3 dimensions, it will be a plane.

* Nonlinearity/power

    * Logistic regression can model nonlinearities if you include feature interactions in the feature space (i.e. add a feature that is a non-linear interaction of two other features - x1 * x2, or x1^2). You can increase the degree arbitrarily but this will exponentially increase your feature space. This is the same for polynomial regression.

* n << p (more features than examples)

    * With any model where you have a small training set, you're liable to overfit so simple models with regularization are key. Logistic regression with L1 regularization fits the bill nicely. Because the training set is small, you can use Bayesian logistic regression and avoid the problem of overfitting altogether - you'd build a probabilistic graphical model using PyMC with all the parameters and run MCMC to get the posterior distribution of the parameters given the data. For forward estimates, you'd use the expected value of each of the parameters.

* Outliers

    * Logistic regression is quite susceptible to outliers, especially as the number of outliers increases. Outliers can increase training time and reduce model performance by warping the decision boundary. It's important in many supervised learning algorithms to look for outliers up front and decide how to handle them. In the case of unbalanced classes or anomaly detection, those data points may be valuable and are not truly outliers. Other times, those outliers might clearly be corrupt or irrelevant data and can be removed without harm. There is no clear cut rule for how to deal with outliers; you'll have to use your good judgment.

* Overfitting

    * As with most of the machine learning algorithms we've learned, logistic regression tries to find the maximum likelihood decision boundary, which leaves us open to overfitting the training data. Cross-validation is your first line of defense against overfitting. Learning curves will give you more insight.

    * How to prevent overfitting: reduce model complexity by removing features, increase regularization parameter (L1 or L2). 

* Hyperparameters

    * Regularization parameter

    * Degree d polynomial terms (for polynomial logistic regression)

    * Threshold for positive classification (typically 0.5)

    * Class weighting for unbalanced classes

    * Parameters for gradient descent:

        * Step size

        * Stopping criterion: minimum change in error or number of iterations

* Generative

    * Logistic regression is a discriminative model, not a generative one.

* Online

    * You can use logistic regression online with stochastic gradient descent.

* Unique attributes

    * The output of logistic regression can be interpreted as a probability of classification.

    * Scale the data if you want the coefficients to give you some indication of feature importance.

* Special use cases

    * Logistic regression probably has the best combination of simplicity and generalizability out of all the classification models and should probably be your first stop in any classification task. You may want to upgrade to more sophisticated models like SVMs, random forests, or AdaBoost, but the simple logistic regression should probably serve as your baseline classifier since it's so simple and produces pretty good results.

####Support Vector Machine

* High dimensionality

    * Works well in high dimensional, nonlinear space, partially due to the kernel method that removes the need to compute the coordinates of the data points in the new feature space.

* Large data / Online

    * Using hinge loss with sklearn's SGDClassifier allows you to do partial_fit, but it'll be the equivalent of a linear SVM (i.e. without the use of a non-linear kernel).

* Training speed

    * Long

* Prediction speed

    * Fast as it only uses a subset of the dataset (the support vectors) to determine the decision boundary.

* Interpretability

    * SVMs don't output a probability of classification nor do they have coefficients to interpret, so are less interpretable than logistic regression or random forests.

* Communication

    * Simply, SVMs find a decision boundary between classes, similar to logistic regression. Explaining the large-margin aspect is a bit more difficult but a simple diagram that shows maximum separability between classes will usually be sufficient.

* Evaluation

    * Same classifier evaluation metrics as logistic regression

* Nonlinearity/power

    * Using nonlinear kernels allow nonlinear decision boundaries without dramatically increasing training time.

* n << p

* Outliers

    * Because of the use of support vectors to build the decision boundary, SVMs are robust to outliers.

* Overfitting

    * The large-margin nature of SVMs gives the model an inherent resistance to overfitting

* Hyperparameters

    * SVM's are relatively black box models requiring little tuning but there are some knobs you can turn

    * Error penalty term C - controls the tradeoff between margin and error

    * Nonlinear kernel choice - degree d polynomial, gaussian (radial basis function)

        * Each kernel has its own hyperparameters

* Generative

    * SVM's are discriminative models

* Unique attributes

    * Large margin classifier

####Decision Tree/Random Forest

####k-Nearest-Neighbors

* Training time

    * Training time is nil since there is no training phase - you only predict.

* High dimensional / large data

    * Performance will suffer in high dimensional space and with many examples since you'll have to compute pair-wise distance metrics for the entire dataset to find the k nearest neighbors.

    * Limited to small and simple datasets for classification.

* Unique attributes / use cases

    * Can be used to fill in missing data by finding k nearest neighbors based on present features and then filling in the missing value based on the average of the nearest neighbors.

    * Make sure to scale your data so distance calculation makes sense.

    * Noisy features will reduce performance, just as they would with k-means, since typically each feature is an equally-weighted dimension in the distance calculation.

* Hyperparameters

    * k - how many neighbors to use

    * neighbor weighting - you can choose to weight the values of the closer neighbors higher than more distant neighbors

#### Naive Bayes

* Works really well in high dimensional space and with huge datasets; all precomputed

* Fast predictor

* Can be used in an online setting

* Handles unbalanced classes seamlessly (priors on labels)

* Generative model means you can generate example text

* Very interpretable because you can inspect the conditional probability tables

* Used to particularly good effect in natural language tasks because of the inherent high dimensionality

* Spam classification was the killer app for Naive Bayes

* Make sure you have Laplace smoothing


**Modeling: 7 - TL**

Regularization methods are used for model selection, in particular to prevent overfitting by penalizing models with extreme parameter values. 

The most common variants in machine learning are *L*₁ and *L*₂ regularization, which can be added to learning algorithms that minimize a[ loss function](http://en.wikipedia.org/wiki/Loss_function) E(*X*, *Y*) by instead minimizing E(*X*, *Y*) + α‖*w*‖, where *w* is the model's weight vector, ‖·‖ is either the *L*₁ norm or the squared *L*₂ norm, and α is a free parameter that needs to be tuned empirically (typically by[ ](http://en.wikipedia.org/wiki/Cross-validation_%28statistics%29)cross-validation). This method applies to many models. When applied in[ ](http://en.wikipedia.org/wiki/Linear_regression)linear regression, the resulting models are termed[ ](http://en.wikipedia.org/wiki/Ridge_regression)ridge or lasso but regularization is also employed in logistic regression, neural nets, SVMs, etc.

*L*₁ regularization is often preferred because it produces sparse models and thus performs feature selection within the learning algorithm, but since the *L*₁ norm is not differentiable, it may require changes to learning algorithms, in particular gradient-based learners.

----------

Regularization is including more information in order to reduce overfitting. 
- Regularization is helpful if two variables are collinear.
- Regularization is helpful if you have a lot of features and want to reduce it to the important features (use Lasso Regression).
- Regularization is helpful for overfitting. 
- Regularization reduces the effect of variables on the direction of the best-fit line. 
- Lasso regression reduces some feature coefficients to 0. 
- Ridge reduces features according to how much predictive power the feature has for the model. If theta is large, the regularization effect is large. 

**Modeling: 8 -- TL**

* Prevent overfitting

* Parsimony:  Easier to explain, faster prediction speed, etc.

* Interpretation:  On top of being easier to explain, may yield more meaningful interpretations.  For example, in the case of linear/logistic regression, multicollinearity amongst the predictors [render the coefficients uninterpretable](http://en.wikipedia.org/wiki/Multicollinearity#Consequences_of_multicollinearity). 

**Modeling: 9  - TL?**

Assume the modeling task is to predict number of retweets after 2 days of observations, but that we have lots of historical data.   

[Heavy](http://arxiv.org/pdf/1304.6777v1.pdf) paper, but get some ideas for thinking about the retweet problem.  [Lighter](http://mitsloanexperts.mit.edu/modeling-twitter/) explanation.  

* One might consider using historical data to model a "universal" retweet atrophy rate for the lifetime of a tweet, given some performance for the first 2 days.  Deviations from the universal atrophy curve might vary depending on the deceleration over the first 2 days, number of followers, popularity of user, attributes of user in context of social graph, topic of the tweet, etc.  

**Modeling: 10  -  TL?**

Analyze social media output (tweets, posts, updates) by location, and use text search for weather related terms or activities (example snow, rain, hot, scarf)

**Modeling: 11  -  TL**

Open-ended question, and depends on the type of feed.  Read about Facebook's [EdgeRank](http://edgerank.net/) to get some ideas.  Or read about [Amazon's collaborative filtering. ](http://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf) 

**Facebook's EdgeRank**

1. Affinity Score

2. Edge Weight

3. Time Decay

#### Affinity Score

Facebook calculates affinity score by looking at explicit actions that users take, and factoring in 1) the strength of the action, 2) how close the person who took the action was to you, and 3) how long ago they took the action.

Explicit actions include clicking, liking, commenting, tagging, sharing, and friending. Each of these interactions has a different weight that reflects the effort required for the action--more effort from the user demonstrates more interest in the content. 

Affinity score measures not only my actions, but also my friends' actions, and their friends' actions. For example, if I commented on a fan page, it's worth more than if my friend commented, which is worth more than if a friend of a friend commented. Not all friends' actions are treated equally. If I click on someone's status updates and write on their wall regularly, that person's actions influence my affinity score significantly more than another friend who I tend to ignore.

Lastly, if I used to interact with someone a lot, but less so now, then their influence will start to wane. Technically, Facebook is just multiplying each action by 1/x, where x is the time since the action happened.

#### Edge Weight

Facebook changes the edge weights to reflect which type of stories they think user will find most engaging. For example, photos and videos have a higher weight than links. Conceivably, this could be adjusted on a per-user level--if Sam tends to comment on photos, and Michelle comments on links, then Sam will have a higher Edge weight for photos and Michelle will have a higher Edge weight for links. 

#### Time Decay

As a story gets older, it loses points because it's "old news."

Facebook is just multiplying the story by 1/x, where x is the time since the action happened. This may be a linear decay function, or exponential--it's not clear.

Additionally, Facebook seems to be adjusting this time-decay factor based on 1) how long since the user last logged into Facebook, and 2) how frequently the user logs into Facebook. 

**Modeling: 12 -- TL**

Make recommendation on friends using triadic closure

Consider : companies position overlap (How many months/years both individuals work together at same company, similar/related field) , school education overlap, strong unconnected people in linkedin connection graph (both individual have many common friends) etc.  For this last part, one might consider using triadic closure, which we talked about in the [markov lecture](https://github.com/zipfian/markov).  

1. Find the people that you have the most friends in common with

2. Triangle closing

3. Who has the most mutual friends?

4. Identify the strength of the relationships that you have with someone (engagement, shared communities, endorsements)

----------

a. Find the people that you have the most friends in common with
b. Triangle closing
c. Who has the most mutual friends?
d. Identify the strength of the relationships that you have with someone (engagement, shared communities, endorsements)

**Modeling: 13 - TL?**

Begin by looking at who they have sent media to in the past. More recent communication indicates these people are still in touch. Also more likely to communicate if they've had 1:1 conversation.  Examine triangular relationships in networks to identify which nodes have many friends in common.

**Modeling: 14  -- TL?**

Build model based on performance of existing franchises to predict success of future opening.  May want to append lots of external data such as type of city, location (street front, plaza), market saturation (how much competition? distance from existing franchises?), etc.   

**Modeling: 15  -- TL?**

Consider some ideas from [Google Instant](http://searchengineland.com/how-google-instant-autocomplete-suggestions-work-62592)

Given a database of previous searches, we would predict the most frequent query of the users eventual search through a "decision tree" like algorithm. For example if you were to start searching with the letter S - the query would go through all possible ‘branches' that start with ‘S' down to the bottom leaves and return the most commonly searched queries. As the next letter is included -- e.g. SE - the search engine would go down a level and then respond with the top most common searched queries given those letters. This would continue until either the search query is completed or the correct query shows up.

If we wanted to have a more personal search completion engine - we would keep a history of all the queries the user typed, then based on the partial word that was typed we would complete the word with queries in the history and suggest several other relevant words based on the top queries from the overall community.

**Modeling: 16 - TL**

Perform feature selection to determine which characteristics of alumni indicate they are likely to give, then use those features to predict which new alumni will likely give.

* Demographics/Personal:  Gender, Age, Marital Status, Income, Years since graduation

* Academic System:  Quality of Institution, Satisfactory interaction with faculty and staff (can obtain via surveys)

* Social networks and other communication tools:  emails, phone calls, previous interactions with alumni, etc. 

Build a model using previous data to see who did donate  (logistic regression, SVM, random forest...)
Consider date of first donation to effectively target recent alumni who might donate sooner

**Modeling: 17** 

**Modeling: 18**

* use historical data

* use probabilistic model with win/loss ratio

**Modeling: 19** 

How can you address this?  If we are simply predicting the probability of a flight delay, then the magnitude of the delay is irrelevant; we can give a binary prediction - delay or no delay.  If we're predicting the length of a delay instead, some consideration should be made about whether the 12hr delays can be considered outliers. We need to look at various other factors in order to determine whether we should remove them from the model. If removed, then the process should be documented and justified. It could also be interesting to remove them one at a time and observe how the model responds.

##Programming (14)

**Programming: 1**

**Programming: 2** 

Given a list of tweets, determine the top 10 most used hashtags.

Using a regular expression to extract all the hashed words into an array of strings; each element representing the hash for each tweet. Then you could pass this array into a bag of words CountVectorizer to count all the words. This will produce a sparse matrix and we can sum the matrix columns. From there we can use argmax to extract the column number of the most popular words. We can use this data to get the column name and therefore the most popular tweets.

```
[Create a Matrix

 Hash Hash Hash Hash Hash

Tweet - 0 0 1 0 0

Tweet - 1 0 0 0 0

Tweet - 1 1 0 0 0

Sum 2 1 1 0 0

argmax() to get the top 10 hashs from the sum row which returns the index of the top 10 tweets.

Convert those values into tweets from the argmax return.]
```

**Programming: 3**

**Programming: 4**

**Programming: 5**

**Programming: 6**

**Programming: 7**

* using cooks distance

**Programming: 8**

**Programming: 9**

**Programming: 10**

Why might a join on a subquery be slow? How might you speed it up?

If using sqlite or something not as powerful as Oracle, the subquery wouldn't be able to utilize indices, thus making the query run slow.

You could make it faster by creating an indexed temporary table of the subquery and using that.

**Programming: 11**

**Programming: 12**

**Programming: 13**
 
```sql
select count(a_id)/count(*) as CTR, data_format(date, '%M') as month
from impressions_table
group by ad_id, data_format(date, '%M')
```

**Programming: 14**

Write a query that returns the name of each department and a count of the number of employees in each:

```sql
EMPLOYEES containing: Emp_ID (Primary key) and Emp_Name
EMPLOYEE_DEPT containing: Emp_ID (Foreign key) and Dept_ID (Foreign key)
DEPTS containing: Dept_ID (Primary key) and Dept_Name.

SELECT Dept_Name, Count(*) FROM DEPTS AS DP, EMPLOYEE_DEPT AS ED
WHERE DP.Dept_ID = ED.Dept_ID GROUP BY Dept_Name;
```

##Probability (19)

**Probability: 1 – TL**

Let P(lineage dies if start with one amoeba) = x

 

Using law of total probability, we see that

```
x = P(x|the one amoeba has 0 offspring)P(the one amoeba has 0 offspring)
 	+ P(x|the one amoeba has 1 offspring)P(the one amoeba has 1 offspring)
 	+ P(x|the one amoeba has 2 offspring)P(the one amoeba has 2 offspring) 

x = 1*0.25 + x*0.25 + (x^2)*0.5

Solve quadratic equation... x = 1/2
```

**Probability: 2 - TL**

```
P(see at least one shooting star in an hour) = 1 – P(see no shoot stars in an hour)

= 1 – (4/5)^4 = 0.5904
```

**Probability: 3 - TL**

Roll the die twice and assign:

```
return 1 if[(1,1), (1,2), (1,3), (1,4), (1,5)]

return 2 if[(1,6), (2,1), (2,2), (2,3), (2,4)]

...

return 7 if[(6,1), (6,2), (6,3), (6,4), (6,5)]

if (6,6), then reroll
```

**Probability: 4 – TL**

Flip the coin twice.  If same (HH or TT), start over.  If HT, count as Heads.  If TH count as Tails.  This will be fair no matter how the coin is weighted. 

**Probability: 5 – TL**

```
|mean1-mean2| > 2 * SD
```

[reference](http://en.wikipedia.org/wiki/Multimodal_distribution#Mixture_of_two_normal_distributions)  

**Probability: 6 – TL**

Let X be drawn from any arbitrary normal distribution. 

Obtain the CDF of X by...

1. Subtracting the mean and dividing by the SD to get the z-score

2. Obtaining the area to the left of z-score (by using the standard normal distribution).

Lastly, rescale to any arbitrary uniform distribution.  For example, suppose CDF(X) = 0.174.  To obtain a random number from uniform (-50, 100), compute 0.174*(100-(-50)) + (-50) = -23.9. 

**Probability: 7 – TL**

A priori, the possibilities are BG, GB, GG, BB, each with equal probability.  If we know that BB is not a possibility, we are left with BG, GB, and GG, each with equal probability.  P(GG|not BB) = 1/3.  

**Probability: 8  - TL**

Let X be the number of children a couple has.  X follows the[‘shifted' Geometric Distribution](http://en.wikipedia.org/wiki/Geometric_distribution) with E(X) = 1/p = 1/0.5 = 2 children.   

The expected gender ratio is ½ boys, ½ girls.

```
Start with no children

Repeat steps

{ Every couple who is still having children has a child.  ½ the couples have males, ½ the couples have females. 

Those couples that have females stop having children
}
```

At every step you have an even number of males and females, and the number of couples reduces by half.

**Probability: 9 – TL**

How many ways can you split 12 people into 3 teams of 4?

Our approach takes multiple stages of combinations. First, from the pool of 12 people, we choose the first team of 4. Next, from the 8 remaining, we choose a second team of 4, leaving us with our final team of 4. The first step is represented by the combination of 12 choose 4, or 12! / (8! * 4!). The second step is 8! / (4! * 4!).

Finally, we've implicitly ordered these teams (i.e. if we have teams of A:[1,2,3,4], B:[5,6,7,8], and C:[9,10,11,12], we've said that [A,B,C] is different from [C,B,A], when for our purposes, it isn't), so we have to divide by the number of possible orders, or 3!. We find that there are 5,775 ways.

**Probability: 10 – TL**

Let X​ be the event of at least one hash collision. The probability of that event is equal to 1-P(not X) = 1-(10/10)*(9/10)*(8/10)*…*(1/10) =   ​1-(10!/(10^10))

Now take a particular number, say 7.  The probability that it is unused is (9/10)^10 = 0.3486784.  Then the expected total number of unused hashes = 10*0.3486784 = 3.486784.   Thus, the expected number of collisions is 10-3.486784 = 6.513216. 

**Probability: 11 – TL**

```
P(UULLL) = 2/5*1/4
P(LLLUU) = 3/5*2/4*1/3
```

**Probability: 12 – TL**

I write a program that should print out all the numbers from 1 to 300, but prints out Fizz instead if the number is divisble by 3, Buzz instead if the number is divisible by 5, and FizzBuzz if the number is divisible by 3 and 5. What is the total number of numbers that is either Fizzed, Buzzed, or FizzBuzzed?

```
Fizzed: 300/3 = 100
Buzzed: 300/5 = 60
Fizzbuzzed: 300/15 = 20
100 + 60 - 20 = 140
```

**Probability: 13 – TL**

Without loss of generality, suppose that we know exactly which adjectives Alice picks.

There are (5 choose 4) = 5 sets of 4 adjectives that Bob can receive that Alice chose. Similarly, there are (19 choose 1) = 19 adjectives that Bob can receive that were NOT given to Alice. 

Note that there are (24 choose 5) different sets of adjectives that a test-taker can receive. 

Numerator is the number of ways that Bob can match Alice on 4 out of 5 adjectives, plus the number of ways that Bob can match Alice on 5 out of 5 adjectives. The denominator is just the number of possible adjective combinations that Bob can receive.

```
[(5 choose 4)*(19 choose 1) + (5 choose 5)*(19 choose 0)]/(24 choose 5) = 4/1771
```

**Probability: 14 – TL**

Let X = number of applications going to the right college 

Breaking up into indicators, we have...

```
X = I_1 + I_2 + … + I_n, where I_k indicates whether or not k-th application went to the right college. 

E(X) = E(I_1) + · · · + E(I_n). 

E(I_k) = P(k-th application went to right college) = 1/n
```

Using linearity of expectation (a common "trick"), we have…

```
E(X) = n*E(I_k) = n*(1/n) = 1
```

**Probability: 15 – TL**

Compared to a taller father, we would expect to be shorter than the father, and compared to a shorter father, we expect to be taller. This is regression to the mean.

**Probability: 16 – TL**

We can formulate this question as a geometric distribution.  What is the expected number of rolls until we get 2 in a row (either HH or TT)?  

Firstly, you have to roll just to get started.  Thereafter, the probability of the roll being the same as the prior one is ½.  Using geometric distribution, E(rolls until 2 in a row) = 1 + 1/(1/2) = 3. 

```
E(rolls until 2 in a row) = E(rolls until HH)/2

E(rolls until HH) = E(rolls until TT) = 6
```

[A more mathematical solution](http://www.quora.com/Whats-the-expected-number-of-coin-flips-until-you-get-two-heads-in-a-row)

**Probability: 17 – TL**

```
Let X = number of flips until heads.  X~Geometric(p=1/2) , so we have...

        	E(X) = 1/p = 1/0.5 = 2

           E(payout) = E(2X-1) = 2*E(X) - 1 = 2*2-1 = 3
```

**Probability: 18  – TL**

Classic Bayes Rule problem. 

```
HH = heads twice
F = fair coin;  F' = biased coin

P(F|HH) = P(HH|F)*P(F) / [  P(HH|F)*P(F) + P(HH|F')*P(F')  ]
= [(1/4)*(1/2)] / [(1/4)*(1/2) + (9/16)*(1/2)] = 4/13
```

**Probability: 19  – TL**

Classic Bayes Rule problem. 

```
10H = heads ten times in a row
F = fair coin;  F' = biased coin

P(F)=0.999;  P(F')=0.001

P(F|10H) = P(10H|F)*P(F) / [  P(10H|F)*P(F) + P(10H|F')*P(F')  ]

= [(0.5^10) * 0.999] / [ (0.5^10) * 0.999+ 1*0.001] = 0.4938211
```

##Statistical Inference (15)

**Statistical Inference: 1 - TL?**

The attribute we are testing for, such as click-through rate, may differ.  But all other attributes are expected to be the same under random assignment.  This could be for example gender, education-level, day of the week, hour of the day.  

If we are testing if education-level was randomly assigned for test group A, one might consider using a [chisquare test](http://en.wikipedia.org/wiki/Chi-square_test) to see if the actual assignments differ from what's expected (proportionate assignment).   

As always when doing multiple hypothesis tests, don't forget the [multiple comparisons problem](http://en.wikipedia.org/wiki/Multiple_comparisons_problem).

**Statistical Inference: 2  -- TL?**

This would be a way to test your experimental framework. In particular, if the two buckets are shown to be significantly different, this could indicate that your sampling or some other part of your experiment is biased.  

For example, a well-known case of setting up a hypothesis test incorrectly, is [repeated significance testing](https://help.optimizely.com/hc/en-us/articles/200040355-Run-and-interpret-an-A-A-test#problems).  

**Statistical Inference: 3**

**Statistical Inference: 4**

There has been a pertubation of the system since some your data is now biased. basically an over representation of one of the classes

**Statistical Inference: 5**

How would you conduct an A/B test on an opt-in feature?

An AB test is the comparison of two alternatives with a desired result. We count the number of people who view each alternative as well as the number of people who opt-in through each alternative and attempt to measure through statistics whether the increased rates are statistically significant. This can be done with a t-test or with Markov Chain Monte Carlo. 

We aren't sure if this question refers to this standard version of A/B testing, or if the opt-in occurs after a user has 'clicked through'. In the latter case, the rate will generally be lower because we are starting with a smaller sample size. It will take longer to collect enough data to detect a statistically significant result.

**Statistical Inference: 6**

**Statistical Inference: 7**

**Statistical Inference: 8**

I have two different experiments that both change the sign-up button to my website. I want to test them at the same time. What kinds of things should I keep in mind?

Make sure that the groups that see each option are not sampled in a biased way, that each experiment varies only in one feature, and that the sample size is big enough to draw statistically significant conclusions. Also try to keep each sample size balanced.

--------

Just change one thing at a time and have individual experiments. Or you can A/B test with all possible combinations and a large enough sample size for all of the groups. Just do one experiment with all combinations. 
Two goals: fastest A/B testing possible; most accurate results

**Statistical Inference: 9**

**Statistical Inference: 10** 

**Statistical Inference: 11**

How would you design an experiment to determine the impact of latency on user engagement?

The first thing you'd want to do is make sure you can measure the latency accurately across all user devices (mobile, desktop, etc.).

There is a two to three second range after which user attention drops off dramatically. You can artificially include a small wait time in your response to check if users become less engaged.

**Statistical Inference: 12** 

**Statistical Inference: 13**

What's the difference between a MAP, MOM, MLE estimator? In which cases would you want to use each?

Maximum Likelihood estimator:

Method of moments estimator:

- mean, variance, skewness, kurtosis

Maximum a posteriori estimator is a mode of the posterior distribution. The MAP can be used to obtain a point estimate of an unobserved quantity on the basis of empirical data. It is closely related to Fisher's method of maximum likelihood (ML), but employs an augmented optimization objective which incorporates a prior distribution over the quantity one wants to estimate. MAP estimation can therefore be seen as a regularization of ML estimation.

-------------
  
MAP: Include a prior for predicting P(A|B) =  P(B|A)*P(A)
MOM: Draw a sample - assume sample is representative of the entire population 
MLE: Look only at P(B|A) for predicting P(A|B)

**Statistical Inference: 14  - TL **

What is a confidence interval and how do you interpret it?

 - An observed interval based on the values from a sample. 
 - A range of values where you can have a certain confidence level that your true population parameter falls within

**Statistical Inference: 15**

##Data Analysis (27)

**Data Analysis: 1**

**Data Analysis: 2**

How much of the variance in the data is explained by the model

* Adjusted R^2 - basically penalizes you for using a complicated model

* RSS

* RMSE

**Data Analysis: 3**

What is the curse of dimensionality?

Things are so far even when they're close together! To elaborate, with many features it is nearly inevitable that distance between similar data points will be high in at least one dimension.

Distance metrics like Euclidean will have trouble identifying clusters.

**Data Analysis: 4**

**Data Analysis: 5**

What are advantages of plotting your data before performing analysis?

Plotting your data before analysis allows you to perform EDA to identify outliers, bad data, and nans that may affect calculations. You could also check distributions. Additionally, features can be plotted against each other to look for correlations and thereby potentially eliminate those and reduce the workload.

**Data Analysis: 6**

How can you make sure that you don't analyze something that ends up meaningless?

Always ask the question "So what" at all stages of an analytics project.



**Data Analysis: 7**

**Data Analysis: 8**

p_value, lasso, ridge

Take out one feature at a time. Look at the performance/accuracy of your model to see how well it performs with vs without that feature.
Linear regression: t-statistic for each feature; get the p-value

**Data Analysis: 9**

How do you deal with some of your predictors being missing?

There are several possible routes to explore:

* We could impute, using several techniques (mean/median, random forest, etc)

* We could gather more data to make sure that the missing features are represented are represented in the new data.

* We could use models that are more robust to missing data, e.g. random forests

* SVD could be useful in some cases where the data is very sparse, e.g. recommendation engines

* Interpolation techniques could be used in many cases, e.g. time series

* In some cases, it may be appropriate to simply fill in the missing values with the mean of its feature

**Data Analysis: 10** 

Multicolinearity

Remedies:
   - Use PCA to reduce features
   - Drop a feature (and re-cross-validate)
   - Lasso vs. Ridge:

**Data Analysis: 11**

Let's say you're given an unfeasible amount of predictors in a predictive modeling task. What are some ways to make the prediction more feasible?

You can perform a number of different feature reduction techniques such as nmf, svd, tf_idf to eliminate stop words, etc. You could also run models that have built in feature reduction methods such as ridge/lasso regression, random forests, stepwise regression, etc.

**Data Analysis: 12**

Now you have a feasible amount of predictors, but you're fairly sure that you don't need all of them. How would you perform feature selection on the dataset?

Check t-statistics on each regressor, look for collinearity, use LASSO or Ridge techniques to penalize with L1 or L2 norms.

**Data Analysis: 13**

**Data Analysis: 14**

the samples are not uniform enough

Probably highly correlated with another variable
You would want to test its importance by taking it out of the model 
Not very interpretable

**Data Analysis: 15**

What is the main idea behind ensemble learning? If I had many different models that predicted the same response variable, what might I want to do to incorporate all of the models? Would you expect this to perform better than an individual model or worse?  The idea is that many weakly predictive models may, through a voting algorithm, provide more accurate predictions. One type of voting algorithm is the weighted average of each models' predictions. Sometimes multiple classifiers are combined using logistic regression.  We would expect this to perform better than an individual model when tuned correctly.

**Data Analysis: 16**

**Data Analysis: 17**

How could you use GPS data from a car to determine the quality of a driver?

GPS could be used to determine speed consistency, excessive acceleration or braking, extreme changes in routes i.e U-turns, going around in circles, etc.

**Data Analysis: 18**

Given accelerometer, altitude, and fuel usage data from a car, how would you determine the optimum acceleration pattern to drive over hills?

Use accelerometer and change in altitude to predict fuel usage. Regression might be a good starting point.

**Data Analysis: 19**

**Data Analysis: 20**

Look at how many hubs and authorities

**Data Analysis: 21**

Given location data of golf balls in games, how would you construct a model that can advise golfers where to aim?  Assuming our data contains information about where the golfer was aiming and/or wind direction and speed, then we can identify the shots that landed closest to the greens and match conditions to suggest a direction.

**Data Analysis: 22**

**Data Analysis: 23**

You have 5000 people that rank 10 sushi in terms of saltiness. How would you aggregate this data to estimate the true saltiness rank in each sushi?

For each sushi, we can calculate the counts for each rating and calculate the distribution curve.

**Data Analysis: 24**

Given data on congressional bills and which congressional representatives co-sponsored the bills, how would you determine which other representatives are most similar to yours in voting behavior? How would you evaluate who is the most liberal? Most republican? Most bipartisan?

Use a similarity metric to build up a graph or similarity matrix of legislators. The similarity could start with number of bills cosponsored and develop from there. To determine ideology, we will need some sort of labels to train on.

**Data Analysis: 26**

market basket

hierarchical clustering

**Data Analysis: 27**

Let's say you're building the recommended music engine at Spotify to recommend people music based on past listening history. How would you approach this problem?

Begin by structuring user preferences quantified on some way (rating, like/dislike). We can create similarity matrices for song-song / artist-artist / user-user similarity to build a collaborative filtering engine. If you don't have ratings, then the number of times a user has listened to a particular song can be used in its place.

##Product Metrics (15)

**Product Metrics: 1**

**Product Metrics: 2**

What would be good metrics of success for a productivity tool? (Evernote, Asana, Google Docs, etc.) A MOOC? (edX, Coursera, Udacity, etc.)

productivity tool: # of members, frequency of use for each, growth rate, drop off rate, duration of use, # of entries

MOOC: all of the above, plus course completion rate, homework completion rate, lecture viewership

---------

For productivity tools: active users, number of notes/user, engagement with the app (churn, last login time, number of notes), time spent inside the app, sentiment analysis on reviews, growth (how quickly is it growing, which users are coming in from certain demographics or customer segments (highly engaged vs not))
Divide users into segments
MOOCs: completion rate, engagement, average lifetime of a user, how many people signed up

**Product Metrics: 3**

What would be good metrics of success for an e-commerce product? (Etsy, Groupon, Birchbox, etc.) A subscription product? (Netflix, Birchbox, Hulu, etc.) Premium subscriptions? (OKCupid, LinkedIn, Spotify, etc.)

E-commerce: Visit rate, conversion rate, whether customers come back and keep purchasing

Subscription: whether customers continue to subscribe, first-time signup rate

Premium subscription: whether customers continue to subscribe, first-time signup rate

---------------

Proportion of page visits (by page) that result in a purchase, weighted by profit from each purchase.
 Which search queries are associated with sales
 Which results for related queries lead to most sales
 Looking at the pages, clicking on other products.
 
   - A subscription product (netflix, hulu, birchbox)
   
 Engagment measures (movies watched per month, avg % of movie watched)
 Recommendation effectiveness (How much do our recs lead to more views)
 Features correlated with continued subscription
 Shares
 
   - Premium subscriptions (OKCupid, LinkedIn, Spotify)
 
 Expected length from free sub to ugrading to premium
 Engagement patterns that lead to premium subs (Conversion rate, churn)

**Product Metrics: 4**

**Product Metrics: 5**

revenues

Customer LTV

**Product Metrics: 6**

A certain metric is violating your expectations by going down or up more than you expect. How would you try to identify the cause of the change?  It could be fruitful to observe the metrics that are varying directly or inversely to the metric at hand.

You could also select samples where the metric is at extremes to see if other features cluster with those samples more than they do with the rest of the data.

**Product Metrics: 7**

**Product Metrics: 8**

You're a restaurant and are approached by Groupon to run a deal. What data would you ask from them in order to determine whether or not to do the deal?

What is the groupon cut? What is your typical engagement in my domain? What happens if they don't use their coupon?

**Product Metrics: 9**

You are tasked with improving the efficiency of a subway system. Where would you start?

This depends on how you define efficiency. Find bottlenecks. Weight bottlenecks by number of people affected. These weightings could be determined by turnstile data or cell phone data.

**Product Metrics: 10**

How far people scroll down their news feed
Where you stop on the page
How quickly you scroll

**Product Metrics: 11**

**Product Metrics: 12**

You are on the data science team at Uber and you are asked to start thinking about surge pricing. What would be the objectives of such a product and how would you start looking into this?

The objective is to maximize profits. The way this is done is to increase price when demand is high, but not to the extent that too many potential customers are lost.

Begin by identifying where and when high demand has occured in the past, and use this to predict where it might be in the future.

**Product Metrics: 13**

**Product Metrics: 14**

What kind of services would find churn (metric that tracks how many customers leave the service) helpful? How would you calculate churn?

Subscriptions and suffer from churn. You can look at the number of users that close their accounts for a given period of time or the users who stop using their accounts. The second metric can be calculated by comparing total users over a time to stagnant accounts and closed accounts for the same period. Closed accounts may be correlated ratio to stagnant accounts.

Therefore, going forward, the calculation of stagnant accounts may be a predictor of churn.

---------

Any service could use churn. Subscriptions most strongly, but all companies with network effects/ products would want to monitor when and why users stop using the product   

Calculated by some variant of (customers lost per month / total customers) = churn rate

**Product Metrics: 15**

Let's say that you're are scheduling content for a content provider on television. How would you determine the best times to schedule content?

Look at expected demographics, viewer numbers, and when those demographics view television. Match content to maximum advertising value.

##Communication (11)

**Communication: 1**

**Communication: 2**

**Communication: 3**

How would you explain an A/B test to an engineer with no statistics background? A linear regression?

An A/B Test is a comparison between two possible scenarios to determine if one is more effective than the other.

A linear regression is trying to estimate an outcome of a potential future input by calculating a line that has the shortest distance between all points and using the formula for that line to create a prediction.

**Communication: 4**

How would you explain a confidence interval to an engi- neer with no statistics background?

What does 95% confidence mean?

A prediction with a confidence interval of x% tells you that an observation has an x% chance of falling within that range.

**Communication: 5**

**Communication: 6**

**Communication: 7**

**Communication: 8**

**Communication: 9**

**Communication: 10**

How would you convince a government agency to release their data in a publicly accessible API?

* Explain an analysis or app that can bring great value if you can get access to the data

* Explain the value of government transparency

* Offer to write the API protocolling yourself

**Communication: 11**
