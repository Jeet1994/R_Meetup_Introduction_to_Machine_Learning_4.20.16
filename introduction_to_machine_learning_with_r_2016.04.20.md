Introduction to Machine Learning with R
========================================================
author: Justin Meyer
date: 4/20/16

How is Machine Learning Used?
========================================================

**Some applications:**
- Driverless cars
- Spam email filtering
- Web search results
- Identify credit card fraud

How is Machine Learning Used?
========================================================

**More applications:**
- Trading stocks
http://www.wsj.com/news/articles/SB10001424052748703834604575365310813948080

- Identifying targets for drone strikes  
http://arstechnica.co.uk/security/2016/02/the-nsas-skynet-program-may-be-killing-thousands-of-innocent-people/

How is Machine Learning Used?
========================================================

**Even more applications:**
- Classifying photos  
https://www.kaggle.com/c/yelp-restaurant-photo-classification

- Recommending movies  
http://www.netflixprize.com/

How is Machine Learning Used?
========================================================

Machine learning is often used for **prediction**. If you can identify a pattern in the past you can **predict the future** (more or less):
- **Predicting** happy customers  
https://www.kaggle.com/c/santander-customer-satisfaction
- **Predicting** search result relevance  
https://www.kaggle.com/c/home-depot-product-search-relevance
- **Predicting** 2016 NCAA basketball tournament results  
https://www.kaggle.com/c/march-machine-learning-mania-2016

How is Machine Learning Used?
========================================================
The textbook application for machine learning is **predicting a binary or numeric 
outcome given one or more categorical or numeric predictors.**

- An application with a binary outcome is known as a *classification* problem.  
  
- An application with a numeric outcome is known as a *regression* problem.

What is Machine Learning?
========================================================

In 1959, Arthur Samuel, a pioneer in the field, defined machine learning as a 
**"Field of study that gives computers the ability to learn without being 
explicitly programmed".**  
  
https://en.wikipedia.org/wiki/Machine_learning

A Simple Example to Explain the Concept
========================================================

We are helping an animal shelter. The shelter runs effective but expensive
ads to let people know about animals that need homes.

The shelter can better target the ads to the animals that need help if we can 
predict which animals will be adopted and which will not.

A Simple Example to Explain the Concept
========================================================

When we take in an animal the staff records: 
- animal name
- date and time
- animal type
- sex
- age
- breed
- color

The staff later records whether the animal was adopted or not.

Without Machine Learning
========================================================

**An analyst would review the data to identify which types of animals are not 
being adopted.**

The analyst might compare:
- cats and dogs
- males and females
- animal age

Without Machine Learning
========================================================

The analyst might find that:
- dogs are more likely to be adopted than cats
- females are more likely to be adopted than males

The shelter would run ads for male cats and not run ads for female dogs. 
**This is more accurate than guessing but machine learning can do better.**

With Machine Learning
========================================================

R will predict which animals will be adopted **without being provided any 
rules to do so.**

R figures out which predictor variables to use and how to use them predict the 
outcome.

This is the "learning" in machine learning. **The computer learns how to make decisions.**

Why is Machine Learning Better?
========================================================

With a few predictors that each have a few categories or a simple distribution 
an analyst can create rules that predict the outcome well.

If there are **lots of predictors** or those **predictors are complicated** a machine 
learning algorithm will probably do better than an analyst.

Why is Machine Learning Better?
========================================================

An analyst can do well if there is just one set of rules to create.

If there are **many different datasets that each require a different set of rules,
machine learning is more efficient.**

For example, the project I'm working on now has 4 outcomes across 4 grades.
Sixteen different sets of rules are required.

Step 1: Data Cleaning and Feature Engineering
========================================================

First we need to **clean the data** and do **"feature engineering."**

- Address missing data
- Recode or transform numeric predictors
- Bin categorical predictors

This can be determined by prior knowledge about the data or the 
requirements of the algorithms we plan to use.

Step 1: Data Cleaning and Feature Engineering
========================================================

We may also need to **scale and center numeric predictors**
and **remove correlated variables.**

Step 2: Split Data into Training and Test Sets
========================================================
Randomly divide the cases in the data into **training** and
**testing** datasets.

Step 3: Select an Algorithm
========================================================

There are hundreds of algorithms (also known as methods or models).

Algorithms apply to:
- classification problems (binary and multi-class)
- regression problems
- both types of problems

Caret has a list of algorithms and applicable problems:  
https://topepo.github.io/caret/modelList.html

Common Types of Algorithms
========================================================
- Logistic and linear **regression**
- **Decision trees**
![alt text](introduction_to_machine_learning_decision_tree_example.png)

Common Types of Algorithms
========================================================

- **Random forests** are lots of decision trees combined
- **Bayesian methods**
- **Neural networks**

See the excellent (and free!) *Introduction to Statistical Learning* at http://www-bcf.usc.edu/~gareth/ISL/ for more detail.

Step 4: Train the Algorithm
========================================================
**Train the algorithm on the training dataset.** The algorithm will "learn" how to 
make predictions.

Step 5: Test the Algorithm
========================================================
**Predict outcomes on the test dataset** using the trained algorithm.

Using data to check that the algorithm works properly on new data avoids overfitting.

Overfitting
========================================================

Overfitting is when the algorithm has trained itself very well to the training 
data but **cannot make accurate predictions on new data.**

Imagine a person who can play a few memorized songs on piano but can't adapt to
other songs or other instruments.

Step 6: Evaluate Algorithm Performance
========================================================

Classification problems can be evaluated using accuracy. How many cases did the 
algorithm predict accurately?

**75 out of 100 cases predicted accurately = 75% accuracy**

If the algorithm can't do better than just guessing (known as the "no information rate")
then it has no value.

Step 6: Evaluate Algorithm Performance
========================================================

Another method is the "confusion matrix," again showing 75/100 correct:

|            |Actual      |        |
|:-----------|:-----------|:-------|
|Predicted   |Not Adopted |Adopted |
|Not Adopted |35          |13      |
|Adopted     |12          |40      |

The confusion matrix addresses true positives, false positives, true negatives,
and false negatives.

Step 6: Evaluate Algorithm Performance
========================================================

Classification problems can also be evaluated using a receiver operating 
characteristic curve chart.

<img src="http://gim.unmc.edu/dxtests/roccomp.jpg" height = "500" width = "500"/>

Step 6: Evaluate Algorithm Performance
========================================================

An ROC curve is the plot of the true positive rate against the false positive rate
across all possible probability cut points. http://www.math.utah.edu/~gamez/files/ROC-Curves.pdf 

<img src="http://gim.unmc.edu/dxtests/roccomp.jpg" height = "500" width = "500"/>

Step 6: Evaluate Algorithm Performance
========================================================

Each point is a true positive/false positive pair corresponding to a particular probability cut point. http://gim.unmc.edu/dxtests/ROC2.htm

<img src="http://gim.unmc.edu/dxtests/t4roc.jpg" height = "500" width = "500"/>

Step 6: Evaluate Algorithm Performance
========================================================

Regression problems can be evaluated using root mean squared error, with small 
values being desirable.

For each case subtract the predicted value from the actual value,  
then square that difference,  
then take the mean of all the squared values,  
and finally take the root of the mean.


```r
RMSE <- sqrt(mean((y-y_pred)^2))
```

https://www.kaggle.com/wiki/RootMeanSquaredError

Step 7: Use the Algorithm to Make Predictions on New Data
========================================================

**Repeat any data cleaning and feature engineering on the new dataset**
before applying the algorithm.

For example, the value of 46 might be 0.55 after centering and scaling. If a
decision tree uses the value of 0.6 for a decision, 0.55 is under the cut but 46 
is over.

Step 7: Use the Algorithm to Make Predictions on New Data
========================================================

**Use the algorithm to make predictions on a new 
dataset.**

Returning to the animal shelter example, when a new animal is brought to the 
shelter apply the algorithm to determine if that 
animal should be advertised.

Machine Learning Tools
========================================================

**R**

Python is an alternative

Machine Learning Tools
========================================================

To use each machine learning algorithm in R requires the appropriate package. 
For example, you might use a random forest from the randomForest package. 
Unfortunately each of these algorithms is accessed using different code and 
returns different output.

The **caret package** solves this by providing a consistent interface and output.

Machine Learning Tools
========================================================

**Caret** also provides tools for:
- data splitting
- pre-processing
- feature selection
- model tuning using resampling
- variable importance estimation

http://topepo.github.io/caret/index.html

Machine Learning Tools
========================================================

It can take a lot of code to run algorithms and keep all of the results organized.

The **EWStools package** by Jared Knowles solves this by running all specified algorithms and returning a dataframe with the train and test results.

https://github.com/jknowles/EWStools

Machine Learning Tools
========================================================

Sometimes ensembling (combining) algorithms produces a better prediction.

The **caretEnsemble** package ensembles algorithms.

https://cran.r-project.org/web/packages/caretEnsemble/index.html

Machine Learning Fun With Kaggle
========================================================

http://www.kaggle.com allows users to download data files, apply machine learning to them to make predictions, and compete with other users for the most accurate predictions.

Machine Learning Fun With Kaggle
========================================================

Competitors **train an algorithm on the training dataset** and **submit predictions on the testing dataset.**

The public leaderboard shows competitors' accuracy in predicting half of the testing dataset. The private leaderboard, hidden until the competition is over, shows accuracy on the other half of the testing dataset.
 
Some Drawbacks to Machine Learning
========================================================

- **Transparency may be limited** due to model complexity. Users of the 
predictions may not be comfortable if they don't understand how the the predictions 
were made.
- Can require **significant computing resources.**
- Even with a very accurate algorithm **some predictions will be wrong.**
- Predicting the future can have **unintended consequences.** If we predict something negative rather than trying to prevent it people may see it as inevitable.

Machine Learning Resources
========================================================

- **An Introduction to Statistical Learning: with Applications in R** by Gareth James, Daniela Witten and Trevor Hastie  
http://www-bcf.usc.edu/~gareth/ISL/
- **Applied Predictive Modeling** by Max Kuhn and Kjell Johnson  
http://appliedpredictivemodeling.com/

Machine Learning Resources
========================================================

- Caret documentation  
http://topepo.github.io/caret/
- Coursera machine learning class  
https://www.coursera.org/learn/machine-learning
- Kaggle  
https://www.kaggle.com/
