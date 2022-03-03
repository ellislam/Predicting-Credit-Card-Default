# Predicting Credit Card Default

## Introduction
Credit card issuers have faced a credit card debt crisis in recent years, with delinquency rates expected to peak in the third quarter. To increase market share, card-issuing banks offer more cash and credit cards to ineligible applicants. At the same time, most cardholders, regardless of their ability to repay, are overspending on their credit cards, accumulating large amounts of credit card and cash card debt. The crisis has shattered confidence in consumer finance and is a huge challenge for banks and cardholders alike.

In a developed financial system, crisis management is downstream and risk forecasting is upstream. Credit risk here refers to the possibility of late repayment of the credit granted. The main purpose of risk forecasting is to use financial information, such as business financial statements, customer transactions, repayment records, etc., to predict the operating performance or credit risk of individual customers and reduce losses and uncertainties.

Many statistical methods, including discriminant analysis, logistic regression, and Bayesian classifiers, have been used to develop risk prediction models. More advanced methods, deep autoencoders, variational autoencoders can even learn new features that have not yet been explored but are critical to accurate prediction accuracy.

Logistic regression can be thought of as a special case of linear regression models. However, the binary response variable violates the normality assumption of the general regression model. A logistic regression model specifies that an appropriate function for the fitted probability of an event is a linear function of the observed values ​​of the available explanatory variables. The main advantage of this method is that it yields a simple classification probability formula.

## Dataset
In this project, I will explore the dataset from Taiwan, which contains information on default payments, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. Totally, there are 30000 customers' records with 23 features and 1 response (default.payment.next.month)

## Features
There are 23 features, which include all types of data: categorical, discrete and continuous numerical variables.  
* LIMIT_BAL(numerical): Amount of given credit in NT dollars (includes individual and family/supplementary credit)
* SEX(categorical): Gender (1=male, 2=female)
* EDUCATION(categorical): (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
* MARRIAGE(categorical): Marital status (1=married, 2=single, 3=others)
* AGE(numerical): Age in years
* PAY_0, Pay2-Pay6(numerical) : Repayment status in September, Aug,July, June, May, April, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
* BILL_AMT1 to BILL_AMT6(numerical): Amount of bill statement in from  September to April,  2005 (NT dollar)
* PAY_AMT1 to PAY_AMT6 (numerical) : Amount of previous payment  in from September to April, 2005 (NT dollar)
* default.payment.next.month: Default payment (1=yes, 0=no)


## Library Used
```
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.linear_model  import LogisticRegressionCV
```

## Technologies
This project is created with:
* Jupyter Notebook 6.0.3
