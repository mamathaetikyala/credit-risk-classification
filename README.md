# credit-risk-classification
credit-risk-classification

# credit-risk-classification - background

In this Challenge, demonstrate various techniques to train and evaluate a model based on loan risk. Dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers.

# Project Title 

credit-risk-classification - peer-to-peer lending services

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Installing](#installing)
- [Usage](#usage)
- [Contributing](#contributing)

## About

Using knowledge of the imbalanced-learn library, a logistic regression model is used to compare two versions of the dataset. First, use the original dataset. Second, resample the data by using the RandomOverSampler module from the imbalanced-learn library.

For both cases, get the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
The purpose of this study is to compare the weather condiitions across the globe for random 600+ cities. 

## Getting Started

Open file in Jupyternotebook. Instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

Run executables credit_risk_classification

mandatory: lending_data.csv in Resources folder

## Installing/Running

Python 3.x.x version to be installed.

Python Packages to be installed :Numpy, pandas, sklearn.metrics, sklearn.model_selection, sklearn.linear_model, imblearn.over_sampling

Jupyter notebook either installed in local machine or Cloud version of Jupyter notebok.

## Usage

Maintain lending_data.csv in Resources folder.

Goto Juyter Notebook and import jupyter source file Weatherpy or VacationPy

Run all Nodes in Jupyter Notebook.

## Overview of the Analysis

Perform analysis on the right machine learning model to be used by peer-to-peer lending services company to check credit worthyness of borrower by loan risk.

## Purpose of the analysis: 

Identify credit worthiness of borrower by using a most appropriate mchine learning model for peer-to-peer lending services company. Use historical lending activity to train model. 

For this use case, imbalanced-learn library, a logistic regression model is used to compare two versions of the dataset. First, use the original dataset. Second, resample the data by using the RandomOverSampler module from the imbalanced-learn library.

For both cases, get the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
The purpose of this study is to compare the weather condiitions across the globe for random 600+ cities. 

## Results

The balanced accuracy scores and the precision and recall scores of all machine learning models as below that were observed as part of the analysis.

* Machine Learning Model 1: Logistics Regression model
  * The Logistics Regression model performs well according to the balanced accuracy score (99%), precision 0.84 and recall 0.94. however this is due to the data being imbalanced. The number of healthy loans (low-risk) highly outweighs the number of non-healthy (high-risk) loans indicating that the model would predict loan status's as healthy better than being able to predict loan status's as non-healthy.

                precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.94      0.89       619

    accuracy                           0.99     19384
   macro avg       0.92      0.97      0.94     19384
weighted avg       0.99      0.99      0.99     19384


* Machine Learning Model 2: the logistic regression model, fit with oversampled data
  * the logistic regression model, fit with oversampled data oversampled model generated same accuracy score of 99% Since original data was imblanced there is no significant improvement observed in overall accuracy. But 99% is very good accuracy score.The oversampled model performs better because it does a exceptional job in catching mistakes such as labeling non-healthy (high-risk) loans as healthy (low-risk). This is analyzed based off of the recall score increasing from the imbalanced model to the oversampled model 0.94 --> 0.99. Precision is same as befire oversampled i.e.0.84. But over all scores improved with this model.
 
                  precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384

## Summary

peer-to-peer lending services company might want a model that requires a higher recall because: in this case "the logistic regression model, fit with oversampled data"

 * healthy loans being identified as a non-healthy loan might be potential loss of business/customer. non-healthy loans being identified as a healthy loan are risky to firm.

 * In ths case "oversampled model" is better recomended model for lending firm.


## Contributing
Contributors names: Mamatha Etikyala

