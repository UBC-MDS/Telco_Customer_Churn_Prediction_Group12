# Telco_Customer_Churn_Prediction_Group12

## Introduction

For this project we are trying to answer the following question: given certain customer characteristics, are they at a high risk of leaving? A natural follow up question that we will also be addressing is: what customer characteristics are positively associated with a high churn risk? This may actually be the more interesting question, as knowing the answer will provide actionable insights to telecommunications companies, allowing them to keep customers for longer. 

The dataset we are using comes from the public IBM github page, and is made available as part of an effort (by IBM) to teach the public how to use some of their machine learning tools. Unfortunately no mention is made of exactly how the data was collected, or who was responsible for the collection. Here is a link to the mini-course that references the dataset we are using. The raw data is here, and lives inside the data folder of the public repository for the mini-course. Each row in the dataset corresponds to a single customer. There are 19 feature columns, along with the target column, “churn”. 

At a high level, our plan to answer the predictive question stated above is to build a predictive classification model. We will focus on models that give some indication as to which features are positively or negatively associated with our target class (ex. A logistic regression model), as this is an important aspect of the overall problem. Our primary objective is to be able to identify customers who are at high risk of churning, therefore we will build a model that aims to reduce type II error at the expense of committing more type I error. 

To begin, we will split the data into train and test sets (80% train/20% test). We will then carry out preliminary EDA on the training data. Specifically, we need to understand whether class imbalance will be an issue in our analysis. Therefore we will present a table that shows the two class counts. Additionally, to try to get some feel for which 
