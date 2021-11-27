Telco_Customer_Churn_Prediction_Report
================
zihan zhou
26/11/2021

``` r
knitr::include_graphics("../results/confusion_matrix.png")
```

<img src="../results/confusion_matrix.png" width="30%" />
<img src="../results/figure_3_numeric_feat_corr.png" width="30%" />
<img src="../results/figure_4_numeric_feat_pairplot.png" width="30%" />
<img src="../results/table_1_churn_dist.png" width="30%" />
<img src="../results/table_2_cat_unique_values.png" width="30%" />

# Summary

In this project, we attempt to examine the following question: Consider
certain telecommunications customer characteristics, predict the
likelihood that a given customer is likely to churn, and further
understand what customer characteristics are positively associated with
high churn risk. We performed a analysis, which yield the

# Introduction

Customer churn risk, the risk of customers switching to another company,
is considered one of the most significant threats to the revenue of
telecommunication companies. (Pustokhina et al. (2021)) The average
churn rate in the telecom industry is approximately 1.9% per month,
which translates to discontinued service for one in fifty subscribers of
a given company.(Team (2020)) Moreover, it is known that the cost of
acquiring new customers is significantly higher than the cost of
retaining them. (Pustokhina et al. (2021)) Thus, it is clear that
reducing churn risk and increasing customer retention rate is a key
strategic challenge in the telecommunication industry. In our project,
we are identifying a predictive classification model that can provide
insights into which customers (based on their traits) are at higher risk
of churning. Answering this question has important practical managerial
implications for telecommunication companies because it is a crucial
step in understanding how to reduce this risk and thus create higher
customer lifetime value. Further, this predictive tool will be
considered a contribution to the modern telecommunication customer
relationship management systems.

# Methods

## Data

The dataset we are using comes from the public IBM github page, and is
made available as part of an effort (by IBM) to teach the public how to
use some of their machine learning tools. Unfortunately no mention is
made of exactly how the data was collected, or who was responsible for
the collection. Here is a link to the
[mini-course](https://developer.ibm.com/patterns/predict-customer-churn-using-watson-studio-and-jupyter-notebooks/)
that references the dataset we are using. The raw data is
[here](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv),
and lives inside the data folder of the [public
repository](https://github.com/IBM/telco-customer-churn-on-icp4d) for
the mini-course. Each row in the dataset corresponds to a single
customer. There are 19 feature columns, along with the target column,
“churn”.

## Analysis

We used the k-nearest neighbors (k-nn) algorithm to build a
classification model to predict which customer features (19 in total,
see in the feature columns) can lead to higher churn risk (see in the
churn column). We used the Python language(Van Rossum and Drake 2009)
and the following Python packages were used to perform this analysis:
docopt (de Jonge 2020), os (Van Rossum and Drake 2009), scikit-learn
(Pedregosa et al. 2011), Pandas (team 2020), Numpy Array(Harris et al.
2020), matplotlib (Hunter 2007)and altair (VanderPlas et al. 2018). We
used the R language(**python?**) and the following Python packages were
used to perform this analysis: knitr (Xie 2014), tidyverse (Wickham
2017). Our code for the analysis and our related resources and progress
reports can be found here:
(<https://github.com/UBC-MDS/Telco_Customer_Churn_Prediction_Group12>)

# Results & Discussion

To begin, we will split the data into train and test sets (80% train/20%
test). We will then carry out preliminary EDA on the training data.
Specifically, we need to understand whether class imbalance will be an
issue in our analysis. Therefore, we will present a table that shows the
two class counts. For each of our categorical/binary features,
distributions across our two classes will be plotted as stacked bar
charts. For each of our numeric features, distributions across our two
classes will be plotted as stacked density charts.

We will perform hyperparameter optimization, and then fit the best model
on our train data before evaluating the model on our test set. At this
point we will assess our final model performance using some combination
of recall, precision, roc auc, and average precision. We will present a
confusion matrix corresponding to our test results as a table in the
final report. Finally, we will present a table showing the features most
positively correlated with a high churn risk.

# Limitations & Future

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-docopt" class="csl-entry">

de Jonge, Edwin. 2020. *Docopt: Command-Line Interface Specification
Language*. <https://CRAN.R-project.org/package=docopt>.

</div>

<div id="ref-2020NumPy-Array" class="csl-entry">

Harris, Charles R., K. Jarrod Millman, Stéfan J van der Walt, Ralf
Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, et al. 2020.
“Array Programming with NumPy.” *Nature* 585: 357–62.
<https://doi.org/10.1038/s41586-020-2649-2>.

</div>

<div id="ref-hunter2007matplotlib" class="csl-entry">

Hunter, John D. 2007. “Matplotlib: A 2d Graphics Environment.”
*Computing in Science & Engineering* 9 (3): 90–95.

</div>

<div id="ref-scikit-learn" class="csl-entry">

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–30.

</div>

<div id="ref-pustokhina2021multi" class="csl-entry">

Pustokhina, Irina V, Denis A Pustokhin, Phong Thanh Nguyen, Mohamed
Elhoseny, and K Shankar. 2021. “Multi-Objective Rain Optimization
Algorithm with WELM Model for Customer Churn Prediction in
Telecommunication Sector.” *Complex & Intelligent Systems*, 1–13.

</div>

<div id="ref-team_2020" class="csl-entry">

Team, OmniSci. 2020. “Strategies for Reducing Churn Rate in the Telecom
Industry.” *RSS*.
<https://www.omnisci.com/blog/strategies-for-reducing-churn-rate-in-the-telecom-industry>.

</div>

<div id="ref-reback2020pandas" class="csl-entry">

team, The pandas development. 2020. *Pandas-Dev/Pandas: Pandas* (version
latest). Zenodo. <https://doi.org/10.5281/zenodo.3509134>.

</div>

<div id="ref-Python" class="csl-entry">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-vanderplas2018altair" class="csl-entry">

VanderPlas, Jacob, Brian Granger, Jeffrey Heer, Dominik Moritz, Kanit
Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben
Welsh, and Scott Sievert. 2018. “Altair: Interactive Statistical
Visualizations for Python.” *Journal of Open Source Software* 3 (32):
1057.

</div>

<div id="ref-tidyverse" class="csl-entry">

Wickham, Hadley. 2017. *Tidyverse: Easily Install and Load the
’Tidyverse’*. <https://CRAN.R-project.org/package=tidyverse>.

</div>

<div id="ref-knitr" class="csl-entry">

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research
in R.” In *Implementing Reproducible Computational Research*, edited by
Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman;
Hall/CRC. <http://www.crcpress.com/product/isbn/9781466561595>.

</div>

</div>
