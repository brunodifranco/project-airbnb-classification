 # WORK IN PROGRESS

<h1 align="center"> Creating a Customer Ranking System for an Insurance Company</h1>

<p align="center">A Classification Project</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214205680-dccc15c4-ec86-439c-b50a-b96593c4416b.png" width="450"/>
</p>

*Obs: The business problem is fictitious, although both company and data are real.*

*The in-depth Python code explanation is available in [this](https://github.com/brunodifranco/project-insuricare-ranking/blob/main/insuricare.ipynb) Jupyter Notebook.*

# 1. **Airbnb and Business Problem**

```diff
+ Final Revision Only
```

<p align="justify"> 
Airbnb is an online marketplace for short-term homestays, and their business model consists of charging a comission from each booking. So they can better understand their customers behaviors and most desired booking locations a Data Scientist was hired, in order to <b> predict the five most likely countries for a USA user to make their next booking</b>. Airbnb provided data from over 200 thousand users, split in two different datasets (more information in <a href="https://github.com/brunodifranco/project-airbnb-classification#2-data-overview">Section 2</a>), so the predictions could be made for over 60 thousand users. There are 12 possible outcomes of the destination country: 'USA', 'France', 'Canada', 'Great Britain', 'Spain', 'Italy', 'Portugal', 'New Zealand', 'Germany' and 'Australia', as well as 'NDF' (which means there wasn't a booking) and 'other countries'. </p>

# 2. **Data Overview**

```diff
! Revision
```

The data was split in users and sessions data, which is the internet browsing information. The Initial features descriptions are available below:

<div align="center">
 
## Users Data:

| **Feature**          | **Definition** |
|----------------------|----------------|
|       id      | user id |
|       date_account_created      | the date of account creation | 
|       timestamp_first_active    | timestamp of the first activity |
|       date_first_booking       | date of first booking |
|       gender    | user's gender |
|       age      | user's age |
|       signup_method     | method of signing up e.g. facebook, basic, google |
|       signup_flow        | the page a user came to signup up from |
|       language        | international language preference |
|       affiliate_channel        | what kind of paid marketing |
|       affiliate_provider        | where the marketing is e.g. google, craigslist, other |
|       first_affiliate_tracked        | whats the first marketing the user interacted with before the signing up |
|       signup_app        | signup app  e.g. Web, Android |
|       first_device_type        | first device type used e.g. Windows Desktop, IPhone, Android Phone|
|       first_browser        | first browser used e.g. Chrome, FireFox, Safari |
|       country_destination        | This is the target variable |

</div>  

<div align="center">
 
## Sessions Data:
 
| **Feature**          | **Definition** |
|----------------------|----------------|
|       user_id      | to be joined with the column 'id' in users table |
|       action      | action performed e.g. show, search_results, confirm_email | 
|       action_type    | action type performed e.g. view, click |
|       action_detail       | action detail e.g. confirm_email_link, view_search_results |
|       device_type    | device used on each action |
|       secs_elapsed      | the time between two actions recorded |
  
 </div>  
  
<i>The data was collected from Kaggle.</i>

# 3. **Business Assumptions and Definitions**

```diff
- To do
```

- Cross-selling is a strategy used to sell products associated with another product already owned by the customer. In this project, health insurance and vehicle insurance are the products. 
- Learning to rank is a machine learning application. In this project, we are ranking customers in a list, from the most likely customer to buy the new insurance to the least likely one. This list will be provided by the ML model.

# 4. **Solution Plan**

## 4.1. How was the problem solved?
```diff
- To do
```
<p align="justify"> To provide an ordered list of these new customers, based on their propensity score of buying the new insurance the following steps were performed: </p>

- <b> Understanding the Business Problem</b>: Understanding the main objective Insuricare was trying to achieve and plan the solution to it. 

- <b> Collecting Data</b>: Collecting data from a PostgreSQL Database, as well as from Kaggle.

- <b> Data Cleaning</b>: Checking data types and Nan's. Other tasks such as: renaming columns, dealing with outliers, changing data types weren't necessary at this point. 

- <b> Feature Engineering</b>: Editing original features, so that those could be used in the ML model. 

- <p align="justify"> <b> Exploratory Data Analysis (EDA)</b>: Exploring the data in order to obtain business experience, look for useful business insights and find important features for the ML model. The top business insights found are available at <a href="https://github.com/brunodifranco/project-insuricare-ranking#5-top-business-insights"> Section 5</a>. </p>

- <b> Data Preparation</b>: Applying <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/"> Rescaling Techniques</a> in the data, as well as <a href="https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/">Enconding Methods</a>, to deal with categorical variables. 

- <b> Feature Selection</b>: Selecting the best features to use in the ML model by using <a href="https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f"> Random Forest</a>. 

- <p align="justify"> <b> Machine Learning Modeling and Model Evaluation</b>: Training Classification Algorithms with cross-validation. The best model was selected to be improved via Bayesian Optimization with Optuna. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#6-machine-learning-models">Section 6</a>. </p>

- <p align="justify"> <b> Model Deployment and Results</b>: Evaluating the model using two metrics: Precision at K and Recall at K, as well as two curves: Cumulative Gains and Lift Curves. </p>

- <p align="justify"> <b> Propensity Score List and Model Deployment </b>: Providing a full list of the 76 thousand customers sorted by propensity score, as well as a Google Sheets that returns propensity score and ranks customers (used for future customers). This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere. More information in <a href="https://github.com/brunodifranco/project-insuricare-ranking#7-business-and-financial-results"> Section 7</a>. </p>
  
## 4.2. Tools and techniques used:

```diff
+ Final Revision Only
```

- [Python 3.10.9](https://www.python.org/downloads/release/python-3109/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Sklearn](https://scikit-learn.org/stable/).
- [SQL](https://www.w3schools.com/sql/) and [PostgresSQL](https://www.postgresql.org/).
- [Jupyter Notebook](https://jupyter.org/) and [VSCode](https://code.visualstudio.com/).
- [Flask](https://flask.palletsprojects.com/en/2.2.x/) and [Python API's](https://realpython.com/api-integration-in-python/).  
- [Streamlit](https://streamlit.io/) and [Render Cloud](https://render.com/)
- [Git](https://git-scm.com/) and [Github](https://github.com/).
- [Exploratory Data Analysis (EDA)](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15). 
- [Techniques for Feature Selection](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/).
- [Classification Algorithms (Logistic Regression, Decision Tree, Random Forest, ExtraTrees, AdaBoost, XGBoost and LGBM Classifiers)](https://scikit-learn.org/stable/modules/ensemble.html).
- [Cross-Validation Methods](https://scikit-learn.org/stable/modules/cross_validation.html), [Bayesian Optimization with Optuna](https://optuna.readthedocs.io/en/stable/index.html) and [Performance Metrics - NDCG at rank K](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings/overview/evaluation).

# 5. **Top Business Insights**

```diff
+ Final Revision Only
```

 - ### 1st - Users take less than 2 days, on average, from first active in the platform to creating an account, considering all destinations.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318780-33ad6f3a-3054-4c4b-90c3-c7db9d9edd85.png" alt="drawing" width="850"/>
</p>

--- 

- ### 2nd - The number of accounts created goes up during the spring.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318784-96eb4214-86dc-4b9e-a449-412b980a1630.png" alt="drawing" width="850"/>
</p>

--- 

- ### 3rd - Women made over 15% more bookings for countries other than USA, in comparison to men.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318787-7ea79725-8b3c-4909-b9dd-722b4d74b44e.png" alt="drawing" width="850"/>
</p>

---

# 6. **Machine Learning Models**

<p align="justify"> Initially, seven models were trained using cross-validation, so we can provide predictions on the five most likely countries for an US Airbnb user to book their next destinations: </p>

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- ExtraTrees Classifier
- AdaBoost Classifier
- XGBoost Classifier 
- Light GBM Classifier

The initial cross validation performance for all seven algorithms are displayed below:

<div align="center">

|         **Model**           | **NDCG at K** |
|:---------------------------:|:------------------:|
|    LGBM Classifier          | 0.8496 +/- 0.0006  |
|    XGBoost Classifier       |	0.8482 +/- 0.0004	|
|    Random Forest Classifier | 0.8451 +/- 0.0006	|
|    AdaBoost Classifier      | 0.8429 +/- 0.0019  |
|    ExtraTrees Classifier    | 0.8390 +/- 0.0008  |
|    Logistic Regression      | 0.8377 +/- 0.001  |
|    Decision Tree Classifier | 0.7242 +/- 0.0023 |
</div>

<i>Where K is equal to 5, given our business problem. </i>

<p align="justify"> The <b>Light GBM Classifier</b> model was chosen for hyperparameter tuning, since it's fast to train and tune, whilst being also the one with the best result without any tuning. In addition to that, it's much better for deployment, as it's much lighter than a XGBoost or Random Forest for instance, especially given the fact that we're using a free deployment cloud. More information in Section XXXXXXXXXXXXXXXXXX.</p>

<p align="justify"> Instead of using cross-validation, which uses only the training dataset, we tuned the model by comparing its performance on the test dataset, which was split before Data Preparation, to avoid <a href="https://towardsdatascience.com/data-leakage-in-machine-learning-how-it-can-be-detected-and-minimize-the-risk-8ef4e3a97562">Data Leakage</a>. After tuning LGBM's hyperparameters using  <a href="https://optuna.readthedocs.io/en/stable/index.html">Bayesian Optimization with Optuna</a> the model performance has improved, as expected:</p>

<div align="center">
<table>
<tr><th>Before Tuning </th><th>Final Model</th></tr>
<tr><td>

|         **Model**        | **NDCG at K** |
|:--:|:--:|
|LGBM Classifier | 0.8514|

</td><td>
 
|         **Model**        | **NDCG at K** |
|:--:|:--:|
|LGBM Classifier | 0.8542|

</td></tr> </table>

</div>

## <i>Metrics Definition and Interpretation</i>

<p align="justify"> <i> As we're ranking customers in a list, there's no need to look into the more traditional classification metrics, such as accuracy, precision, recall, f1-score, aoc-roc curve, confusion matrix, etc.

Instead, **ranking metrics** will be used:

- **Precision at K** : Shows the fraction of correct predictions made until K out of all predictions. 
  
- **Recall at K** : Shows the fraction of correct predictions made until K out of all true examples. 

In addition, two curves can be plotted: 

- <b>Cumulative Gains Curve</b>, indicating the percentage of customers, ordered by probability score, containing a percentage of all customers interested in the new insurance. 

- <b>Lift Curve</b>, which indicates how many times the ML model is better than the baseline model (original model used by Insuricare). </i> </p>

# 7. **Business and Financial Results**

## 7.1. Business Results

**1) By making 20,000 calls how many interested customers can Insuricare reach with the new model?**
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152035-48c27ead-53f8-440e-af92-f049456dac33.png" alt="drawing" width="1000"/>
</p>

<p align="justify"> 

- 20,000 calls represents 26.24% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 71.29% of customers interested in the new vehicle insurance, since 0.7129 is our recall at 20,000. </p>

- As seen from the Lift Curve, our **LGBM model is 2.72 times better than the baseline model at 20,000 calls.** 

**2) Now increasing the amount of calls to 40,000 how many interested customers can Insuricare reach with the new model?**

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/198152040-929e3f17-d07e-401a-892c-50bf9c01f475.png" alt="drawing" width="1000"/>
</p>

- 40,000 calls represents 52.48% of our database. So if the sales team were to make all these calls Insuricare would be able to contact 99.48% of customers interested in the new vehicle insurance, since 0.9948 is our recall at 40,000.

- At 40,000 calls, our **LGBM model is around 1.89 times better than the baseline model.**  

## 7.2. Expected Financial Results

To explore the expected financial results of our model, let's consider a few assumptions:

- The customer database that will be reached out is composed of 76,222 clients.
- We expect 12.28% of these customers to be interested in the new vehicle insurance, since it's the percentage of interest people that participated in the Insuricare research. 
- The annual premium for each of these new vehicle insurance customers will be US$ 2,630 yearly. *

*<i> The annual premium of US$ 2,630 is set for realistic purposes, since it's the lowest and most common value in the dataset. </i>

The expected financial results and comparisons are shown below:

<div align="center">

|    **Model**    |  **Annual Revenue - 20,000 calls** | **Annual Revenue - 40,000 calls** |  **Interested clients reached out - 20,000 calls** | **Interested clients reached out - 40,000 calls** |
|:---------------:|:---:|:-----------------------------------:|:---:|:---------------------------------------:|
|       LGBM      | US$ 17,515,800.00    |US$ 24,440,590.00          | 6660   |9293                  |
|     Baseline    |  US$ 6,446,130.00    |US$ 12,894,890.00           | 2451  |4903                  |
| $\Delta$ (LGBM, Baseline) |  11,069,670.00     |US$ 11,545,700.00         |  4209   |   4390                  |

</div>

<i> $\Delta$ (LGBM, Baseline) is the difference between models. </i>

As seen above the LGBM model can provide much better results in comparison to the baseline model, with an annual financial result around 172% better for 20,000 calls and 89% better for 40,000 calls, which is exactly what was shown in the Lift Curve. 

# 8. **Propensity Score List and Model Deployment**

<p align="justify"> The full list sorted by propensity score is available for download <a href="https://github.com/brunodifranco/project-insuricare-ranking/blob/main/insuricare_list.xlsx">here</a>. However, for other new future customers it was necessary to deploy the model. In this project Google Sheets and Render Cloud were chosen for that matter. The idea behind this is to facilitate the predictions access for any new given data, as those can be checked from anywhere and from any electronic device, as long as internet connection is available. The spreadsheet will return you the sorted propensity score for each client in the requested dataset, all you have to do is click on the "Propensity Score" button, then on "Get Prediction".

<b> Click here to access the spreadsheet </b>[![Sheets](https://www.google.com/images/about/sheets-icon.svg)](https://docs.google.com/spreadsheets/d/1K2tJP6mVJwux4qret1Dde9gQ23KsDRGRl8eJbsigwic/edit?usp=sharing)

<i> Because the deployment was made in a free cloud (Render) it could take a few minutes for the spreadsheet to provide a response, <b> in the first request. </b> In the following requests it should respond instantly. </i>

</p>

# 9. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> We managed to provide a list of new customers ordered by their buy propensity score and a spreadsheet that returns the buy propensity score for other new future customers. Now, the Sales Team can focus their attention on the 20,000 or 40,000 first customers on the list, and in the future focus on the top K customers of the new list. </b> In addition to that, five interesting and useful insights were found through Exploratory Data Analysis (EDA), so that those can be properly used by Insuricare, as well as Expected Financial Results. </p>
 
# 10. **Next Steps**
<p align="justify"> Further on, this solution could be improved by a few strategies:
  
 - Conducting more market researches, so that more useful information on customers could be collected, since there was a lack of meaningful variables.
  
 - Applying <a href="https://builtin.com/data-science/step-step-explanation-principal-component-analysis">Principal Component Analysis (PCA) </a> in the dataset.
  
 - Try other classification algorithms that could better capture the phenomenon.

# Contact

- brunodifranco99@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BrunoDiFrancoAlbuquerque/)
