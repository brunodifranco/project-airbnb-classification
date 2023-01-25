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
Airbnb is an online marketplace for short-term homestays, and their business model consists of charging a comission from each booking. So they can better understand their customers behaviors and most desired booking locations a Data Scientist was hired, in order to <b> predict the five most likely countries for a USA user to make their next booking</b>. Airbnb provided data from over 200 thousand users, split in two different datasets (more information in <a href="https://github.com/brunodifranco/project-airbnb-classification#2-data-overview">Section 2</a>), so the predictions could be made for around 61 thousand users. There are 12 possible outcomes of the destination country: 'USA', 'France', 'Canada', 'Great Britain', 'Spain', 'Italy', 'Portugal', 'New Zealand', 'Germany' and 'Australia', as well as 'NDF' (which means there wasn't a booking) and 'other countries'. </p>

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
<p align="justify"> To predict the five most likely countries for a USA user to make their next booking the following steps were performed: </p>

- <b> Understanding the Business Problem</b>: Understanding the main objective we are trying to achieve and plan the solution to it. 

- <b> Collecting Data</b>: Collecting data from Kaggle.

- <p align="justify"> <b> Data Cleaning</b>: Checking data types and Nan's. Other tasks such as: renaming columns, dealing with outliers, fixing missing values, changing data types, etc.</p>

- <p align="justify"> <b> Feature Engineering</b>: Creating new features from the original ones, so that those could be used in the ML model. The full new features created with their definitions are available <a href="https://github.com/brunodifranco/project-airbnb-classification/blob/main/new_features.md">here.</a> </p>

- <p align="justify"> <b> Exploratory Data Analysis (EDA)</b>: Exploring the data in order to obtain business experience, look for data inconsistencies, useful business insights and find important features for the ML model. This process is split in Univariate, Bivariate (Checking Hypotheses) and Multivariate Analysis. The univariate analysis was done by using the <a href="https://pypi.org/project/pandas-profiling/">Pandas Profiling</a> library. The profile report is available for download <a href="https://github.com/brunodifranco/project-outleto-clustering/tree/main/pandas-profiling-reports"> here</a>. The top business insights found are available at <a href="https://github.com/brunodifranco/project-airbnb-classification#5-top-business-insights"> Section 5</a>. </p>

```diff
- link do profile report
```

- <b> Data Preparation</b>: Applying <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/"> Rescaling Techniques</a> in the data, as well as <a href="https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/">Enconding Methods</a>, to deal with categorical variables. 

- <b> Feature Selection</b>: Selecting the best features to use in the ML model by using <a href="https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f"> Random Forest</a>. 

- <p align="justify"> <b> Machine Learning Modeling and Model Evaluation</b>: Training Classification Algorithms. The best model was selected to be improved via Bayesian Optimization with Optuna. More information in <a href="https://github.com/brunodifranco/project-airbnb-classification#6-machine-learning-models">Section 6</a>.</p>


```diff
! Revision - Elaborar melhor Model Deployment and Result
```

- <p align="justify"> <b> Model Deployment and Results </b>: Providing a list of the five most likely destinations predictions for 61 thousand USA Airbnb users, as well as graphical analysis of the predictions by age, gender and overall analysis. This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere in a Streamlit App. In addition to that, if new data from new users comes in it's easy to get new predictions, as a Flask application using Render Cloud was built. More information in <a href="https://github.com/brunodifranco/project-airbnb-classification#7-model-deployment-and-results"> Section 7</a>.</p>
  
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

<p align="justify"> Initially, seven models were trained using cross-validation, so we can provide predictions on the five most likely countries for an US Airbnb user to book their next destinations: Logistic Regression, Decision Tree, Random Forest, Extra Trees, AdaBoost, XGBoost and Light GBM.</p>

The initial cross validation performance for all seven algorithms are displayed below:

<div align="center">

|         **Model**           |   **NDCG at K**    |
|:---------------------------:|:------------------:|
|    Light GBM                | 0.8496 +/- 0.0006  |
|    XGBoost                  |	0.8482 +/- 0.0004	 |
|    Random Forest            | 0.8451 +/- 0.0006	 |
|    AdaBoost                 | 0.8429 +/- 0.0019  |
|    Extra Trees              | 0.8390 +/- 0.0008  |
|    Logistic Regression      | 0.8377 +/- 0.0010  |
|    Decision Tree            | 0.7242 +/- 0.0023  |
 
</div>

<i>Where K is equal to 5, given our business problem</i>.

<p align="justify"> The <b>Light GBM</b> was chosen as a final model, since it's fast to train and tune, whilst being also the one with the best result without any tuning. In addition to that, it's much better for deployment, as it's much lighter than a XGBoost or Random Forest for instance, especially given the fact that we're using a free deployment cloud. More information in Section XXXXXXXXXXXXXXXXXX.</p>

```diff
! Revision - Section XXXXXXXXXXXXXXXXXX
```

<p align="justify"> Instead of using cross-validation, which uses only the training dataset, we tuned the model's hyperparameters by comparing its performance on the test dataset, which was split before Data Preparation, to avoid <a href="https://towardsdatascience.com/data-leakage-in-machine-learning-how-it-can-be-detected-and-minimize-the-risk-8ef4e3a97562">Data Leakage</a>. After tuning LGBM's hyperparameters using <a href="https://optuna.readthedocs.io/en/stable/index.html">Bayesian Optimization with Optuna</a> the model performance has improved, as expected:</p>

<div align="center">
<table>
<tr><th>Before Tuning </th><th>Final Model</th></tr>
<tr><td>

|         **Model**        | **NDCG at K** |
|:------------------------:|:-------------:|
|        Light GBM         |     0.8514    |

</td><td>
 
|         **Model**        | **NDCG at K** |
|:------------------------:|:-------------:|
|        Light GBM         |     0.8542    | 

</td></tr> </table>

</div>

## <i>Metrics Definition and Interpretation</i>

<p align="justify"> As the goal in this project is to predict not only the most likely next booking destination for each user, but the five most likely ones the <b>Normalized discounted cumulative gain (NDCG) at rank K</b> was chosen.</p>
 
<p align="justify"> NDCG at K <i>“measures the performance of a recommendation system based on the graded relevance of the recommended entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal ranking of the entities.”</i> Therefore, for this instance (where k equals 5), <b>it not only measures how well we can predict the five most likely next booking locations for each user, but also how well can rank them from the most likely to the least</b>.</p>

# 7. **Model Deployment and Results**
```diff
+ Final Revision Only - tentar aumentar a fonte dos links streamlit e flask
```

<p align="justify"> The model deployment was performed in three steps: 
 
- <p align="justify"> <b>Step 1</b>: The original data (both datasets in <a href="https://github.com/brunodifranco/project-airbnb-classification#2-data-overview"> Section 2</a>) was saved in a PostgreSQL Database from <a href="https://neon.tech/">Neon.tech</a>. </p>
 
 - <p align="justify"> <b>Step 2</b>: A Flask application was built using <a href="https://render.com/"> Render Cloud </a>, on which it extracts the original data from that PostgreSQL Database, cleans and transforms the data, loads the saved ML model, creates predictions for each user and adds these predictions back in a different table inside the same Database. Let's name this table 'df_pred' for the sake of the explanation.</p>

 - <p align="justify"> <b>Step 3</b>: Streamlit retrieves the df_pred data from the Database and displays it in a table inside Streamlit with filters, where you can find the five most likely destinations predictions for the 61 thousand USA Airbnb users. In addition to that, graphical analysis from the predictions were built, split by age, gender and overall analysis. This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere in a Streamlit App.</p>

<b>Click here to access the Streamlit App:</b> [![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://airbnb.streamlit.app/)

<b> And here to access the Flask App:</b> [![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://airbnb-predict.onrender.com)

<p align="justify"> The Flask App is particularly useful for when new data comes in, as we can get new predictions with a click of a button, so it can be later retrieved by the Streamlit App. The Streamlit App code is available <a href="https://github.com/brunodifranco/airbnb-app">here</a> and the Flask App code can be seen <a href="https://github.com/brunodifranco/airbnb-predict">here</a>.</p>

<p align="justify"><i>Because the deployment was made in a free cloud (Render Cloud) the Flask App's functionality could be slow, in the other hand, the main deployment product, which is the Streamlit App, should work quickly.</i></p>

# 8. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> We managed to provide a list of the five most likely destinations predictions for 61 thousand USA Airbnb users, as well as graphical analysis of the predictions by age, gender and overall analysis. This can all be found in a Streamlit App, for better visualization. Also, a Flask application was built for when new data comes in, making it possible to get new predictions easily. In addition to that, three interesting and useful insights were found through Exploratory Data Analysis (EDA), so that those can be properly used by Airbnb. </p>
 
# 9. **Next Steps**
<p align="justify"> Further on, this solution could be improved by a few strategies:
  
 - Creating even more features from the existent ones.
 - Try other classification algorithms, such as Neural Networks.
 - Using a paid Cloud, such as AWS.
 
# Contact

- brunodifranco99@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BrunoDiFrancoAlbuquerque/)
