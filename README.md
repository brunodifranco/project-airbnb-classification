<h1 align="center">Predicting Next Booking Destinations for Airbnb Users</h1>

<p align="center">A Classification Project</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214205680-dccc15c4-ec86-439c-b50a-b96593c4416b.png" width="450"/>
</p>

*Obs: The business problem is fictitious, although both company and data are real.*

*The in-depth Python code explanation is available in [this](https://github.com/brunodifranco/project-airbnb-classification/blob/main/airbnb.ipynb) Jupyter Notebook.*

# 1. **Airbnb and Business Problem**

<p align="justify"> 
Airbnb is an online marketplace for short-term homestays, and their business model consists of charging a commission for each booking. So they can better understand their customers behaviors and most desired booking locations a Data Scientist was hired, in order to <b> predict the five most likely countries for a USA user to make their next booking</b>. Airbnb provided data from over 200 thousand users, split in two different datasets (more information in <a href="https://github.com/brunodifranco/project-airbnb-classification#2-data-overview">Section 2</a>), so the predictions could be made for around 61 thousand users. There are 12 possible outcomes of the destination country: 'USA', 'France', 'Canada', 'Great Britain', 'Spain', 'Italy', 'Portugal', 'New Zealand', 'Germany' and 'Australia', as well as 'NDF' (which means there wasn't a booking) and 'other countries'. </p>

# 2. **Data Overview**

The data was split in users and sessions data, which is the internet browsing information. The Initial features descriptions are available below:

<div align="center">
<table>
<tr><th><h3>Users</h3> </th><th><h3>Sessions</h3></th></tr>
<tr><td>
 
| **Feature**    | **Definition** |
|----------------------|----------------|
|       <h5>id</h5>                | <h5>user id</h5>  |
|       <h5>date_account_created</h5>      | <h5>the date of account creation</h5>  | 
|       <h5>timestamp_first_active</h5>    | <h5>timestamp of the first activity</h5> |
|       <h5>date_first_booking</h5>        | <h5>date of first booking</h5> |
|       <h5>gender</h5>                    | <h5>user's gender</h5> |
|       <h5>age</h5>                       | <h5>user's age</h5> |
|       <h5>signup_method</h5>             | <h5>method of signing up e.g. facebook,  google</h5>|
|       <h5>signup_flow</h5>               | <h5>the page a user came to signup up from</h5> |
|       <h5>language</h5>                  | <h5>international language preference</h5> |
|       <h5>affiliate_channel</h5>         | <h5>what kind of paid marketing</h5> |
|       <h5>affiliate_provider</h5>        | <h5>where the marketing is e.g. google, craigslist</h5> |
|       <h5>first_affiliate_tracked</h5>   | <h5>first marketing the user interacted with</h5> |
|       <h5>signup_app</h5>                | <h5>signup app  e.g. Web, Android</h5> |
|       <h5>first_device_type</h5>         | <h5>first device type used e.g. Windows, IPhone, Android</h5>|
|       <h5>first_browser</h5>             | <h5>first browser used e.g. Chrome, FireFox, Safari</h5> |
|       <h5>country_destination</h5>       | <h5>target variable</h5> |

</td><td>
  
| **Feature**          | **Definition** |
|----------------------|----------------|
|       <h5>user_id</h5>      | <h5>same as 'id' in users table</h5> |
|       <h5>action</h5>       | <h5>action performed e.g. show, search_results</h5> | 
|       <h5>action_type</h5>  | <h5>action type performed e.g. view, click</h5> |
|       <h5>action_detail</h5>| <h5>action detail e.g. confirm_email_link</h5> |
|       <h5>device_type</h5>  | <h5>device used on each action</h5> |
|       <h5>secs_elapsed</h5> | <h5>the time between two actions recorded</h5> |

</td></tr> </table>

</div>

<i>The data was collected from Kaggle.</i>

# 3. **Assumptions**

- <p align="justify"> Out of 'action', 'action_type', 'action_detail' only '<b>action_type</b>' was kept due to their high correlation and because they seem to represent similar events. The choice for '<b>action_type</b>' is due to it having only 28 unique values, unlike 'action' and 'action_detail' that have hundreds, which made encoding easier later.</p>
- <p align="justify"> Missing values on 'first_affiliate_tracked' were replaced with 'untracked', as it would be the most logical replacement in this instance.</p>
- <p align="justify"> Missing values on 'age' were replaced with the ages median.</p>
- <p align="justify"> 'date_first_booking' was dropped since it doesn't exist in the new users dataset.</p>

# 4. **Solution Plan**

## 4.1. How was the problem solved?

<p align="justify"> To predict the five most likely countries for a USA user to make their next booking the following steps were performed: </p>

- <b> Understanding the Business Problem</b>: Understanding the main objective we are trying to achieve and plan the solution to it. 

- <b> Collecting Data</b>: Collecting data from Kaggle.

- <p align="justify"> <b> Data Cleaning</b>: Checking data types and Nan's. Other tasks such as: renaming columns, dealing with outliers, fixing missing values, changing data types, etc.</p>

- <p align="justify"> <b> Feature Engineering</b>: Creating new features from the original ones, so that those could be used in the ML model. The full new features created with their definitions are available <a href="https://github.com/brunodifranco/project-airbnb-classification/blob/main/new_features.md">here.</a> </p>

- <p align="justify"> <b> Exploratory Data Analysis (EDA)</b>: Exploring the data in order to obtain business experience, look for data inconsistencies, useful business insights and find important features for the ML model. This process is split in Univariate, Bivariate (Checking Hypotheses) and Multivariate Analysis. The univariate analysis was done by using the <a href="https://pypi.org/project/pandas-profiling/">Pandas Profiling</a> library. The report is available for download <a href="https://github.com/brunodifranco/project-airbnb-classification/blob/main/report.html"> here</a>. The top business insights found are available in <a href="https://github.com/brunodifranco/project-airbnb-classification#5-top-business-insights"> Section 5</a>. </p>

- <b> Data Preparation</b>: Applying <a href="https://www.atoti.io/articles/when-to-perform-a-feature-scaling/"> Rescaling Techniques</a> in the data, as well as <a href="https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/">Enconding Methods</a>, to deal with categorical variables. 

- <b> Feature Selection</b>: Selecting the best features to use in the ML model by using <a href="https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f"> Random Forest</a>. 

- <p align="justify"> <b> Machine Learning Modeling and Model Evaluation</b>: Training Classification Algorithms. The best model was selected to be improved via Bayesian Optimization with Optuna. More information in <a href="https://github.com/brunodifranco/project-airbnb-classification#6-machine-learning-models">Section 6</a>.</p>

- <p align="justify"> <b> Model Deployment and Results </b>: Providing a list of the five most likely destinations predictions for 61 thousand USA Airbnb users, as well as graphical analysis of the predictions by age, gender and overall analysis. This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere in a Streamlit App. In addition to that, if new data from new users comes in, it's easy to get new predictions, as a Flask application using Render Cloud was built. More information in <a href="https://github.com/brunodifranco/project-airbnb-classification#7-model-deployment-and-results"> Section 7</a>.</p>
  
## 4.2. Tools and techniques used:

- [Python 3.10.9](https://www.python.org/downloads/release/python-3109/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Sklearn](https://scikit-learn.org/stable/).
- [SQL](https://www.w3schools.com/sql/) and [PostgresSQL](https://www.postgresql.org/).
- [Jupyter Notebook](https://jupyter.org/) and [VSCode](https://code.visualstudio.com/).
- [Flask](https://flask.palletsprojects.com/en/2.2.x/) and [Render Cloud](https://render.com/).
- [Streamlit](https://streamlit.io/).
- [Git](https://git-scm.com/) and [Github](https://github.com/).
- [Exploratory Data Analysis (EDA)](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15). 
- [Techniques for Feature Selection](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/).
- [Classification Algorithms (Logistic Regression, Decision Tree, Random Forest, ExtraTrees, AdaBoost, XGBoost and LGBM Classifiers)](https://scikit-learn.org/stable/modules/ensemble.html).
- [Cross-Validation Methods](https://scikit-learn.org/stable/modules/cross_validation.html), [Bayesian Optimization with Optuna](https://optuna.readthedocs.io/en/stable/index.html) and [Performance Metrics (NDCG at rank K)](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings/overview/evaluation).

# 5. **Top Business Insights**

 - ### 1st - Users take less than 2 days, on average, from first active in the platform to creating an account, considering all destinations.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318780-33ad6f3a-3054-4c4b-90c3-c7db9d9edd85.png" alt="drawing" width="750"/>
</p>

--- 

- ### 2nd - The number of accounts created goes up during the spring.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318784-96eb4214-86dc-4b9e-a449-412b980a1630.png" alt="drawing" width="750"/>
</p>

--- 

- ### 3rd - Women made over 15% more bookings for countries other than USA, in comparison to men.
<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/214318787-7ea79725-8b3c-4909-b9dd-722b4d74b44e.png" alt="drawing" width="750"/>
</p>

---

# 6. **Machine Learning Models**

<p align="justify"> Initially, seven models were trained using cross-validation, so we can provide predictions on the five most likely countries for a US Airbnb user to book their next destinations: Logistic Regression, Decision Tree, Random Forest, Extra Trees, AdaBoost, XGBoost and Light GBM.</p>

The initial cross validation performance of all seven algorithms are displayed below:

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

<p align="justify"> The <b>Light GBM</b> was chosen as a final model, since it's fast to train and tune, whilst being also the one with the best result without any tuning. In addition to that, it's much better for deployment, as it's much lighter than a XGBoost or Random Forest for instance, especially given the fact that we're using a free deployment cloud. More information in <a href="https://github.com/brunodifranco/project-airbnb-classification#7-model-deployment-and-results"> Section 7</a>.</p>

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

<p align="justify"> The model deployment was performed in three steps: 
 
- <p align="justify"> <b>Step 1</b>: The original data (both datasets in <a href="https://github.com/brunodifranco/project-airbnb-classification#2-data-overview"> Section 2</a>) was saved in a PostgreSQL Database from <a href="https://neon.tech/">Neon.tech</a>. </p>
 
 - <p align="justify"> <b>Step 2</b>: A Flask application was built using <a href="https://render.com/"> Render Cloud </a>, on which it extracts the original data from that PostgreSQL Database, cleans and transforms the data, loads the saved ML model, creates predictions for each user and adds these predictions back in a different table in the same Database. Let's name this table 'df_pred' for the sake of the explanation.</p>

 - <p align="justify"> <b>Step 3</b>: Streamlit retrieves the df_pred data from the Database and displays it in a table inside Streamlit with filters, where you can find the five most likely destinations predictions for the 61 thousand USA Airbnb users. In addition to that, graphical analysis of the predictions were built, split by age, gender and overall analysis. This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere in a Streamlit App.</p>


<div align="center">
<table>
<tr><th>Click on the respective icon to access the link</th></tr>
<tr><td>
 
 <div align="center">

|         **Streamlit App**        | **Flask App** |
|:------------------------:|:-------------:|
|        [![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://airbnb.streamlit.app/)       |     [![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://airbnb-predict.onrender.com)  | 
</div>
 
 </td></tr> </table>
</div>

<p align="justify"> The Flask App is particularly useful for when new data comes in, as we can get new predictions with a click of a button, so it can be later retrieved by the Streamlit App. The Streamlit App code is available <a href="https://github.com/brunodifranco/airbnb-app">here</a> and the Flask App code can be seen <a href="https://github.com/brunodifranco/airbnb-predict">here</a>.</p>

<p align="justify"><i>Because the deployment was made in a free cloud (Render Cloud) the Flask App's functionality could be slow, in the other hand, the main deployment product, which is the Streamlit App, should work quickly.</i></p>

# 8. **Conclusion**

In this project the main objective was accomplished:

 <p align="justify"> <b>We managed to provide a list of the five most likely destinations predictions for 61 thousand USA Airbnb users, as well as graphical analysis of the predictions by age, gender and overall analysis. This can all be found in a Streamlit App, for better visualization.</b> Also, a Flask application was built for when new data comes in, making it possible to get new predictions easily. In addition to that, three interesting and useful insights were found through Exploratory Data Analysis (EDA), so that those can be properly used by Airbnb. </p>
 
# 9. **Next Steps**

<p align="justify"> Further on, this solution could be improved by a few strategies:
  
 - Creating even more features from the existing ones.
 - Try other classification algorithms, such as Neural Networks.
 - Using a paid Cloud, such as AWS.
 
# Contact

- brunodifranco99@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BrunoDiFrancoAlbuquerque/)
