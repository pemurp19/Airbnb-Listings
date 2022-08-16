# Airbnb Listings Data
------

### Airbnb Background:

3 Trends from Airbnb’s latest earnings report:
- Airbnb has over 6 million active listings. Guests continue to return to cities and cross borders. Gross nights booked to high-density urban areas in Q2 2022 accelerated from Q1 2022, and once again exceeded pre-pandemic levels
- Guests continue to stay longer on Airbnb. In Q2 2022, long-term stays of 28 days or more remained our fastest-growing category by trip length compared to 2019. Long-term stays increased nearly 25% from a year ago 
- Guest demand is driving growth of the Host community - from Q2 2019, active listings grew 23% in the same period, demonstrating how supply growth continues to meet demand.

### **Problem Statement**:

With the growth in supply of Airbnb listings, there are inevitably more properties with limited to no ratings or reviews. Vacationers tend to stray away from these properties in favor of established listings with a verifiable track record of reviews and ratings. Top rated properties fall on the top of the page when you go to look for a listing. While helpful for prospective vacationers, this disincentivizes new hosts from listing their properties due to the initial hurdle of building a promising ratings track record. The goal of this project is to find a work around to this problem.

**For listings on Airbnb that are brand new or don’t have many reviews, is it possible to predict what these places would be rated to help vacationers make more informed decisions about their stays?**


### Table of Contents (Notebook Order):
data_collection_cleaning_listings
data_collection_cleaning_reviews
reviews_eda
descriptions_eda
listings_eda
classification_nlp_reviews
classification_nlp_host_descriptions
classification_no_text
extra_cities_cleaning
regression

—------

### Datasets 

Utilized various data types from [‘Inside Airbnb’](http://insideairbnb.com/get-the-data), including text data (user reviews and host descriptions) in addition to categorical and numerical features of Airbnb listings, such as neighborhood, number of amenities, and number of bedrooms and bathrooms, to predict the listings’ ratings. 

Of course, new listings do not have reviews available; however, it made sense to conduct NLP modeling on reviews to gauge how well they predict a rating relative to the hosts’ descriptions. 

* [`boston_listings_9-21.csv`]('./data/boston_listings_9-21.csv`): Boston, MA Airbnb Listings Data 2021-2022
* [`boston_listings_12-21.csv`]('./data/boston_listings_12-21.csv'): Boston, MA Airbnb Listings Data 2021-2022
* [`boston_listings_3-22.csv`]('./data/boston_listings_3-22.csv'): Boston, MA Airbnb Listings Data 2021-2022
* [`boston_listings_6-22.csv`]('./data/boston_listings_6-22.csv'): Boston, MA Airbnb Listings Data 2021-2022
* [`seattle_listings_9-21.csv`](./data/seattle_listings_9-21.csv): Seattle, WA Airbnb Listings Data 2021-2022
* [`seattle_listings_12-21.csv`](./data/seattle_listings_12-21.csv): Seattle, WA Airbnb Listings Data 2021-2022
* [`seattle_listings_3-22.csv`](./data/seattle_listings_3-22.csv): Seattle, WA Airbnb Listings Data 2021-2022
* [`seattle_listings_6-22.csv`](./data/seattle_listings_6-22.csv): Seattle, WA Airbnb Listings Data 2021-2022
* [`sf_listings_9-21.csv`](./data/sf_listings_9-21.csv): San Francisco, CA Airbnb Listings Data 2021-2022
* [`sf_listings_12-21.csv`](./data/sf_listings_12-21.csv): San Francisco, CA Airbnb Listings Data 2021-2022
* [`sf_listings_3-22.csv`](./data/sf_listings_6-22.csv): San Francisco, CA Airbnb Listings Data 2021-2022
* [`boston_reviews.csv`](./data/boston_reviews.csv): Boston, MA Airbnb Listings Data 2021-2022

### Data Dictionary [*source*](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=1322284596).

---
# Software Requirements
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn


### **Analysis**

Text EDA:
Based on EDA - appeared to be much more uniformity in the host descriptions, relative to user reviews. Each host description is longer, meaning there’s more opportunity to pick up on the pattern and sentiment of the description and correctly classify it. Also more consistency, especially in terms of lengths with vast majority being about 1,000 characters long. 


*NLP Modeling:*
First had to set up the target variable - we had a continuous variable in Rating to work with but wanted to turn this into a multi-class classification problem. In looking at the distribution of ratings, 10,000 places were rated between 4 and 5 and only 350 below 4. Hence, it made sense to break up the classes into quartiles (1 for 0 - 4.59; 2 for 4.6 - 4.8; 3 for 4.81 - 4.95; 4 for ratings greater than 4.95).

Host Descriptions typically getting more into the amenities and features of the listing for the model to be able to distinguish. I.e. “street parking” or “fully equipped kitchen”.

Reviews: Model had a hard time distinguishing among mid level ratings - should perform much better on the ends difference between a poor tier and best tier but what about the average ones

*Reviews:*


*Descriptions:*




Multi-class Classification on listing features:
*Takeaways from EDA:*
Though price should intuitively have an impact on the rating - through EDA, I discovered no discernible relationship between the price and rating variables (correlation of 0.037). As shown below, the number of amenities offered by a listing, appeared to have a more linear relationship with the rating (correlation of 0.18). There was a lack of strong correlations to the target rating variable. The most correlated features included the number of amenities and whether the host was a superhost (binarized), yet no correlations were greater than 0.2 in magnitude.



This informed my modeling process - I first attempted to model with limited features, but was mindful of the need to potentially add dimensionality due the difficulty in modeling with such low correlations to the target variable. I added dimensionality (including nearly 130 features for the Random Forest Classifier to train on, yet this produced a significantly overfit model. I sought to reduce overfitting through principal component analysis, though this ultimately had limited impact on the model’s performance. 

Clustering was implemented with the goal of transfer learning and boosting the performance of the Random Forest Classifier. KMeans clustering on the latitudes and longitudes of listings (creating 26 clusters) resulted in a slight improvement in the models performance. 



The best model was a Random Forest classifier, which had a 79.6% accuracy and 80% precision.

---

### Findings and Recommendations

**Takeaways**
I recommend using both the host description and property features to predict rating for new listings. Vacationers can be confident in gauging the rating of new home listings, despite these listings not having reviews. Able to predict what a new Boston listing would be rated with 80% accuracy. Further, these results were validated with additional listings data from San Francisco and Seattle - on larger sample sizes (15,000 and 16,000 properties, respectively), the model posted 82% accuracy on Seattle listings and 78% for San Francisco. 

The model’s not ready for deployment yet. However, it's a promising first step and its applicability can be wide-ranging. The goal is to continue to improve and train on new city data, including international locales. 
 
A productionized model would be beneficial not only for vacationers but for hosts too, particularly those who are trying to build trust with prospective vacationers. And, the model could prove impactful for Airbnb as well. Airbnb wants more selection for vacationers - as this model would incentivize new hosts - hosts that may be on the fence about listing - to list their properties, it could bolster the number of listings in less common cities and help Airbnb continue to expand its platform offering to new cities.
