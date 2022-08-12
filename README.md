# capstone-airbnb


What is problem statement - why would this be helpful:
Airbnb owner- what is the easiest way to tell how my property will be rated? Should I be concerned with reviews or more worried about my listing?

Figure out best predictor of an Airbnb’s rating so that future vacationers know what to focus when they’re looking at a listing. Best thing to focus on to gauge the true quality of an Airbnb listing.

Doing classification on post descriptions versus vacationer’s descriptions to see which yields better rating predictor? For future vacationers to gauge how much stock they want to place on either side 

Look back to eda / data viz rules  for framework / ideas

EDA: (highlight means come back to after full analysis)
outliers

Then preprocessing
Perhaps dummify amenity column
Reviews per listing for each host
Collinearity - preprocessing - - - Potential for feature engineering: Accommodates and number of beds probably similar  - collinearity? Could multiply by each other
Host response time(remember this is range of 0-3 ordinal)* host response rate
Bedrooms and beds and accommodates

Overall:
Break out rating tiers - important for classification bucketing - hist

Baseline accuracy
Throw all features at it

Prepared to try
PCA
Then transfer learning with clustering


Mix of python and SQL and Tableau - python big picture and plots, SQL for quick analysis of different features and grouping, tableau for visualization in EDA stage



Grid search log regression:
max_iter
penalty: [None, ‘l2’]
Validating results
San fran
Seattle

Metrics:
Adapting binary classification metrics to my problem

Balanced data and false positives and false negatives are equally as harmful in this context.
Accuracy
Precision

Cross Val score

Still small 
Train and test model on larger sample sizes - i.e. san fran and seattle - model held up well


The F-1 Score metric is preferable when:
We have imbalanced class distribution
We’re looking for a balanced measure between precision and recall (Type I and Type II errors)
As the F-1 score is more sensitive to data distribution, it’s a suitable measure for classification problems on imbalanced datasets.

Use predict prob-a and try to create some solid visualizations for evaluation

Multi-class classification too - include amenities count, host licensed, host_verified_id, neighborhood, property type, room type. etc, (not price - didn’t find that as good predictor of rating)
