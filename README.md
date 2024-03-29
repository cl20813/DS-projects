# DS-projects


## Exercise 1: Spotify genre classification using random forest.
  1. Remove duplicated rows, columns with high-entropy features.
  2. Missing values: use mean for continous and mode for categorical features.
  3. Box cox (postive data only) or yeojohnson transformation to make features more normal.
  4. Remove extreme values or outliers.
  5. Split data into train and test.
  6. Random forest, lowest auc score is 92 for 5 labels in genre.

 -[Note Link](cl20813_SPOTIFY_GENRE.ipynb)


## Exercise 2. NLP with Python: Spam detection.

  1. Read a text-based dataset into pandas: If I have a raw data, I would have to remove redundant columns other than 'label', and 'message'.
  2. Text pre processing: 1) Remove common words, ('the','a', etc...) by using NLTK library. 2) Remove punctuation (!@#$%).
  3. Use Countervectorizer to convert text into a matrix of token counts.
  4. Exploratory Data Analysis(EDA): Spam messages tend to have lesser characters.
  5. Multinomial Naive Bayes Classifer is suitable for classification with discrete features(word counts), note that it requires **integer** feature counts.
  6. We can examine type 1 error and type 2 error and AUC score.

 -[Note Link](NLP_exercise_scam_detector/NLP_exercise_scam_detector.ipynb)

