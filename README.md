# Aspect-Sentiment-Classification
Alexander Kane, Jonas Bode, Alanis Dahl, Johanna Driever, Ayush Mishra
## Research Topic
We will be using the Aspect Sentiment Classification Dataset to classify the sentiment towards each aspect as Positive, Negative, Neutral. We will focus on the restaurant reviews.
## Motivation and Objectives
Sentiment analysis is useful in the context of restaurant reviews because it can help businesses understand how their customers feel about different aspects of their restaurant, such as the food, service, ambiance, and pricing. By analyzing the sentiment towards these aspects in the reviews, businesses can identify areas where they are doing well and areas where they need to improve.
It could also help potential customers to quickly get an overview over a multitude of reviews through taking a look at the aggregated metrics.
## Dataset 
The dataset contains a sentence, a polarity (positive/negative/neutral), and a term (important thing customer mentioned). Given this, we should be able to train a learning model to predict the sentiment and extract an important aspect from the text.
## Methodology
We will use the sentiment analysis methods introduced in the lecture. We can also look at what the researchers used in associated papers, as it will give us an idea on how similar tasks were accomplished.
## Expected Results
We expect for our NLP algorithm to properly predict the sentiment for unseen examples more than 75% of the time, but higher results would be nice. We also expect to extract the correct term 75% of the time.
## Evaluation Metrics
We can use accuracy, precision, recall, and F1 score to evaluate the accuracy of the positivity of the review as well as whether the term is correct. We can also use a confusion matrix, ROC curve, or AUC-ROC.
## Challenges and Limitations
- Limited Sample Size: The dataset of restaurant reviews may be too small to provide a representative sample of customer sentiment toward the restaurant. This would be difficult to address without further research.
- Context: The meaning of a sentence can be greatly affected by the context in which it is used. For example, a statement that might be considered negative in one context may be positive in another. This would best be addressed by using more advanced NLP models.
- Bias: The terms and polarity set may be biased, given that the real sentiment is actually unknown.
- Term selection: A sentence may have multiple terms of varying importance. We will attempt to tune it for the one term which is most important (according to the training data). We can also exclude sentences which have more than one term.
## Task Assignment
We have several tasks which will be assigned to group members:
- Data Preprocessing: The data is already formatted and cleaned in a JSON file, but some additional processing may be needed to suit our needs.
- Model Selection: The next task is to select an appropriate machine learning model for sentiment analysis. There are several models that can be used for sentiment analysis, some of which have already been introduced in the lecture.
- Model Training: Once the model has been selected, it needs to be trained using the annotated data. The training data is used to teach the model to accurately classify sentiments and identify important terms.
- Model Evaluation: After training the model, it is important to evaluate its performance using a separate test dataset. This can be done using the evaluation metrics discussed earlier, such as accuracy, precision, recall, and F1 score.
- Model Improvement: If the model's performance is not satisfactory, it may need to be improved by adjusting the model parameters, selecting different features, or using more advanced techniques.
