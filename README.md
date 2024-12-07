Deep Learning Based Sentiment Analysis of COVID-19 Vaccination Responses from Twitter Data 


This repository contains a Jupyter Notebook implementation of sentiment analysis on text data using machine learning and deep learning techniques. The project utilizes LSTM (Long Short-Term Memory) and BiLSTM (Bidirectional LSTM) models for classifying text sentiments into Positive, Negative, or Neutral categories. We tested the methodology of the selected research paper on new datasets to evaluate its effectiveness in different contexts.

Features

	•	Data preprocessing: Text cleaning, tokenization, lemmatization.
	•	Sentiment analysis using NLTK’s SentimentIntensityAnalyzer.
	•	Deep learning models implemented with TensorFlow/Keras:
	•	LSTM Model.
	•	BiLSTM Model.
	•	Evaluation metrics:
	•	Accuracy, Precision, Recall, F1-Score.
	•	Confusion Matrix.
	•	Visualizations:
	•	Sentiment distribution bar charts.
	•	Accuracy and loss line charts for both models.

Getting Started

Prerequisites

Before running the code, ensure you have the following installed:
	•	Python 3.7 or higher
	•	Jupyter Notebook
	•	Required Python libraries:
 Or use google collab to run the notebook.

Run the Notebook:
	•	Execute the cells sequentially to preprocess data, train models, and evaluate performance.
	•	Model outputs, visualizations, and evaluation metrics will be displayed inline.

Refer to the notebook for detailed results and visualizations.

Abstract:-

This project explores sentiment analysis using advanced deep learning models, specifically Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM) networks. Sentiment analysis, a crucial task in natural language processing (NLP), involves classifying text into categories such as positive, negative, or neutral sentiments. The dataset used consists of text data with associated sentiment labels.

The analysis begins with extensive preprocessing, including text cleaning, tokenization, and lemmatization, to prepare the data for modeling. Sentiment classification is initially performed using NLTK’s Sentiment Intensity Analyzer, providing a baseline for comparison. Subsequently, deep learning models are employed to enhance performance. The LSTM model captures sequential dependencies in text, while the BiLSTM model leverages both past and future context for improved accuracy.

Key evaluation metrics, including accuracy, precision, recall, and F1-score, are used to assess model performance. Visualizations such as accuracy and loss curves, bar charts, and confusion matrices provide insights into the models’ effectiveness. The results demonstrate that the BiLSTM model outperforms the LSTM model, showcasing its ability to better understand contextual information in sequential data.

This project highlights the potential of LSTM-based architectures for sentiment analysis and their practical applications in text classification tasks.

Results:-

By initializing the sentiment polarities into 3 groups (positive, negative and neutral), the overall scenario was visualized here and our findings came out as 39.50% positive, 29.90% negative and 30.50% neutral responses. Besides, the timeline analysis shown in this research, as sentiments fluctuated over time between the mentioned timeline above. Recurrent Neural Network (RNN) oriented architectures such as LSTM and Bi-LSTM were used to assess the performance of the predictive models, with LSTM achieving an accuracy of 90.30% and Bi-LSTM achieving an accuracy of 92.00%. Other performance metrics such as Precision, Recall, F-1 score and Confusion matrix were also shown to validate our models and findings more effectively. This study will help everyone to understand public opinion on the COVID-19 vaccines and have an impact on the aim of eradicating the Coronavirus from our beautiful world.


Overall Architecture:- 



<img src = "https://github.com/soorajpoly/Machine-Learning-Programming/blob/main/Covid-19%20view.png">
