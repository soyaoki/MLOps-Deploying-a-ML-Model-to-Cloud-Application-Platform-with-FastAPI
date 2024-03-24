# Model Card - Income Prediction

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Auther : Soya AOKI
- Date : 23rd, March, 2024
- Version : 1.0.0
- Type : Binary classification
- Information : This model utilizes a Random Forest algorithm for binary classification tasks, specifically predicting income levels based on census data.
- License : MIT License

## Intended Use
- Primary intended uses : Predicting whether an individual's income exceeds $50,000 per year based on census data.
- Primary intended users : Data scientists, researchers, developers.
- Out-of-scope use cases : Any uses beyond binary classification tasks related to income prediction, such as regression or clustering.

## Training Data
- Datasets : [Census Income Dataset by UC Irvine](https://archive.ics.uci.edu/dataset/20/census+income) training data split 80%.
- Motivation : The Census Income Dataset provides comprehensive socio-economic information which can be utilized to predict income levels.
- Preprocessing : Data cleaning, feature encoding, and feature scaling were performed to prepare the data for training.

## Evaluation Data
- Datasets : [Census Income Dataset by UC Irvine](https://archive.ics.uci.edu/dataset/20/census+income) test data split 20%.


## Metrics
|Precision|Recall   |Fbeta    |
|---------|---------|---------|
|0.732    | 0.618   |0.670    |


## Ethical Considerations
- Data : The model is trained on census data which may contain sensitive personal information. Strict data privacy measures must be implemented to protect individuals' privacy rights.
- Human life : The model's predictions should not be used in contexts where they could impact individuals' access to resources or opportunities.
- Risks and harms : There is a risk of reinforcing societal biases if the model's predictions are not carefully monitored and mitigated.
- Use cases : Care should be taken to ensure that the model is used responsibly and ethically, avoiding potential discrimination or harm to individuals or groups.


## Caveats and Recommendations

- Model Limitations: The model's performance may vary when applied to populations or datasets that differ significantly from the training data. Caution should be exercised when interpreting its predictions.
- Data Quality: The accuracy of predictions heavily depends on the quality and representativeness of the training data. Regular monitoring of data quality and updates to the training dataset may be necessary to maintain model performance.
- Fairness: Fairness metrics should be regularly monitored to ensure the model does not exhibit bias against certain demographic groups. Additionally, measures should be taken to mitigate any identified biases.
- Transparency: The model's decision-making process should be transparent and explainable to users. Providing insight into feature importance and model behavior can help build trust and facilitate understanding.