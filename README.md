# Titanic - Survival Prediction

A machine learning solution to the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic), predicting which passengers survived the shipwreck from features like age, sex, ticket class, and family relationships.

## Results

Achieved a top-tier public leaderboard rank of **216** with a score of **0.81818**.

| Version | Approach | Public Score |
|---------|----------|--------------|
| v1 | Gradient Boosting with feature engineering | 0.76076 |
| v2 | KNN with family/group survival feature | **0.81818** |

The main improvement came from the family-survival feature: passengers travelling together tended to share the same fate, so a relative's outcome is highly predictive.

## Usage

1. Download `train.csv` and `test.csv` from the [competition data page](https://www.kaggle.com/competitions/titanic/data) into this folder.
2. Install dependencies:
