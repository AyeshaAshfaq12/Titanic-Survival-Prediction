"""
Titanic v2 - adds the family/group survival feature, the single biggest
driver for beating the gender baseline on this competition.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["PassengerId"].copy()
data = pd.concat([train, test], sort=False).reset_index(drop=True)

# ---------- Family / group survival feature ----------
data["Last_Name"] = data["Name"].apply(lambda x: x.split(",")[0].strip())
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

DEFAULT = 0.5
data["Family_Survival"] = DEFAULT

# Group people who share a surname AND fare (very likely the same family)
for _, grp in data.groupby(["Last_Name", "Fare"]):
    if len(grp) != 1:
        for ind, row in grp.iterrows():
            others = grp.drop(ind)["Survived"]
            smax, smin = others.max(), others.min()
            pid = row["PassengerId"]
            if smax == 1.0:
                data.loc[data["PassengerId"] == pid, "Family_Survival"] = 1
            elif smin == 0.0:
                data.loc[data["PassengerId"] == pid, "Family_Survival"] = 0

# Fill remaining gaps using people who share a ticket (travel companions)
for _, grp in data.groupby("Ticket"):
    if len(grp) != 1:
        for ind, row in grp.iterrows():
            if row["Family_Survival"] in (0, 0.5):
                others = grp.drop(ind)["Survived"]
                smax, smin = others.max(), others.min()
                pid = row["PassengerId"]
                if smax == 1.0:
                    data.loc[data["PassengerId"] == pid, "Family_Survival"] = 1
                elif smin == 0.0:
                    data.loc[data["PassengerId"] == pid, "Family_Survival"] = 0

known = (data["Family_Survival"] != 0.5).sum()
print(f"Passengers with a known family/group survival signal: {known} of {len(data)}")

# ---------- Other features ----------
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# Fare bins
data["FareBin"] = pd.qcut(data["Fare"], 5, duplicates="drop")
data["FareBin_Code"] = LabelEncoder().fit_transform(data["FareBin"])

# Age filled by title median, then binned
data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
data["Title"] = data["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
data["Age"] = data["Age"].fillna(data.groupby("Title")["Age"].transform("median"))
data["Age"] = data["Age"].fillna(data["Age"].median())
data["AgeBin"] = pd.qcut(data["Age"], 4, duplicates="drop")
data["AgeBin_Code"] = LabelEncoder().fit_transform(data["AgeBin"])

# Sex
data["Sex"] = LabelEncoder().fit_transform(data["Sex"])  # female=0, male=1

features = ["Pclass", "Sex", "AgeBin_Code", "FareBin_Code", "Family_Survival", "FamilySize"]

X_all = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

X = X_scaled[:891]
y = train["Survived"].astype(int)
X_test = X_scaled[891:]

# ---------- Tune a KNN classifier ----------
grid = {"n_neighbors": range(4, 20), "weights": ["uniform", "distance"], "p": [1, 2]}
gs = GridSearchCV(KNeighborsClassifier(), grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X, y)
print("Best params:", gs.best_params_)
print(f"Local CV accuracy: {gs.best_score_:.4f}  (note: mildly optimistic due to the group feature)")

preds = gs.best_estimator_.predict(X_test).astype(int)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
submission.to_csv("submission_v2.csv", index=False)
print("submission_v2.csv written:", submission.shape[0], "rows")
print("Predicted survival rate:", round(preds.mean(), 3))
