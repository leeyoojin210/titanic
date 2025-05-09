import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# 데이터 로드
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

# 기본 데이터 정보 확인
print("Train Data Shape:", train_data.shape)
print("\nTrain Data Info:")
print(train_data.info())
print("\nTrain Data Description:")
print(train_data.describe())

# 결측치 확인
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())

# 생존자 비율 시각화
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Survived')
plt.title('Survival Count')
plt.savefig('survival_count.png')
plt.close()

# 성별에 따른 생존율
plt.figure(figsize=(8, 6))
sns.barplot(data=train_data, x='Sex', y='Survived')
plt.title('Survival Rate by Gender')
plt.savefig('survival_by_gender.png')
plt.close()

# 데이터 전처리
def preprocess_data(df):
    # 데이터프레임 복사
    df = df.copy()
    
    # 성별을 숫자로 변환
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    
    # 결측치 처리
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 새로운 특성 생성
    # 가족 크기
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 승객 등급과 성별 조합
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex'].astype(str)
    
    # 나이 구간화 (중복 제거 옵션 추가)
    df['AgeBin'] = pd.qcut(df['Age'], 5, labels=False, duplicates='drop')
    
    # 요금 구간화
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False, duplicates='drop')
    
    # 이름에서 호칭 추출
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
        'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2,
        'Countess': 3, 'Ms': 2, 'Lady': 3, 'Jonkheer': 1,
        'Don': 1, 'Mme': 3, 'Capt': 5, 'Sir': 5
    }
    df['Title'] = df['Title'].map(title_mapping)
    
    # Embarked를 원-핫 인코딩
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    
    # Pclass_Sex를 원-핫 인코딩
    pclass_sex_dummies = pd.get_dummies(df['Pclass_Sex'], prefix='Pclass_Sex')
    df = pd.concat([df, pclass_sex_dummies], axis=1)
    
    # 사용할 특성 선택
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
        'FamilySize', 'Title', 'AgeBin', 'FareBin',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ] + [col for col in df.columns if col.startswith('Pclass_Sex_')]
    
    return df[features]

# 데이터 전처리 적용
X = preprocess_data(train_data)
y = train_data['Survived']

# 학습/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
print("\nRandom Forest Best Parameters:", rf_grid_search.best_params_)
print("Random Forest Best Score:", rf_grid_search.best_score_)

# 2. XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
print("\nXGBoost Best Parameters:", xgb_grid_search.best_params_)
print("XGBoost Best Score:", xgb_grid_search.best_score_)

# 3. LightGBM
lgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63],
    'subsample': [0.8, 1.0],
    'min_child_samples': [20, 50],
    'min_child_weight': [0.001, 0.01],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}

lgb_model = lgb.LGBMClassifier(
    random_state=42,
    verbose=-1,  # 경고 메시지 숨기기
    min_gain_to_split=0.01  # 분할을 위한 최소 gain 설정
)
lgb_grid_search = GridSearchCV(lgb_model, lgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lgb_grid_search.fit(X_train, y_train)
print("\nLightGBM Best Parameters:", lgb_grid_search.best_params_)
print("LightGBM Best Score:", lgb_grid_search.best_score_)

# 앙상블 모델 생성
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_grid_search.best_estimator_),
        ('xgb', xgb_grid_search.best_estimator_),
        ('lgb', lgb_grid_search.best_estimator_)
    ],
    voting='soft'
)

# 앙상블 모델 학습
ensemble.fit(X_train, y_train)

# 각 모델의 검증 데이터 성능 평가
models = {
    'Random Forest': rf_grid_search.best_estimator_,
    'XGBoost': xgb_grid_search.best_estimator_,
    'LightGBM': lgb_grid_search.best_estimator_,
    'Ensemble': ensemble
}

print("\nValidation Results:")
for name, model in models.items():
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"{name} Accuracy: {accuracy:.4f}")

# 특성 중요도 시각화 (Random Forest 기준)
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_grid_search.best_estimator_.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 최종 예측 (앙상블 모델 사용)
test_features = preprocess_data(test_data)
test_predictions = ensemble.predict(test_features)

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file has been created!") 