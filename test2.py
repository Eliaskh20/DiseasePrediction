import numpy as np
import joblib
import pickle

# تحميل النموذج و HRScaler
knn_model = joblib.load('HRknn_model.pkl')
HRScaler = joblib.load('HRScaler.pkl')
Na_blood_glucose_modell= joblib.load('heart_disease_pred.sav')
with open('heart_disease_pred.sav', 'rb') as file:
    Na_blood_glucose_model2 = pickle.load(file)

# بيانات اختبارية
test_data = np.array([[70,0,3,300,500,1,2,180,1,2.0,2]])  # استخدم البيانات المناسبة هنا


# التنبؤ
prediction = Na_blood_glucose_model2.predict(test_data)
print("Prediction:", prediction)
