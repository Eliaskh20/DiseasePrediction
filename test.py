import numpy as np
import joblib

# تحميل النموذج و HRScaler
knn_model = joblib.load('HRknn_model.pkl')
HRScaler = joblib.load('HRScaler.pkl')

# بيانات اختبارية
test_data = np.array([[70,1,4,130,200,0,2,109]])  # استخدم البيانات المناسبة هنا

# تحجيم البيانات
scaled_test_data = HRScaler.transform(test_data)

# التنبؤ
prediction = knn_model.predict(scaled_test_data)
print("Prediction:", prediction)
