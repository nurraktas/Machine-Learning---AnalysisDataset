import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
# Veri okuma
df = pd.read_csv("02_MY2021_Fuel_Consumption_Ratings.csv")

# Eksik veri kontrolü
print(df.isnull().sum())
df = df.fillna(0)  # Gerekirse eksik veriyi doldur

# Feature ve target seçimi
x = df[
    [
        "Engine_Size",
        "Cylinders",
        "Fuel_Consumption_city",
        "Fuel_Consumption_Hwy",
        "Fuel_Consumption_Comb"
    ]
]
y = df["Fuel_Consumption_Comb_MPG"]  # Hata buradaydı

# Scaler olmadan Linear Regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("r2 (no scaling):", r2_score(y_test, y_pred))

# Scaler ile Linear Regression
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model_s = LinearRegression()
model_s.fit(x_train_s, y_train_s)

y_pred_s = model_s.predict(x_test_s)
print("r2 (scaled):", r2_score(y_test_s, y_pred_s))


# katsayilari datframe olarak alma

coefficients = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient":model.coef_[0]
})

#renkleri belirleyerek pozitif ve negatif
colors =['blue' if c > 0 else 'red' for c in coefficients["Coefficient"] ]

#en buyuk mutlak katsayiyi bul 
max_idx= np.argmax(np.abs(coefficients["Coefficient"]))

color = ['dodgerblue' if i == max_idx and coefficients["Coefficient"] [i]>0
          else 'lightcoral' if i==max_idx and coefficients["Coefficient"][i]<0
          else 'lightblue' if coefficients["Coefficient"][i]>0
          else 'pink'
          for i in range(len(coefficients)) ]


plt.figure(figsize=(8,5))
plt.bar(coefficients["Feature"], coefficients["Coefficient"], color=colors)
plt.title("Linear Regression Feature Importance")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.show()
