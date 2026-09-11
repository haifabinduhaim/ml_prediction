from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

# ==========================================
# 1. تحميل البيانات من الإنترنت
# ==========================================

car_data = fetch_ucirepo(id=10)

data = car_data.data.features.copy()

print("✅ تم تحميل البيانات")
print("عدد السيارات قبل التنظيف:", len(data))


# ==========================================
# 2. تنظيف البيانات
# ==========================================

# تحويل السعر إلى رقم
data["price"] = pd.to_numeric(data["price"], errors="coerce")

# حذف السيارات التي لا يوجد لها سعر
data = data.dropna(subset=["price"])

print("عدد السيارات بعد التنظيف:", len(data))


# ==========================================
# 3. تحديد الهدف والبيانات
# ==========================================

y = data["price"]

# حذف السعر من البيانات التي سيتعلم منها النموذج
X = data.drop(columns=["price"])


# ==========================================
# 4. تحديد الأعمدة النصية والرقمية
# ==========================================

categorical_columns = X.select_dtypes(include=["object"]).columns
numeric_columns = X.select_dtypes(exclude=["object"]).columns

print("\nالأعمدة النصية:")
print(categorical_columns.tolist())

print("\nالأعمدة الرقمية:")
print(numeric_columns.tolist())


# ==========================================
# 5. تجهيز البيانات
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            "passthrough",
            numeric_columns
        )
    ]
)


# ==========================================
# 6. إنشاء نموذج ML
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 7. إنشاء Pipeline
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==========================================
# 8. تقسيم البيانات
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n✅ تم تقسيم البيانات!")
print("بيانات التدريب:", X_train.shape)
print("بيانات الاختبار:", X_test.shape)


# ==========================================
# 9. تدريب النموذج
# ==========================================

print("\n🤖 بدأ تدريب النموذج...")

pipeline.fit(X_train, y_train)

print("✅ انتهى التدريب!")


# ==========================================
# 10. اختبار النموذج
# ==========================================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\n📊 نتيجة النموذج")
print("متوسط الخطأ:", round(mae, 2))

# ==========================================
# 11. تجربة التنبؤ على سيارة من بيانات الاختبار
# ==========================================

# نأخذ أول سيارة من بيانات الاختبار
car = X_test.iloc[[0]]

# السعر الحقيقي للسيارة
real_price = y_test.iloc[0]

# النموذج يتوقع السعر
predicted_price = pipeline.predict(car)[0]

print("\n================================")
print("🚗 تجربة التنبؤ")
print("================================")

print("💰 السعر الحقيقي:", round(real_price, 2))
print("🤖 السعر المتوقع:", round(predicted_price, 2))

difference = abs(real_price - predicted_price)

print("📉 الفرق:", round(difference, 2))
# ==========================================
# 12. اختيار سيارة من البيانات والتنبؤ بسعرها
# ==========================================

print("\n================================")
print("🚗 اختر سيارة للتنبؤ بسعرها")
print("================================")

# نعرض أول 10 سيارات
for i in range(10):
    print(i, "-", X_test.iloc[i]["make"])

choice = int(input("\nاختر رقم السيارة من 0 إلى 9: "))

# السيارة التي اختارها المستخدم
selected_car = X_test.iloc[[choice]]

# السعر الحقيقي
real_price = y_test.iloc[choice]

# السعر المتوقع
predicted_price = pipeline.predict(selected_car)[0]

print("\n================================")
print("📊 النتيجة")
print("================================")

print("🚗 الشركة:", selected_car.iloc[0]["make"])
print("💰 السعر الحقيقي:", round(real_price, 2))
print("🤖 السعر المتوقع:", round(predicted_price, 2))

difference = abs(real_price - predicted_price)

print("📉 الفرق:", round(difference, 2))
# ==========================================
# 13. إدخال سيارة جديدة والتنبؤ بالسعر
# ==========================================

print("\n================================")
print("🚗 أدخل مواصفات السيارة الجديدة")
print("================================")

# نستخدم متوسطات البيانات لباقي الخصائص
new_car = X.iloc[[0]].copy()

# ==========================================
# البيانات التي يدخلها المستخدم
# ==========================================

new_car["make"] = input("الشركة (مثال: toyota): ")
new_car["fuel-type"] = input("نوع الوقود (gas/diesel): ")
new_car["aspiration"] = input("التيربو (std/turbo): ")
new_car["body-style"] = input("نوع الهيكل (sedan/hatchback/wagon): ")
new_car["drive-wheels"] = input("نظام الدفع (fwd/rwd/4wd): ")

new_car["horsepower"] = float(
    input("قوة المحرك Horsepower (مثال: 120): ")
)

new_car["engine-size"] = float(
    input("حجم المحرك Engine Size (مثال: 120): ")
)

new_car["city-mpg"] = float(
    input("استهلاك المدينة City MPG (مثال: 25): ")
)

new_car["highway-mpg"] = float(
    input("استهلاك الطريق Highway MPG (مثال: 30): ")
)


# ==========================================
# استخدام قيم افتراضية لباقي البيانات
# ==========================================

for column in numeric_columns:
    if column not in [
        "horsepower",
        "engine-size",
        "city-mpg",
        "highway-mpg"
    ]:
        new_car[column] = X[column].median()


# ==========================================
# التنبؤ
# ==========================================

predicted_price = pipeline.predict(new_car)[0]

print("\n================================")
print("🤖 نتيجة الذكاء الاصطناعي")
print("================================")

print("🚗 الشركة:", new_car.iloc[0]["make"])
print("💰 السعر المتوقع:", round(predicted_price, 2))

