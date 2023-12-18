# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importar los datos
data = pd.read_csv("data.csv")

# Limpieza de los datos
data = data.dropna()

# División de los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(data["x"], data["y"], test_size=0.25)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluacion del modelo
score = model.score(X_test, y_test)

print(score)


# Importar las bibliotecas necesarias
import numpy as np