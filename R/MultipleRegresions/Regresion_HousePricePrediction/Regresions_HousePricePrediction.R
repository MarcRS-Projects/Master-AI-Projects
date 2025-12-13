# ==========================
# == Adición de librerias ==
# ==========================
library(tidyverse)   # ggplot2, dplyr, readr, etc.
library(skimr)       # resumen rápido
library(janitor)     # limpieza de nombres
library(naniar)      # visualización y resumen de NA
library(corrplot)    # mapa de calor de correlaciones
library(GGally)      # ggpairs (matriz de gráficos)
library(caret)       # particionado y utilidades
library(recipes)     # preprocesado y pipeline
library(glmnet)      # Lasso / Ridge
library(yardstick)   # métricas (RMSE, MAE)
library(Hmisc)       # utilidades avanzadas (cut2, imputación, análisis descriptivo)
library(embed)       # técnicas modernas de encoding (target, frequency, embeddings)
library(knitr)       # Simplemente para la tabla de metricas finales que se vea bonita en salida de codigo
library(DT)          # Simplemente para la tabla de metricas finales que se vea bonita en imagen



# ====================
# == Carga de datos ==
# ====================

df <- read.csv("train.csv")
view(df)



# ============================
# == Visualización de datos ==
# ============================

# Vamos a ver las dimensiones del df y el tipo de datos
df_dim <- dim(df)
df_dim    # Guardamos las dimesiones por si de caso
str(df)    # Vemos los tipos de variables
summary(df)    # Visualizamos un simple summary general

# Miramos la distribución de varaibles numericas y categoricas
num_var <- df %>% select(where(is.numeric)) %>% names()
cat_var <- df %>% select(where(~!is.numeric(.))) %>% names()
list(numericas = num_var, categoricas = cat_var)     

# Vamos a empezar por ver si tenemos valores N/A
anyNA(df)
colSums(is.na(df))[colSums(is.na(df)) > 0]    # Vemos cuales son las columnas que tienen NA


# Vamos a ver la distribución de datos de forma general rapida
skim(df)



# ==================================
# == Visualización de datos (EDA) ==
# ==================================

# ---- BOXPLOTS PARA VARIABLES NUMÉRICAS ----

# Creamos boxplots individuales para cada variable numérica
# Usamos tidy-eval moderno en ggplot2 (ya que sino tiene demasiada carga para RStudio)
for (var in num_var) {
  
  p <- ggplot(df, aes(y = .data[[var]])) +
    geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.alpha = 0.4) +
    theme_bw() +
    labs(
      title = paste("Boxplot de", var),
      y = var,
      x = ""
    )
  
  print(p)    # Sacamos el gráfico en la ventana de plots
}


# ---- BOXPLOTS PARA VARIABLES CATEGÓRICAS VS SALESPRICE ----

for (var in cat_var) {
# Ordenamos las categorías por mediana de SalePrice para visualizar mejor
  df_plot <- df %>%
    mutate(!!var := as.factor(.data[[var]])) %>%
    group_by(.data[[var]]) %>%
    mutate(med_price = median(SalePrice, na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(!!var := fct_reorder(.data[[var]], med_price))
  
  # Crear el boxplot para esa variable
  p <- ggplot(df_plot, aes(x = .data[[var]], y = SalePrice)) +
    geom_boxplot(fill = "orange", alpha = 0.7, outlier.alpha = 0.4) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, size = 7),
      plot.title = element_text(size = 14, face = "bold")
    ) +
    labs(
      title = paste("SalePrice según", var),
      x = var,
      y = "SalePrice"
    )
  
  print(p)    # Sacamos el gráfico en la ventana de plots
}


# ---- PLOTS PARA VARIABLES NUMERICAS VS SALESPRICE ----

# Excluimos SalePrice de las variables predictoras
num_var_no_target <- setdiff(num_var, "SalePrice")  

for (var in num_var_no_target) {
  
  p <- ggplot(df, aes(x = .data[[var]], y = SalePrice)) +
    geom_point(color = "purple1", alpha = 0.6) +
    theme_bw() +
    labs(
      title = paste("SalePrice vs", var),
      x = var,
      y = "SalePrice"
    )
  
  print(p)
}


# ---- MATRIZ DE CORRELACIÓN ----

# Vamos a ver de forma sencilla la matriz de correlación
num_vars <- df %>% select(where(is.numeric)) # Selecciona solo las variables numéricas
mat_cor <- cor(num_vars, use = "complete.obs") # Calcula matriz de correlación
corrplot(mat_cor, method = "circle", type = "upper", tl.cex = 0.8) # Diagrama visual

# Ya que por la gran cantidad de variables es compleja de ver vamos a hacer una tabla ordenada para ver los resultados
# de las correlaciones que mas peso tienen (tanto positivo como negativo)
mcor_list <- mat_cor %>%
  as.data.frame() %>%
  rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "cor") %>%
  filter(var1 != var2, cor > 0.4 | cor < -0.4) %>% # Sacamos solo las correlaciones fuertes y quitamos los 1.0
  arrange(desc(cor))

view(mcor_list)



# =======================
# == División de datos ==
# =======================

# Separamos las variables predictoras (X) y la variable objetivo (y) (quitamos tambien Id ya que no hace nada)
X <- df[, !(colnames(df) %in% c("SalePrice", "Id"))]
y <- df$SalePrice

cat("Dimensiones del dataset:", nrow(df), "filas,", ncol(X), "predictores\n\n")

# Vamos a hacer la división con los % típicos (60-20-20)
set.seed(12800042)    # Seed personalizada (marca propia)

# Separamos el 60% para entrenamiento
trainIndex <- createDataPartition(y, p = 0.6, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]

# Del 40% restante, dividimos en 50%-50% (validación y reserva/test)
remainingIndex <- setdiff(seq_len(nrow(df)), trainIndex)
valIndex <- createDataPartition(y[remainingIndex], p = 0.5, list = FALSE)

X_val <- X[remainingIndex[valIndex], ]
y_val <- y[remainingIndex[valIndex]]

X_test <- X[remainingIndex[-valIndex], ]  # Renombrado a "test" 
y_test <- y[remainingIndex[-valIndex]]

# Visualziamos como han quedado los datos
cat("Entrenamiento:", nrow(X_train),"/ Validación:", nrow(X_val), "/ Test:", nrow(X_test))



# ====================================
# == Modificación de datos en train ==
# ====================================

# -- Vamos a deshacernos de variables que no aporten nada --
# __________________________________________________________

# Empezamos buscando variables con un alto % de NA
porcentajes <- sapply(X_train, function(x) sum(is.na(x)) / nrow(X_train) * 100)    # Sacamos los valores de los NA en %
porcentajes <- porcentajes[porcentajes > 0]    # Solo columnas con NA
porcentajes    # Vemos en formato de % los NA que hay en cada variable
porcentajes_extremos <- porcentajes[porcentajes>93]
porcentajes_extremos    # Vamos a eliminar estas variables

# Ahora vamos a eliminar las que tengan una varianza igual o cercana a cero
# Función para detectar variables con un valor dominante (> 95%)
dominant_value <- function(X_train, threshold = 0.98) {
  sapply(X_train, function(x) {
    # Contamos la frecuencia del valor más común
    prop_max <- max(table(x)) / length(x)
    # TRUE si la proporción supera el threshold
    return(prop_max > threshold)
  })
}

# Aplicamos a todo X_train
highly_dominant <- names(X_train)[dominant_value(X_train, threshold = 0.98)]

# Nos aseguramos rapidamente de que tiene sentido nuestra elección
for (var in highly_dominant) {
  cat("Variable:", var, "\n")
  print(table(X_train[[var]], useNA = "ifany"))  # table de cada columna
  cat("\n------------------------------\n")
}

# Juntamos las listas para eliminarlas
list_to_del <- unique(c(names(porcentajes_extremos), highly_dominant))
list_to_del
# Eliminamos las columnas del train
X_train <- X_train %>% select(-all_of(list_to_del))
dim(X_train)

# -- Vamos a codificar las variables categoricas --
# __________________________________________________________

# Vamos a hacerle una codificación ordinal a las que representen orden/calidad/estado 
# Vamos a hacerle una codificación One-Hot Encoding a las que son nominales sin orden
# Vamos a hacerle una codificación por frecuencia a las restantes

# Creamos las listas de nuestras variables categoricas según el metodo que le vamos a aplicar
ordinal_vars <- c(
  "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC",
  "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PavedDrive", "Fence", "Functional"
)

nominal_vars <- c(
  "MSZoning", "LotShape", "LandContour", "LotConfig", "LandSlope", "Condition1", "BldgType", "HouseStyle", 
  "RoofStyle", "RoofMatl", "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType",
  "GarageFinish", "SaleType", "SaleCondition"
)

extra_cat_vars <- c("Neighborhood", "Exterior1st", "Exterior2nd")

# Nos sacamos de encima las variables a las que vamos a aplicar el encoding por frecuencia
# Guardamos todas las tablas de frecuencia de extra_cat_vars en una lista
freq_tables <- list()

for (var in extra_cat_vars) {
  freq_table <- X_train %>%
    group_by(.data[[var]]) %>%
    summarise(freq = n()/nrow(X_train))
  
  freq_tables[[var]] <- freq_table  # guardamos en la lista
  
  # Reemplazamos la columna original con la frecuencia
  X_train <- X_train %>%
    left_join(freq_table, by = var) %>%
    mutate(!!var := freq) %>%
    select(-freq)
}

# Variables categoricas
catg_vars <- c(ordinal_vars, nominal_vars)

# Convertimos valores faltantes a "NA" explícito
for (var in catg_vars) {
  if (var %in% names(X_train)) {
    X_train[[var]] <- as.character(X_train[[var]])
    X_train[[var]][is.na(X_train[[var]])] <- "NA"
  }
}

# Creamos una lista con todos los valores que puede tener cada variable categorica ordinal
levels_list <- list(
  ExterQual      = c("Po", "Fa", "TA", "Gd", "Ex"),
  ExterCond      = c("Po", "Fa", "TA", "Gd", "Ex"),
  BsmtQual       = c("NA", "Po", "Fa", "TA", "Gd", "Ex"),
  BsmtCond       = c("NA", "Po", "Fa", "TA", "Gd", "Ex"),
  BsmtExposure   = c("NA", "No", "Mn", "Av", "Gd"),
  BsmtFinType1   = c("NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"),
  BsmtFinType2   = c("NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"),
  HeatingQC      = c("Po", "Fa", "TA", "Gd", "Ex"),
  KitchenQual    = c("Po", "Fa", "TA", "Gd", "Ex"),
  FireplaceQu    = c("NA", "Po", "Fa", "TA", "Gd", "Ex"),
  GarageQual     = c("NA", "Po", "Fa", "TA", "Gd", "Ex"),
  GarageCond     = c("NA", "Po", "Fa", "TA", "Gd", "Ex"),
  PavedDrive     = c("N", "P", "Y"),
  Fence          = c("NA", "MnWw", "GdWo", "MnPrv", "GdPrv"),
  Functional     = c("Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ")
)

# Convertimos a factor con niveles ordenados
for (var in ordinal_vars) {
  if (var %in% names(X_train)) {
    X_train[[var]] <- factor(X_train[[var]], 
                             levels = levels_list[[var]],
                             ordered = TRUE)
    # Convertimos a numérico (la codificación ordinal final)
    X_train[[var]] <- as.numeric(X_train[[var]])
  }
}

# Con la lista anterior de variables nominales procedemos a hacer el One-Hot Encoding
# Creamos los dummies con model.matrix
X_train_dummies <- model.matrix(~ . -1, data = X_train[, nominal_vars])
X_train_dummies <- as.data.frame(X_train_dummies)

# Reemplazamos las columnas originales por los dummies
X_train <- X_train %>%
  select(-all_of(nominal_vars)) %>%  # eliminamos las nominales
  cbind(X_train_dummies)             # combinamos con los dummies

# Revisamos
dim(X_train)
view(X_train)

# -- Vamos a codificar los NA de las variables numericas --
# __________________________________________________________

# Vamos a sacar las variables numericas que tengan NA entre ellas
colSums(is.na(X_train))[colSums(is.na(X_train)) > 0]

# Vamos a imputar por mediana de Neighborhood la variable LotFrontage, ya que el espacio que hay
# en LotFrontage corresponde al barrio
X_train$LotFrontage <- X_train %>%
  group_by(Neighborhood) %>%
  mutate(LotFrontage = ifelse(is.na(LotFrontage),
  median(LotFrontage, na.rm = TRUE),
  LotFrontage)) %>%
  pull(LotFrontage)

dim(X_train)

# Si MasVnrArea es NA simplemente lo pasamos a 0, ya que significa que no tiene este tipo de revestimiento
X_train$MasVnrArea[is.na(X_train$MasVnrArea)] <- 0

# Si GarageYrBlt es NA simplemente lo pasamos a 0, ya que significa que no tiene garaje
X_train$GarageYrBlt[is.na(X_train$GarageYrBlt)] <- 0

# Confirmamos que ya no nos quedan NA en el df(X_train)
colSums(is.na(X_train))[colSums(is.na(X_train)) > 0]

# -- Vamos a eliminar datos sospechosos que empeoren nuestro modelo --
# _____________________________________________________________________

# ESTO SOLO SE VA A APLICAR EN EL TRAIN
# Juntamos momentaneamente los X e y para eliminar las filas
train_full <- cbind(X_train, SalePrice = y_train)

# Detectamos los outliers que hacen que nuestro modelo no coja correctamente la relación tamaño precio
outliers <- which(train_full$GrLivArea > 4500 & train_full$SalePrice < 300000)
cat("Outliers detectados:", outliers, "\n")    # En la partición de datos de train no nos ha tocado el caso anterior

# Quitamos las filas si es necesario
if (length(outliers) == 0) {
  cat("No se detectaron outliers.\n")
} else {
  cat("Outliers detectados:", outliers, "\n")
  train_full <- train_full[-outliers, ]
}

# Volvemos a dividir en X e y
y_train <- train_full$SalePrice
X_train <- subset(train_full, select = -SalePrice)

# Miramos las dimensiones con las que nos hemos quedado en el train 
dim(X_train)



# ===========================================
# == Escalado de train a validation y test ==
# ===========================================

# -- Eliminar mismas columnas en val y test --
# ______________________________________________

X_val  <- X_val %>% select(-all_of(list_to_del))
X_test <- X_test %>% select(-all_of(list_to_del))

# -- Codificamos las variables categoricas --
# _____________________________________________

# Por frecuencia
for (var in extra_cat_vars) {
  freq_table <- freq_tables[[var]]  # tomamos la tabla correspondiente
  
  X_val <- X_val %>%
    left_join(freq_table, by = var) %>%
    mutate(!!var := ifelse(is.na(freq), 0, freq)) %>%
    select(-freq)
  
  X_test <- X_test %>%
    left_join(freq_table, by = var) %>%
    mutate(!!var := ifelse(is.na(freq), 0, freq)) %>%
    select(-freq)
}

# Convertir NA explícitos en variables categóricas
for (var in catg_vars) {
  
  if (var %in% names(X_val)) {
    X_val[[var]] <- as.character(X_val[[var]])
    X_val[[var]][is.na(X_val[[var]])] <- "NA"
  }
  
  if (var %in% names(X_test)) {
    X_test[[var]] <- as.character(X_test[[var]])
    X_test[[var]][is.na(X_test[[var]])] <- "NA"
  }
}

# Codificación ordinal con los mismos niveles del TRAIN
for (var in ordinal_vars) {
  if (var %in% names(X_val)) {
    X_val[[var]] <- factor(X_val[[var]], 
                           levels = levels_list[[var]],
                           ordered = TRUE)
    X_val[[var]] <- as.numeric(X_val[[var]])
  }
  
  if (var %in% names(X_test)) {
    X_test[[var]] <- factor(X_test[[var]], 
                            levels = levels_list[[var]],
                            ordered = TRUE)
    X_test[[var]] <- as.numeric(X_test[[var]])
  }
}

# One-Hot Encoding con las MISMAS columnas que el TRAIN
# Generamos los dummies
X_val_dummies  <- model.matrix(~ . -1, data = X_val[, nominal_vars])
X_test_dummies <- model.matrix(~ . -1, data = X_test[, nominal_vars])

# Convertimos a dataframe
X_val_dummies  <- as.data.frame(X_val_dummies)
X_test_dummies <- as.data.frame(X_test_dummies)

# Mismas columnas que en el train
dummy_cols <- colnames(X_train_dummies)  # guardado del paso del train

# Añadir columnas faltantes
for (col in dummy_cols) {
  if (!(col %in% colnames(X_val_dummies)))  X_val_dummies[[col]] <- 0
  if (!(col %in% colnames(X_test_dummies))) X_test_dummies[[col]] <- 0
}

# Asegurar el MISMO orden
X_val_dummies  <- X_val_dummies[, dummy_cols]
X_test_dummies <- X_test_dummies[, dummy_cols]

# Reemplazamos las nominales por los dummies
X_val <- X_val %>%
  select(-all_of(nominal_vars)) %>%
  cbind(X_val_dummies)

X_test <- X_test %>%
  select(-all_of(nominal_vars)) %>%
  cbind(X_test_dummies)

# -- Codificamos las variables numericas --
# __________________________________________

# LotFrontage por mediana por Neighborhood
mediana_LF <- X_train %>%
  group_by(Neighborhood) %>%
  summarise(medLF = median(LotFrontage, na.rm = TRUE))

X_val <- X_val %>%
  left_join(mediana_LF, by = "Neighborhood") %>%
  mutate(LotFrontage = ifelse(is.na(LotFrontage), medLF, LotFrontage)) %>%
  select(-medLF)

X_test <- X_test %>%
  left_join(mediana_LF, by = "Neighborhood") %>%
  mutate(LotFrontage = ifelse(is.na(LotFrontage), medLF, LotFrontage)) %>%
  select(-medLF)

# MasVnrArea y GarageYrBlt igual que en train
X_val$MasVnrArea[is.na(X_val$MasVnrArea)]   <- 0
X_test$MasVnrArea[is.na(X_test$MasVnrArea)] <- 0

X_val$GarageYrBlt[is.na(X_val$GarageYrBlt)]   <- 0
X_test$GarageYrBlt[is.na(X_test$GarageYrBlt)] <- 0

# Revisamos
dim(X_train)
dim(X_val)
dim(X_test)
view(X_train)
view(X_val)
view(X_test)



# ================================
# == Normalización de los datos ==
# ================================

rec <- recipe(~ ., data = X_train) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric())

rec_prep <- prep(rec, training = X_train)

X_train <- bake(rec_prep, new_data = X_train)
X_val   <- bake(rec_prep, new_data = X_val)
X_test  <- bake(rec_prep, new_data = X_test)



# =========
# == PCA ==
# =========

# Ajustamos PCA
pca_model <- prcomp(
  X_train, 
  center = FALSE,   # ya está centrado en recipe
  scale  = FALSE    # ya está escalado en recipe
)

# Varianza explicada
var_exp <- pca_model$sdev^2 / sum(pca_model$sdev^2)
var_exp_acum <- cumsum(var_exp)

# Visualización rápida de varianza acumulada
plot(var_exp_acum, type = "b", pch = 19, col = "blue",
     xlab = "Número de Componentes",
     ylab = "Varianza explicada acumulada",
     main = "PCA")
abline(h = 0.90, col = "red", lty = 2)    # Linea para ver el % deseado mas facil

# Escalamos el PCA de train a validation y test
PC_train <- predict(pca_model, newdata = X_train)
PC_val   <- predict(pca_model, newdata = X_val)
PC_test  <- predict(pca_model, newdata = X_test)

# -- Seleccionamos el numero optimo de variables --
# ___________________________________________________

# Empezamos con la metrica de RMSE
rmse_results <- data.frame(
  componentes = integer(),
  RMSE_val = numeric()
)

# Sacamos un pequeño vistazo otra vez a las dimensiones para ver la cantidad de variables que tenemos
X_train_dims <- dim(X_train)[2]
X_train_dims

for (k in 1:X_train_dims) {   # probamos con todos los valores desde 1 hasta el maximo
  # Subconjunto de componentes
  train_k <- PC_train[, 1:k, drop = FALSE]
  val_k   <- PC_val[,   1:k, drop = FALSE]
  
  # Modelo de regresión lineal
  model_k <- lm(y_train ~ ., data = as.data.frame(train_k))
  
  # Predicciones en validation
  pred_val_train <- predict(model_k, newdata = as.data.frame(val_k))
  
  # RMSE
  rmse_k <- sqrt(mean((pred_val_train - y_val)^2))
  rmse_results <- rbind(rmse_results, data.frame(componentes = k, RMSE_val = rmse_k))
}

# Mejor número de componentes
best_k <- rmse_results$componentes[which.min(rmse_results$RMSE_val)]
best_k

plot(rmse_results$componentes, rmse_results$RMSE_val, type="b", pch=19,
     xlab="Número de Componentes",
     ylab="RMSE (Validation)",
     main="Selección de Componentes PCA")
abline(v = best_k, col="red", lty=2)



# ==========================
# == Modelo final con PCA ==
# ==========================

# Aplicamos el mejor valor del PCA al entrenamiento
train_final <- as.data.frame(PC_train[, 1:best_k])
val_final   <- as.data.frame(PC_val[,   1:best_k])
test_final  <- as.data.frame(PC_test[,  1:best_k])

model_pca <- lm(y_train ~ ., data = train_final)

# Predecimos los valores de validation y test
pred_val_pca <- predict(model_pca, newdata = val_final)
pred_test_pca <- predict(model_pca, newdata = test_final)

# Sacamos las metricas finales
rmse_val_pca  <- sqrt(mean((pred_val_pca - y_val)^2))
rmse_test_pca <- sqrt(mean((pred_test_pca - y_test)^2))

# RMSE de validation por PCA
rmse_val_pca

# RMSE de test por PCA
rmse_test_pca



# =====================================
# == Regresión con modelo Lasso (L1) ==
# =====================================

# Convertimos a matrices para glmnet
X_train_mat <- as.matrix(X_train)
X_val_mat   <- as.matrix(X_val)
X_test_mat  <- as.matrix(X_test)

# -- Ajuste de LASSO (alpha=1) --
# _________________________________

lasso_model <- glmnet(
  X_train_mat, y_train,
  alpha = 1,
  standardize = FALSE
)

# Secuencia de lambdas probadas por glmnet
lambdas <- lasso_model$lambda
rmse_lasso <- c()

# Hacemos el for para ver todos los RMSE de cada lambda con Lasso
for (lambd_l in lambdas) {
  preds <- predict(lasso_model, s = lambd_l, newx = X_val_mat)
  rmse_lasso <- c(rmse_lasso, sqrt(mean((preds - y_val)^2)))
}

# Escogemos la mejor lambda para el modelo Lasso
best_lambda_lasso <- lambdas[which.min(rmse_lasso)]
best_lambda_lasso

plot(log(lambdas), rmse_lasso, type="b", pch=19,
     main="Selección de lambda - LASSO",
     xlab="log(lambda)", ylab="RMSE (Validation)")
abline(v=log(best_lambda_lasso), col="red", lty=2)

# Modelo final
pred_val_lasso  <- predict(lasso_model, s=best_lambda_lasso, newx=X_val_mat)
pred_test_lasso <- predict(lasso_model, s=best_lambda_lasso, newx=X_test_mat)

rmse_val_lasso  <- sqrt(mean((pred_val_lasso - y_val)^2))
rmse_test_lasso <- sqrt(mean((pred_test_lasso - y_test)^2))

# RMSE de validation por Lasso
rmse_val_lasso

# RMSE de test por Lasso
rmse_test_lasso



# =====================================
# == Regresión con modelo Ridge (L2) ==
# =====================================

ridge_model <- glmnet(
  X_train_mat, y_train,
  alpha = 0,
  standardize = FALSE
)

lambdas_r <- ridge_model$lambda
rmse_ridge <- c()

# Hacemos el for para ver todos los RMSE de cada lambda con Ridge
for (lambd_r in lambdas_r) {
  preds <- predict(ridge_model, s=lambd_r, newx=X_val_mat)
  rmse_ridge <- c(rmse_ridge, sqrt(mean((preds - y_val)^2)))
}

# Escogemos la mejor lambda para el modelo Ridge
best_lambda_ridge <- lambdas_r[which.min(rmse_ridge)]
best_lambda_ridge

plot(log(lambdas_r), rmse_ridge, type="b", pch=19,
     main="Selección de lambda - RIDGE",
     xlab="log(lambda)", ylab="RMSE (Validation)")
abline(v=log(best_lambda_ridge), col="red", lty=2)

# Modelo final
pred_val_ridge  <- predict(ridge_model, s=best_lambda_ridge, newx=X_val_mat)
pred_test_ridge <- predict(ridge_model, s=best_lambda_ridge, newx=X_test_mat)

rmse_val_ridge  <- sqrt(mean((pred_val_ridge - y_val)^2))
rmse_test_ridge <- sqrt(mean((pred_test_ridge - y_test)^2))

# RMSE de validation por Ridge
rmse_val_ridge

# RMSE de test por Ridge
rmse_test_ridge



# ======================================
# == Regresión con modelo Lasso + PCA ==
# ======================================

# Convertimos a matrices para glmnet
PC_train_mat <- as.matrix(PC_train)
PC_val_mat   <- as.matrix(PC_val)
PC_test_mat  <- as.matrix(PC_test)

lasso_pca_model <- glmnet(
  PC_train_mat, y_train,
  alpha = 1,
  standardize = FALSE
)

lambdas_lp <- lasso_pca_model$lambda
rmse_lasso_pca <- c()

# Hacemos el for para ver todos los RMSE de cada lambda con Lasso + PCA
for (lamd_lpca in lambdas_lp) {
  preds <- predict(lasso_pca_model, s=lamd_lpca, newx=PC_val_mat)
  rmse_lasso_pca <- c(rmse_lasso_pca, sqrt(mean((preds - y_val)^2)))
}

# Escogemos la mejor lambda para el modelo Lasso + PCA
best_lambda_lasso_pca <- lambdas_lp[which.min(rmse_lasso_pca)]
best_lambda_lasso_pca

# Modelo final
pred_val_lasso_pca  <- predict(lasso_pca_model, s=best_lambda_lasso_pca, newx=PC_val_mat)
pred_test_lasso_pca <- predict(lasso_pca_model, s=best_lambda_lasso_pca, newx=PC_test_mat)

rmse_val_lasso_pca  <- sqrt(mean((pred_val_lasso_pca - y_val)^2))
rmse_test_lasso_pca <- sqrt(mean((pred_test_lasso_pca - y_test)^2))

# RMSE de validation por Lasso + PCA
rmse_val_lasso_pca

# RMSE de test por Lasso + PCA
rmse_test_lasso_pca



# ======================================
# == Regresión con modelo Ridge + PCA ==
# ======================================

ridge_pca_model <- glmnet(
  PC_train_mat, y_train,
  alpha = 0,
  standardize = FALSE
)

lambdas_rp <- ridge_pca_model$lambda
rmse_ridge_pca <- c()

# Hacemos el for para ver todos los RMSE de cada lambda con Ridge + PCA
for (lambd_rpca in lambdas_rp) {
  preds <- predict(ridge_pca_model, s=lambd_rpca, newx=PC_val_mat)
  rmse_ridge_pca <- c(rmse_ridge_pca, sqrt(mean((preds - y_val)^2)))
}

# Escogemos la mejor lambda para el modelo Ridge + PCA
best_lambda_ridge_pca <- lambdas_rp[which.min(rmse_ridge_pca)]
best_lambda_ridge_pca

# Modelo final
pred_val_ridge_pca  <- predict(ridge_pca_model, s=best_lambda_ridge_pca, newx=PC_val_mat)
pred_test_ridge_pca <- predict(ridge_pca_model, s=best_lambda_ridge_pca, newx=PC_test_mat)

rmse_val_ridge_pca  <- sqrt(mean((pred_val_ridge_pca - y_val)^2))
rmse_test_ridge_pca <- sqrt(mean((pred_test_ridge_pca - y_test)^2))

# RMSE de validation por Ridge + PCA
rmse_val_ridge_pca

# RMSE de test por Ridge + PCA
rmse_test_ridge_pca



# ===========================================================
# == CALCULAREMOS TAMBIEN EL MAE Y R² para mas información ==
# ===========================================================

# Creamos la funcion para MAE
mae <- function(actual, pred) {
  mean(abs(actual - pred))
}

# Creamos la funcion para R²
r2 <- function(actual, pred) {
  1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
}

# Creamos vectores para cada métrica
MAE_Validation <- c(
  mae(y_val, pred_val_pca),
  mae(y_val, pred_val_lasso),
  mae(y_val, pred_val_ridge),
  mae(y_val, pred_val_lasso_pca),
  mae(y_val, pred_val_ridge_pca)
)

MAE_Test <- c(
  mae(y_test, pred_test_pca),
  mae(y_test, pred_test_lasso),
  mae(y_test, pred_test_ridge),
  mae(y_test, pred_test_lasso_pca),
  mae(y_test, pred_test_ridge_pca)
)

R2_Validation <- c(
  r2(y_val, pred_val_pca),
  r2(y_val, pred_val_lasso),
  r2(y_val, pred_val_ridge),
  r2(y_val, pred_val_lasso_pca),
  r2(y_val, pred_val_ridge_pca)
)

R2_Test <- c(
  r2(y_test, pred_test_pca),
  r2(y_test, pred_test_lasso),
  r2(y_test, pred_test_ridge),
  r2(y_test, pred_test_lasso_pca),
  r2(y_test, pred_test_ridge_pca)
)



# ========================
# == RESULTADOS FINALES ==
# ========================

resultados <- data.frame(
  Modelo = c(
    "PCA + Linear",
    "LASSO",
    "RIDGE",
    "LASSO + PCA",
    "RIDGE + PCA"
  ),
  RMSE_Validation = c(
    rmse_val_pca,
    rmse_val_lasso,
    rmse_val_ridge,
    rmse_val_lasso_pca,
    rmse_val_ridge_pca
  ),
  RMSE_Test = c(
    rmse_test_pca,
    rmse_test_lasso,
    rmse_test_ridge,
    rmse_test_lasso_pca,
    rmse_test_ridge_pca
  ),
  MAE_Validation = MAE_Validation,
  MAE_Test = MAE_Test,
  R2_Validation = R2_Validation,
  R2_Test = R2_Test
)

# Tabla para salida por codigo "bonita"
kable(resultados, 
      caption = "Comparación de Modelos: RMSE, MAE y R²",
      digits = 2)

# Tabla para imagen "bonita"
datatable(resultados, 
          options = list(pageLength = 5),
          caption = 'Comparación de Modelos: RMSE, MAE y R²')
