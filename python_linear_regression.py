import os
print("hello worldwor")
from pycaret.datasets import get_data
#dataset = get_data('credit')
# Importing data using pandas
import pandas as pd
os.getcwd()
dataset = pd.read_csv('/home/iakov/Documents/work_about_pycaret/credit.csv')
dataset.shape
#мы создаем иллюзию "удаленния данных" для того что бы симулировать реальное положение дел. изымаем 1200 записей якобы они не были доступны во время эксперемента с данными
data = dataset.sample(frac=0.95, random_state = 785)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
'''
setup() инициализирует среду в pycaret и создает коннвеер преобразований для подготовки данных для моделирования и развертывания
два обязательный параметра, фрейм данных пандас и имя целевого столбца. Все остальные параметры являются необязательными и используется в следующих руководствах
когдда setup() выполнянется, алгоритм pycaret автоматически определяет типы данных, тип данных следует определять правильно, но это не всегда так. Выводит таблицу для сверки данных
Можно нажать enter если все данные правильно определенны, если нет то   quit . Внимательно к этому отнестись.
для перезаписи введеных типов данных используются параметры
numeric_features и categoryorical_features в setup()
'''
from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'default', session_id = 123)
'''
информация о работе setup()
session_id псевдослучайное число для повторной работы эксперемента
target type двоичный или мультиклассовый, все функции индетичны разницы впринципе нет
Label Encoded: когда целевая переменная имеет строковый тип(например, "Да" или "Нет") вместо 1 или 0,
он автоматически кодирует метку 1 в 0 и отображает (0:нет, 1:Да) для справки. В этом экс, кодирование меток не нужно
посколько переменная имеет числовой тип
Original Data: отображает исходную форму набора данных. В этом (22800, 24) 22 800 образцов и 24 функции, включая целевой столбец
Missing Values: отсуствующие значения (если отсуствуют = TRUE), в этом наборе данных таких Нет
Numeric Features: колличевство функции выводимых как числовые, в этом наборе данных 14 из 24 числовые
Categoryorical Features: колличевство характеристик, определенных как категориальные, в этом наборе 9 из 24 считаются категориальные
Tranformed Train Set: отображает форму преобразованого обучающего набора. Обратите внимане что исходная форма (22800, 24) -> (15959, 91) для преобразования набора train,
а колличевство функций 24 -> 91 из за категориального кодирования
Tranformed Test set: отображает форму преобразованого набора тестов/удержания. В наборе для испытания выдержка 6841 образец. это разбиение по значению 70/30, которое можно изменить с помощью параметра train_set
Перекресткая проверка для уточнения модели ML
'''
best_model = compare_models()
print(best_model)
'''

    Классификатор дерева решений ('dt')
    Классификатор соседей K ('knn')
    Классификатор случайного леса ('rf')

'''
models()
#Классификатор дерева решений
dt = create_model('dt')
rf = create_model('rf')
import pandas as pd

# Enable the table_schema option in pandas, 
# data-explorer makes this snippet available with the `dx` prefix:
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

# (Your dataframe here)
iris_filename = './iris.csv'
df1 = pd.read_csv(iris_filename)

df1
