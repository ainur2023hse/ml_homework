# ml_homework

Был изучен и обработан датасет с признаками автомобилей и с целевой переменной в виде цены авто. В датасете значения признаков были приведены к удобному формату, примитивным образом заполнены пропуски и стандартизированы числовые признаки,
визуализированы парные распределения и корреляций

На числовых признаках были обучены регресионные модели (Linear, Lasso, Elasticnet) оптимальные параметры подбирались с помощью GridSearch, значения R2 во всех случаях оказалось близко к 0.6.

На числовых и категориальных признаках была обучена Ridge модель (параметры регуляризации подбирались с помощью GridSearch с метрикой качества R2), при этом категориальные признаки были закодированы методом OneHot, удалось получить R2 близкое к 0.65.
Так же был написан простой сервис для предсказания цены на FastApi

Наибольший буст в качестве дало добавление категориальных признаков.





![Screenshot from 2023-11-30 02-03-35](https://github.com/ainur2023hse/ml_homework/assets/148802637/656e2b7f-fe8e-4761-b889-9449c7838491)



![Screenshot from 2023-11-30 02-04-40](https://github.com/ainur2023hse/ml_homework/assets/148802637/20f489c5-de3e-4a35-bb3b-06efdcc2ddc0)
