"""
K-Nearest Neighbours

-> Is a classification algorithm wich classify cases based on their similarity to other cases;
-> Cases that are near each other are said to be neighbors;
-> Similar cases with same class labels are near each other.

How it really works?

1. Pick a value for K;
2. Calculate the distance of unkown case from all cases;
3. Select de K-Observations in the training data that are "nearest" to the unknown data point;
4. Predict the response of the unknown data point using the most popular (moda) resopnse value from
the K-nearest neighbors.

2. How we can calculate the distance?
    -> Distância euclidianada: Distância vetorial, ou seja, Pitágoras.

1. What de best valures of de K for KNN?
    -> Se o K for muito pequeno, podemos nos deparar uma anomalia nos dados, o que nos da uma péssima acurácia
. Além disso, temos um problema de overfiting, servindo apenas para os dados utilizados e não para um caso
diferente;
    -> Se o K for muito grande, o modelo se torna overgeneralized.
    -> Ou seja, para saber qual o melhor, é necessário ir testando.


"""