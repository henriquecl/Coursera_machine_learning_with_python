"""
Evaluation metrics

How accurate are a model?

1 - Jaccard index
y: actual labels
y_hat: predicted labels
J(y,y_hat) = y interseção y_hat / y união y_hat
Quanto mais próximo de 1, melhor!

2 - F1-Score
TP = True Positive, FP = False Positive, FN = FALSE NEGATIVE
Precision = TP / (TP + FP)
Recall = TP/ (TP + FN)
F1-SCORE = 2x (prc * rec) / (prc+rec)
 0 to 1 , quanto mais próximo de um melhor.

3 - Log loss
formula no vídeo. Quanto mais próximo de 0 melhor.

"""