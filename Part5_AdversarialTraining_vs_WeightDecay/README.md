Adversarial Training of Linear Models versus Weight Decay1. Test de la méthode précédente avec une régression logistique pour classifier des 3 et des 7 (p. 3 fin, p. 4)

- modèle régression logistique erreur 1.6%
- fast gradient sign adversarial examples for the logistic regression model eps = .25 erreur 99%
- multiclass softmax regression, maxout networks on MNIST, good results using adversarial training with eps = .25
- idem précédent, coefficient = .0025 => > 5% error
- smaller weight decay coefficients permitted succesful training but conferred no regularization benefit
