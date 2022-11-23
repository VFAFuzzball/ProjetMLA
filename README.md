# Projet de l'UE "Machine Learning Avancé", M2 Systèmes Avancés et Robotique, Sorbonne Université 2022-2023

## Groupe :

- Bounadja Bilal
- Daube Victor 3671193
- Nameki Malo 3800560
- Touami Abdelrahmaine

## Objectif :

- Reproduire les résultats expérimentaux obtenus dans l'article "Goodfellow, Ian & Shlens, Jonathon & Szegedy, Christian. (2014). Explaining and Harnessing Adversarial Examples. arXiv 1412.6572. "

## Framework utilisé :

- TensorFlow

## Résultats reproduits :

1. Linear Perturbation of Non-Linear Models :
   1. Méthode du signe du gradient rapide (p. 3 début, dataset : ImageNet ou CIFAR-10, modèle : GoogLeNet)
      - en utilisation eps = .25, un classificateur shallow softmax doit avoir une erreur de 99.9% avec une confiance moyenne de 79.3% sur le test set de MNIST
      - même configuration avec un réseau maxout 89.4% d'erreurs et confiance à 97.6%
      - eps = .1 réseau convolutionnel maxout sur une version prétraité de CIFAR-10 erreur 87.15% confiance à 96.6% sur les labels incorrects
      - rotation de x par un petit angle en direction du gradient donne également des exemples adversariaux
2. Adversarial Training of Linear Models versus Weight Decay
   1. Test de la méthode précédente avec une régression logistique pour classifier des 3 et des 7 (p. 3 fin, p. 4)
      - modèle régression logistique erreur 1.6%
      - fast gradient sign adversarial examples for the logistic regression model eps = .25 erreur 99%
      - multiclass softmax regression, maxout networks on MNIST, good results using adversarial training with eps = .25
      - idem précédent, coefficient = .0025 => > 5% error
      - smaller weight decay coefficients permitted succesful training but conferred no regularization benefit
3. Adversarial Training of Deep Networks :
   1. Szegedy et al. (2014b) : demander s'il faut aussi faire les tests cités
   2. Training with an adversarial objective funtion based on the fast gradient sign method, alpha = 0.5 => effective regularizer (p. 5)
      - train a maxout network that was also regularized with dropout, error rate 0.94% without adversarial training to 0.84% with adversarial training
        - The original maxout result uses early stopping, and terminates learning after the validation set error rate has not decreased for 100 epochs
      - model larger : 1600 units per layer rather than 240 used by the original maxout network. wo adv error 1.14% on test set / with adv error very slow progress (paragraphe 3 p. 5)
      - retrain on 60000 examples => error rate 0.77 0.83% (MNIST)
   3. paragraphe 4 p. 5 : the error rate fell to 17.9%, adversarial examples generated via the original model error rate 19.6% on the adversially trained model // adversarial examples generated via the new model error rate 40.9% on original model // when adversarially trained model does misclassify an adversarial example, its predictions are unfortunately still highly confident => confidence on a misclassified example 81.4% + weights of the adversarially trained model significantly more localized and interpretable
   4. 
4. Different Kinds of Model Capacity

## Installation et exécution des programmes :

### Programmes d'entrainement :

### Programmes de prédiction :

## Description des fichiers :
