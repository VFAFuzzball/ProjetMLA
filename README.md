# Reimplementing important results from "[Goodfellow, Ian & Shlens, Jonathon & Szegedy, Christian. (2014). Explaining and Harnessing Adversarial Examples. arXiv 1412.6572.](https://arxiv.org/abs/1412.6572) "

## Quick definition by the authors of the article :

> Adversarial examples : inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence

## Goals :

We tried to reimplement some important empirical results from "[Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)". We successfully implemented an adversarial example generator as described in the article and we tried it on a shallow softmax network and a maxout network (both linear neural networks). We did not manage to implement GoogLeNet and to try our adversarial examples generator on Tiny ImageNet. We then proved that a Radial Basis Function (RBF) network is more resistant to adversarial attacks than the linear models cited above.  

## Framework used :

- Python
- TensorFlow/Keras

## How to run the project :

We mostly used Jupyter Notebooks so you will only need to run the notebooks to use them. You may need to install common ML librairies to run the notebooks :
- tensorflow
- tensorflow_addons
And more common one :
- numpy
- matplotlib.pyplot
- google.colab
- PIL

## Résultats reproduits :

1. Linear Perturbation of Non-Linear Models :
   1. Fast Gradient Sign Method (FGSM)
      - With $\epsilon = 0.25$ (coming from the article), a shallow softmax classifier should have an error ra doit avoir une erreur de 99.9% avec une confiance moyenne de 79.3% sur le test set de MNIST perturbé par la FGSM
      - De même, avec un réseau maxout, on doit obtenir 89.4% d'erreurs et une confiance à 97.6%
      - Enfin, avec $\epsilon = 0.007$, le réseau GoogLeNet et la base de données ImageNet, on tente d'effectuer le même processus. Malheureusement, ImageNet étant une base de données assez lourde, nous avons opté pour TinyImageNet.
      
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
pip install requirement.txt

### Programmes d'entrainement :

### Programmes de prédiction :

## Description des fichiers :
- Le fichier fgsm.ipynb contient notre générateur d'exemples contradictoires ainsi que les résultats obtenus avec les réseaux shallow softmax 
RBF_network file is used to implement a Radial basis function network and we also test it on adversarial examples generated with the Fast gradient sign method
Part 5 repository is used to implement Logistic Regression to classify 3 and 7s from MNIST dataset. This model was used to have
an intuition in the process in creating the adversarial examples.
