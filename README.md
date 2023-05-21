# cern_higgs_boson_ml

The goal of this task is to distinguish between two types of events that occur when particles collide at high energies in the **ATLAS experiment** at the **Large Hadron Collider (LHC)** at CERN. One type of event is the decay of a **Higgs boson** into two **tau leptons**, which is a rare and important process that reveals the origin of mass.  The other type of event is the **background**, which consists of other processes that mimic the signal but are not related to the Higgs boson. The challenge is to find a way to maximize the **discovery significance** of the signal over the background, using a statistical measure called the **approximate median significance (AMS)**(Please check the notebooks for more details). 

Here, I have used a hybrid boosted bagging algorithm which we can call **Boosted Extremely Randomized Trees(BXT)**. This model builds uses an **AdaBoost** algorithm with a collection of **Extremely Randomized Trees(ET)** as estimators. This should in theory, should nullify the shorcomings of each model with AdaBoost having a tendecncy to overfit and ETs having high bias, vis a vis reduce the bias and variance especially for a an unbalanced data set such as this. Furthermore, I am doing a comparitive study between **BXT** and other models such as **Random Forest**, **AdaBoost** and a **Deep Neural Network** with two hidden layers, to analyse how well the BXT fares against these models.

# RESULTS

A hyperparameter search was done on all models. Higher the AMS, better is the model perfomarmance.

| Model       | AMS Value  |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

