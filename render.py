import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve

def plot_calibration_curve(classificadores, X_train, y_train, X_test, y_test, pos_label):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Calibrado perfeitamente")
    
    for clf, name in classificadores:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=pos_label)
        #pos_label: positive label será usado o máximo, 1 = SIM (DESISTENTE)
        #print("%s:" % name)
        #print("\tBrier: %1.3f" % (clf_score))
        #print("\tPrecision: %1.3f" % precision_score(y_test, y_pred, pos_label=pos_label))
        #print("\tRecall: %1.3f" % recall_score(y_test, y_pred, pos_label=pos_label))
        #print("\tF1: %1.3f\n" % f1_score(y_test, y_pred, pos_label=pos_label))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fração de positivos")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower right")
    #ax1.set_title('Gráfico de calibragem  (curva de confiança)')

    ax2.set_xlabel("Valor médio previsto")
    ax2.set_ylabel("Contagem")
    #ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

from time import time
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train, clf_params):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    if False:
        start = time()
        svr = GridSearchCV(clf, clf_params)
        svr.fit(X_train, y_train[0].values)
        end = time()
        clf = svr.best_estimator_
    else:
        start = time()
        clf.fit(X_train, y_train)
        end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    if True:
        return f1_score(target.values, y_pred, pos_label='SIM')
    else:
        return f1_score(target.values, y_pred, pos_label=54)


def train_predict(clf, X_train, y_train, X_test, y_test, clf_params):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train, clf_params)
    
    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
