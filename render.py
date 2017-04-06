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
