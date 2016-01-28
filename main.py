from helpers import loadData, saveResults, showFreq
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt

TEST_FILE = "test.csv"


def main():
    target, train = loadData()
    showFreq(target)

    rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, verbose=3)
    print "Training..."
    t0 = time()
    rfc.fit(train, target)
    print "done in %0.3fs" % (time() - t0)

    # print "Scoring..."
    # print rfc.score(train, target)

    print "Predicting..."
    test = genfromtxt(TEST_FILE, dtype='int', delimiter=',', skip_header=True)
    results = rfc.predict(test)
    saveResults(results)

    # for index, image in enumerate(test[:4]):
    #     plt.subplot(2, 4, index + 5)
    #     plt.axis('off')
    #     plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()

    importances = rfc.feature_importances_
    importances = importances.reshape((28, 28))

    # Plot pixel importances
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.title("Pixel importances with forests of trees")
    plt.show()


if __name__ == '__main__':
    main()
