import pickle
import numpy
from numpy import genfromtxt
import matplotlib.pyplot as plt

TRAIN_FILE = 'train.csv'
DATA_PICKLE = 'train.pkl'
FRST_PICKLE = 'rfc.pkl'
HAS_HEADER = True


def unserialize(name):
    """
    Tries to load the pickle with the given name, returning the object
    or None
    """
    try:
        with open(name, 'rb') as f:
            print "Unserializing from %s" % name
            return pickle.load(f)
    except IOError:
        return None
    except Exception, e:
        raise e


def serialize(object, name):
    """
    Create pickle from given object with given filename
    """
    print "Serializing to %s" % name
    with open(name, 'wb') as f:
        pickle.dump(object, f)


def loadData():
    """
    Returns a tuple with the train data and target values (data, values)
    """
    train_data = None
    train_target = None
    temp_data = unserialize(DATA_PICKLE)
    if temp_data is None:
        temp_data = genfromtxt(TRAIN_FILE, dtype='int', delimiter=',', skip_header=HAS_HEADER)
        serialize(temp_data, DATA_PICKLE)
    train_target = numpy.array([i[0] for i in temp_data])
    train_data = numpy.array([i[1:] for i in temp_data])
    return (train_target, train_data)


def saveResults(results):
    """
    Takes the results array and prints it to the results.csv file, with
    header ImageId, Label
    """
    with open('results.csv', 'wb') as f:
        f.write('ImageId,Label\n')
        for index, result in enumerate(results):
            f.write('%d,%d\n' % (index+1, int(result)))


def showFreq(target):
    """
    Plots the train statistics
    """
    plt.hist(target, bins=numpy.arange(0, 11, 1)-0.5, normed=True, alpha=0.75,
             label="Digit percentage (%d total)" % len(target))
    plt.plot(list(range(-1, 11)), [0.1 for x in range(12)], 'r--', linewidth=2, label="Uniform distribution [0,9]")
    plt.title("Training distribution")
    plt.xlabel("Digits")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.xticks(numpy.arange(10))
    plt.xlim(-0.5, 9.5)
    plt.legend()
    plt.show()
