import numpy as np
import csv


class Perceptron:

    def __init__(self):
        self._weights = None
        self._vectors = None

    def train(self, training, labels):

        rows = len(training)                 # e.g. 450
        columns = len(training[0])           # e.g. 41567

        self._vectors = np.zeros((rows, columns))
        self._weights = np.zeros((rows, 1))

        vectors = self._vectors              # v-vector
        weights = self._weights              # c-vector

        k = 0                                # k-value
        t = 0                                # t-value
        epochs = 20                          # T-value

        while t < epochs:
            for i in range(rows):
                point = training[i]
                label = 1 if labels[i] == 1 else -1
                pred = self.predict(point)
                if pred == label:
                    weights[k] += 1                 # c[k] = c[k] + 1
                else:
                    v = vectors[k - 1] if k > 0 else np.zeros((1, columns))     # previous value or init new vector
                    temp = label * point            # y_i * x_i
                    vectors[k] = np.add(v, temp)    # V[k+1] = V[k]+y_i*x_i
                    weights[k] = 1                  # c[k+1] = 1
                    k += 1
            t += 1

    def predict(self, instance):
        vectors = self._vectors
        weights = self._weights
        product = np.matmul(vectors, instance)
        sign = np.sign(product)
        sub = np.sum(np.multiply(weights, sign))
        s = np.sum(sub)
        return np.sign(s)

    def classify(self, vectors):
        predictions = []
        for instance in vectors:
            pred = int(self.predict(instance))
            if pred <= 0:
                res = 0
            else:
                res = 1
            # print(res)
            predictions.append(res)
        # print(predictions)
        return predictions

    def evaluate(self, vectors, labels):
        results = self.classify(vectors)
        score = 0
        for i, prediction in enumerate(results):
            if prediction == labels[i]:
                score += 1
        return score / len(vectors)


def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):

    # Read data
    training = read_csv(Xtrain_file)
    labels = read_csv(Ytrain_file)
    testing = read_csv(test_data_file)

    # Train perceptron model
    perceptron = Perceptron()
    perceptron.train(training, labels)

    # Generate predictions
    predictions = perceptron.classify(testing)

    # Save to file
    np.savetxt(pred_file, predictions, fmt='%1d', delimiter=",")


def read_csv(file):
    return np.genfromtxt(file, delimiter=',')


if __name__ == "__main__":

    Xtrain_file = 'Xtrain.csv'
    Ytrain_file = 'Ytrain.csv'
    pred_file = 'result.txt'

    all_data = read_csv(Xtrain_file)
    all_labels = read_csv(Ytrain_file)

    total_size = len(all_data)                  # 500
    training_size = int(total_size * .9)        # 450/500
    testing_size = int(total_size * .1)         # 50/500

    # Partition data into training/testing sets

    training_data = all_data[: training_size]
    training_labels = all_labels[: training_size]
    # print(len(training_data))

    testing_data = all_data[-testing_size:]
    testing_labels = all_labels[-testing_size:]
    # print(len(testing_data))

    model = Perceptron()
    model.train(training_data, training_labels)
    test = model.classify(testing_data)
    print(test)

    output = np.savetxt(pred_file, test, fmt='%1d', delimiter=",")
    fd = open('result.txt', 'r')
    print(fd.read())

    # output = model.evaluate(testing_data, testing_labels)
    # print(output)
