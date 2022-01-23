from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
from sklearn.svm import SVC
import pandas as pd
import paramiko
import winsound
import time
import os


def prepare_data(file):
    df = pd.read_csv(file, na_values='?')

    encoder = LabelEncoder()
    df['Summary'] = encoder.fit_transform(df['Summary'])

    class_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("\nOutput classes:", class_dict, sep="\n")

    # Splitting dataset into features and labels
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    return X, y, class_dict


def random_forest(X_train, X_test, y_train, y_test):
    print("\nRandom Forest\n")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    rf_matrix = confusion_matrix(y_test, y_pred)

    print("\nRandom Forest Confussion matrix:\n\n", rf_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return rf_classifier


def logistic_reg(X_train, X_test, y_train, y_test):
    print("\nLogistic Regression\n")
    lr_classifier = LogisticRegression(random_state=0, max_iter=300)
    lr_classifier.fit(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return


def knn(X_train, X_test, y_train, y_test):
    print("\nK-NN\n")
    knn_classifier = KNeighborsClassifier(n_neighbors=len(class_map.keys()), metric='minkowski', p=2)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return


def svm(X_train, X_test, y_train, y_test):
    print("\nSupport Vectore Machines\n")
    svm_classifier = SVC(kernel='linear', random_state=0)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return


def neural_network(X_train, X_test, y_train, y_test):
    print("\nNeural Network\n")
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=800)
    nn_classifier.fit(X_train, y_train)
    y_pred = nn_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return


def decision_tree(X_train, X_test, y_train, y_test):
    print("\nDecision Tree\n")
    dtree_classifier = DecisionTreeClassifier()
    dtree_classifier = dtree_classifier.fit(X_train, y_train)
    y_pred = dtree_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return


def test_model(rf_classifier, sample, X_test, y_test, class_dict):
    accuracy = rf_classifier.score(X_test, y_test)
    print("\nAccuracy of model:", accuracy)

    output = ""

    prediction = rf_classifier.predict(sample)
    for key, value in class_dict.items():
        if prediction[0] == value:
            print("\nPredicted class for sample (", temperature, ',', humidity, ',', pressure, "):", '\n',
                  prediction[0], '-', key)
            output = key

    prediction = rf_classifier.predict_proba(sample)
    for index in range(len(prediction[0])):
        print("Class:", index, " Probability:", prediction[0][index])

    with open('C:/Users/Onur/Desktop/result.txt', 'w') as file:
        file.write(output)

    return output


def get_from_rpi():
    host = "raspberrypi.mshome.net"
    port = 22
    username = "pi"
    password = "raspberry"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)

    sftp = ssh.open_sftp()

    sftp.get("/home/pi/Desktop/Weather Forecast/sensor_values.txt",
             os.path.join("C:/Users/Onur/Desktop", "sensor_values.txt"))

    sftp.close()
    ssh.close()

    sensor_file = open("C:/Users/Onur/Desktop/sensor_values.txt", "r")
    sensor_values = sensor_file.read()

    sensor_values = sensor_values.split(" ")

    sensor_file.close()

    return sensor_values


def send_to_rpi():
    host = "raspberrypi.mshome.net"
    port = 22
    username = "pi"
    password = "raspberry"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)

    sftp = ssh.open_sftp()

    sftp.put("C:/Users/Onur/Desktop/result.txt",
             os.path.join("/home/pi/Desktop/Weather Forecast/", "result.txt"))
    sftp.close()
    ssh.close()

    return


if __name__ == "__main__":
    start_time = time.monotonic()

    dataset = "weather_history.csv"

    features, labels, class_map = prepare_data(dataset)

    # Splitting data into train and test
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=1)

    # Viewing shape of data
    print("\nDataset shape:", train_X.shape, test_X.shape, train_y.shape, test_y.shape, sep="\n")

    # Running and Saving Random Forest
    model = random_forest(train_X, test_X, train_y, test_y)
    # logistic_reg(train_X, test_X, train_y, test_y)
    # knn(train_X, test_X, train_y, test_y)
    # svm(train_X, test_X, train_y, test_y)
    # neural_network(train_X, test_X, train_y, test_y)
    # decision_tree(train_X, test_X, train_y, test_y)

    # pickle.dump(model, open("RF.sav", 'wb'))

    # Loading saved models and testing
    # model = pickle.load(open("RF.sav", 'rb'))

    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)

    print("\nExecution time:", total_time)


    test_sample = []
    try:
        program_select = int(input("\nEnter program mode {1 - Rpi} {2 - User}: "))
        while True:
            if program_select == 1:
                values = get_from_rpi()
                print("\nInfo: Sensor values are received from Raspberry Pi")

                temperature = values[0]
                humidity = values[1]
                pressure = values[2]

                test_sample = [[temperature, humidity, pressure]]

                print("Received Values", test_sample)

                result = test_model(model, test_sample, test_X, test_y, class_map)

                send_to_rpi()
                print("\nInfo: Sensor values are sent to Raspberry Pi")

                time.sleep(5)

            elif program_select == 2:
                temperature = float(input("\nEnter temperature:"))
                humidity = float(input("Enter humidity:"))
                pressure = float(input("Enter pressure:"))

                test_sample = [[temperature, humidity, pressure]]

                result = test_model(model, test_sample, test_X, test_y, class_map)

            else:
                print("\nPlease enter a valid program mode!")
                program_select = int(input("\nEnter program mode {1 - Rpi} {2 - User}: "))

    except KeyboardInterrupt:
        pass
