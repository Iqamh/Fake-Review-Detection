import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import csv
import os

# Read CSV File
data = pd.read_csv(
    'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\Final_Label_Preprocessed_Balance_Final-Plis.csv', delimiter=";")

# Hapus data kosong setelah preprocessed
data.dropna(subset=['Tokenized_Review'], inplace=True)

# Penentuan data fitur dan label
X = data["Tokenized_Review"]
y = data["Label Final"]

# TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X).toarray()

# Optimasi
hidden_layer = (50, 25, 25)
random_states = 42

# Batch sizes and alphas
batch_sizes_list = [8, 16, 32, 64, 128]
alphas_list = [0.01, 0.1, 0.16, 0.32, 0.5]
a_list = ["01", "10", "16", "32", "50"]


# Fungsi evaluasi model
def train_and_evaluate_model_splits(X, y, hidden_layer, random_states, batch_size, alpha, a, test_size, validation_size=0.1, epochs=100):

    # Set pembagian training dan test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_states)

    # Set pembagian training dan validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=random_states)

    # Mendefinisikan MLP Classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='relu', learning_rate_init=0.01,
                        solver='adam', alpha=alpha, batch_size=batch_size, max_iter=1, warm_start=True)

    loss_train = []
    accuracy_train = []
    loss_val = []
    accuracy_val = []

    # Iterasi per epoch
    for epoch in range(epochs):
        mlp.fit(X_train, y_train)

        loss_train.append(mlp.loss_)
        y_pred_train = mlp.predict(X_train)
        accuracy_train.append(accuracy_score(y_train, y_pred_train))

        y_pred_val_prob = mlp.predict_proba(X_val)
        loss_val_epoch = log_loss(y_val, y_pred_val_prob)
        loss_val.append(loss_val_epoch)
        y_pred_val = mlp.predict(X_val)
        accuracy_val.append(accuracy_score(y_val, y_pred_val))

        # Cetak progres
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {mlp.loss_:.4f} - Accuracy: {accuracy_train[-1]:.4f} - Val Loss: {loss_val[-1]:.4f} - Val Accuracy: {accuracy_val[-1]:.4f}")

    # Hasil Model
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy (Test Size {test_size}):", accuracy)
    print(f"\nClassification Report (Test Size {test_size}):")
    print(cr)
    print(f"Confusion Matrix (Test Size {test_size}):")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                'Asli', 'Palsu'], yticklabels=['Asli', 'Palsu'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix (Test Size {test_size})")
    plt.savefig(
        f'{dir_path}\\confusion_matrix_{test_size}_hs{hidden_layer}_bs{batch_size}_a{alpha}_lr0.001_ver4_real.png')
    # plt.show()

    # Grafik training dan validation loss dan accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_val, label='Validation Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Training and Validation")
    plt.legend()
    plt.savefig(
        f'{dir_path}\\loss_plot_{test_size}_hs{hidden_layer}_bs{batch_size}_a{alpha}_lr0.001_ver4_real.png')
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_train, label='Training Accuracy')
    plt.plot(accuracy_val, label='Validation Accuracy', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Training and Validation")
    plt.legend()
    plt.savefig(
        f'{dir_path}\\accuracy_plot_{test_size}_hs{hidden_layer}_bs{batch_size}_a{alpha}_lr0.001_ver4_real.png')
    # plt.show()

    return mlp, accuracy, cm, cr, loss_train, accuracy_train, loss_val, accuracy_val


# Loop through each combination of batch_size, alpha, and a
for alpha, a in zip(alphas_list, a_list):
    for batch_size in batch_sizes_list:

        # Directory base path
        base_path = f'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\AnotherFinal\\layer{hidden_layer[0]}-{hidden_layer[1]}-{hidden_layer[2]}\\a{a}'

        # Create directory for the current combination if it doesn't exist
        dir_name = f'layer{hidden_layer[0]}-{hidden_layer[1]}-{hidden_layer[2]}_lr01_bs{batch_size}_a{a}'
        dir_path = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Train and evaluate the model
        mlp_80_10_10, accuracy_80_10_10, cm_80_10_10, cr_80_10_10, loss_80_10_10, acc_80_10_10, loss_val_80_10_10, acc_val_80_10_10 = train_and_evaluate_model_splits(
            X_tfidf, y, hidden_layer, random_states, batch_size, alpha, a, test_size=0.1)
        mlp_70_20_10, accuracy_70_20_10, cm_70_20_10, cr_70_20_10, loss_70_20_10, acc_70_20_10, loss_val_70_20_10, acc_val_70_20_10 = train_and_evaluate_model_splits(
            X_tfidf, y, hidden_layer, random_states, batch_size, alpha, a, test_size=0.2)
        mlp_60_30_10, accuracy_60_30_10, cm_60_30_10, cr_60_30_10, loss_60_30_10, acc_60_30_10, loss_val_60_30_10, acc_val_60_30_10 = train_and_evaluate_model_splits(
            X_tfidf, y, hidden_layer, random_states, batch_size, alpha, a, test_size=0.3)

        # Simpan hasil model dan data untuk pengujian kembali
        model_data = {
            'mlp_80_10_10': mlp_80_10_10,
            'mlp_70_20_10': mlp_70_20_10,
            'mlp_60_30_10': mlp_60_30_10,
            'vectorizer': vectorizer
        }
        model_data_file = os.path.join(dir_path, 'model_data.pkl')
        with open(model_data_file, "wb") as file:
            pickle.dump(model_data, file)

        result = {
            "80_10_10": {
                "accuracy": accuracy_80_10_10,
                "classification_report": cr_80_10_10,
                "confusion_matrix": cm_80_10_10,
                "loss": loss_80_10_10,
                "accuracy_train": acc_80_10_10,
                "loss_validation": loss_val_80_10_10,
                "accuracy_validation": acc_val_80_10_10
            },
            "70_20_10": {
                "accuracy": accuracy_70_20_10,
                "classification_report": cr_70_20_10,
                "confusion_matrix": cm_70_20_10,
                "loss": loss_70_20_10,
                "accuracy_train": acc_70_20_10,
                "loss_validation": loss_val_70_20_10,
                "accuracy_validation": acc_val_70_20_10
            },
            "60_30_10": {
                "accuracy": accuracy_60_30_10,
                "classification_report": cr_60_30_10,
                "confusion_matrix": cm_60_30_10,
                "loss": loss_60_30_10,
                "accuracy_train": acc_60_30_10,
                "loss_validation": loss_val_60_30_10,
                "accuracy_validation": acc_val_60_30_10
            }
        }

        evaluation_file = os.path.join(dir_path, 'model_result.pkl')
        with open(evaluation_file, "wb") as file:
            pickle.dump(result, file)

        # Save the results to a CSV file
        modelling_csv_path = 'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\modelling2.csv'
        with open(modelling_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['Layer', 'Alpha', 'Batch Size', 'Loss 70:30', 'Accuracy 70:30', 'Loss Val 70:30', 'Accuracy Val 70:30', 'Loss 80:20',
                          'Accuracy 80:20', 'Loss Val 80:20', 'Accuracy Val 80:20', 'Loss 90:10', 'Accuracy 90:10', 'Loss Val 90:10', 'Accuracy Val 90:10']
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, delimiter=';')
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({
                'Layer': f'{hidden_layer[0]}-{hidden_layer[1]}-{hidden_layer[2]}',
                'Alpha': alpha,
                'Batch Size': batch_size,
                'Loss 70:30': f'{loss_60_30_10[-1]:.4f}',
                'Accuracy 70:30': f'{acc_60_30_10[-1]:.4f}',
                'Loss Val 70:30': f'{loss_val_60_30_10[-1]:.4f}',
                'Accuracy Val 70:30': f'{acc_val_60_30_10[-1]:.4f}',
                'Loss 80:20': f'{loss_70_20_10[-1]:.4f}',
                'Accuracy 80:20': f'{acc_70_20_10[-1]:.4f}',
                'Loss Val 80:20': f'{loss_val_70_20_10[-1]:.4f}',
                'Accuracy Val 80:20': f'{acc_val_70_20_10[-1]:.4f}',
                'Loss 90:10': f'{loss_80_10_10[-1]:.4f}',
                'Accuracy 90:10': f'{acc_80_10_10[-1]:.4f}',
                'Loss Val 90:10': f'{loss_val_80_10_10[-1]:.4f}',
                'Accuracy Val 90:10': f'{acc_val_80_10_10[-1]:.4f}'
            })
