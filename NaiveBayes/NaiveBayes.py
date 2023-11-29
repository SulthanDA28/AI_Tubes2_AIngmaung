import numpy as np
import pandas as pd
import sys

class NaiveBayesClassifier:
    """
    Naive Bayes classifier untuk data numerik dan kategorikal
    """
    def __init__(self):
        """
        Konstruktor kelas NaiveBayesClassifier
        
        Atribut:
        - class_priors (dict): prior probability dari kelas-kelas pada kolom target 
        - num_feature_means (dict): rata-rata nilai fitur kolom numerik untuk setiap kelas
        - num_feature_stds (dict): standar deviasi nilai fitur kolom numerik untuk setiap kelas
        - cat_feature_probs (dict): probabilitas nilai fitur kolom non numerik untuk setiap kelas
        - classes (np.ndarray): daftar kelas yang muncul pada kolom target
        """
        self.class_priors: dict = None
        self.num_feature_means: dict = None
        self.num_feature_stds: dict = None
        self.cat_feature_probs: dict = None
        self.classes: np.ndarray = None

    def fit(self, X: pd.DataFrame, y: pd.Series, numerical_columns: list, categorical_columns: list) -> None:
        """
        Melatih model Naive Bayes.

        Parameters:
        - X (pd.DataFrame): DataFrame berisi fitur-fitur dataset.
        - y (pd.Series): Seri berisi label-label kelas.
        - numerical_columns (list): Daftar kolom fitur numerik.
        - categorical_columns (list): Daftar kolom fitur kategorikal.
        """
        # Calculate class priors
        self.class_priors = {c: np.sum(y == c) / len(y) for c in np.unique(y)}
        self.classes = np.unique(y)
        
        # Calculate numerical feature statistics
        self.num_feature_means = {}
        self.num_feature_stds = {}
        for c in self.classes:
            c_mask = (y == c)
            class_features = X.loc[c_mask, numerical_columns]
            self.num_feature_means[c] = class_features.mean()
            self.num_feature_stds[c] = class_features.std()
            
        # Calculate categorical feature probabilities
        self.cat_feature_probs = {}
        for c in self.classes:
            c_mask = (y == c)
            class_features = X.loc[c_mask, categorical_columns]
            cat_probs = (class_features.apply(lambda x: x.value_counts(normalize=True)) + 1) / (len(class_features) + len(categorical_columns))
            self.cat_feature_probs[c] = cat_probs

    def predict(self, X: pd.DataFrame) -> list:
        """
        Melakukan prediksi kelas untuk setiap instance dalam dataset.
        Prediksi untuk numerik dengan gaussian pdf (desnity), dan prediksi untuk kategorikal dengan probabilitas kategorikal.

        Parameters:
        - X (pd.DataFrame): DataFrame berisi fitur-fitur dataset.

        Returns:
        - predictions (list): Daftar prediksi kelas untuk setiap instance.
        """
        predictions = []
        for _, instance in X.iterrows():
            class_probs = []
            for c in self.classes:
                num_probs = np.sum(np.log(self.gaussian_pdf(instance, self.num_feature_means[c], self.num_feature_stds[c])))
                cat_probs = np.sum(instance[categorical_columns].apply(lambda x: np.log(self.cat_feature_probs[c].get(x, 1))))
                class_prob = np.log(self.class_priors[c]) + num_probs + cat_probs
                class_probs.append(class_prob)
            predicted_class = self.classes[np.argmax(class_probs)]
            predictions.append(predicted_class)
        return predictions

    def gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """
        Menghitung nilai PDF (Probability Density Function) distribusi Gaussian.

        Parameters:
        - x (float): Nilai yang akan dihitung probabilitasnya.
        - mean (float): Rata-rata distribusi Gaussian.
        - std (float): Deviasi standar distribusi Gaussian.

        Returns:
        - pdf_value (float): Nilai PDF untuk input x.
        """

        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def save_model(self, filename):
        model_dict = {
            "class_priors": self.class_priors,
            "num_feature_means": self.num_feature_means,
            "num_feature_stds": self.num_feature_stds,
            "cat_feature_probs": self.cat_feature_probs,
            "classes": self.classes.tolist()
        }
        filename = "NaiveBayes/" + filename
        np.savez(filename, **model_dict, allow_pickle=True)

    def load_model(self, filename):
        filename = "NaiveBayes/" + filename
        model_data = np.load(filename, allow_pickle=True)
        self.class_priors = model_data["class_priors"].item()
        self.num_feature_means = model_data["num_feature_means"].item()
        self.num_feature_stds = model_data["num_feature_stds"].item()
        self.cat_feature_probs = model_data["cat_feature_probs"].item()
        self.classes = np.array(model_data["classes"])

if __name__ == "__main__":
    data_train = pd.read_csv("Data/" + sys.argv[1])
    data_validation = pd.read_csv("Data/" + sys.argv[2])
    X_train = data_train.drop("price_range", axis=1)
    y_train = data_train["price_range"]
    label_column = "price_range"
    if label_column in data_validation.columns:
        X_validation = data_validation.drop(label_column, axis=1)
        y_validation = data_validation[label_column]
    else:
        X_validation = data_validation
    numerical_columns = ["battery_power", "clock_speed", "fc", "int_memory", "m_dep", "mobile_wt", "n_cores", 
                        "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time"]
    categorical_columns = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train, numerical_columns, categorical_columns)
    model.save_model("model.npz")
    loaded_model = NaiveBayesClassifier()
    loaded_model.load_model("model.npz")
    predictions = loaded_model.predict(X_validation)
    if label_column in data_validation.columns:
        result_df = pd.DataFrame({"id": data_validation.index, "price_range": predictions})
    else:
        result_df = pd.DataFrame({"id": range(data_validation.shape[0]), "price_range": predictions})
    result_df.to_csv("Out/" + sys.argv[3], index=False)
