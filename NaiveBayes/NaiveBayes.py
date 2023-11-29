import numpy as np
import pandas as pd
import sys

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.num_feature_means = None
        self.num_feature_stds = None
        self.cat_feature_probs = None
        self.classes = None

    def fit(self, X, y, numerical_columns, categorical_columns):
        self.class_priors = {c: np.sum(y == c) / len(y) for c in np.unique(y)}
        self.classes = np.unique(y)
        self.num_feature_means = {}
        self.num_feature_stds = {}
        for c in self.classes:
            c_mask = (y == c)
            class_features = X.loc[c_mask, numerical_columns]
            self.num_feature_means[c] = class_features.mean()
            self.num_feature_stds[c] = class_features.std()
        self.cat_feature_probs = {}
        for c in self.classes:
            c_mask = (y == c)
            class_features = X.loc[c_mask, categorical_columns]
            cat_probs = (class_features.apply(lambda x: x.value_counts(normalize=True)) + 1) / (len(class_features) + len(categorical_columns))
            self.cat_feature_probs[c] = cat_probs

    def predict(self, X):
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

    def gaussian_pdf(self, x, mean, std):
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
