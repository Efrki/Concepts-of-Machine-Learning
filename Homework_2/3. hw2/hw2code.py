import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin



class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def get_params(self, deep=True):
        return {
            'feature_types': self.feature_types,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _find_best_split(self, feature_vector, target_vector):
        # Преобразуем в numpy массивы для корректной работы с индексами
        feature_vector = np.array(feature_vector)
        target_vector = np.array(target_vector)
        
        sorted_indices = np.argsort(feature_vector)
        feature_sorted = feature_vector[sorted_indices]
        target_sorted = target_vector[sorted_indices]
        
        unique_features = np.unique(feature_sorted)
        if len(unique_features) < 2:
            return np.array([]), np.array([]), None, None
        
        thresholds = (unique_features[:-1] + unique_features[1:]) / 2.0
        
        n = len(target_sorted)
        split_indices = np.searchsorted(feature_sorted, thresholds, side='right')
        
        valid_mask = (split_indices > 0) & (split_indices < n)
        valid_indices = split_indices[valid_mask]
        valid_thresholds = thresholds[valid_mask]
        
        if len(valid_indices) == 0:
            return np.array([]), np.array([]), None, None
        
        cum_1 = np.cumsum(target_sorted == 1)
        cum_0 = np.cumsum(target_sorted == 0)
        
        n_left = valid_indices
        n_right = n - n_left
        
        p1_left = cum_1[valid_indices - 1] / n_left
        p0_left = cum_0[valid_indices - 1] / n_left
        h_left = 1 - p1_left**2 - p0_left**2
        
        p1_right = (cum_1[-1] - cum_1[valid_indices - 1]) / n_right
        p0_right = (cum_0[-1] - cum_0[valid_indices - 1]) / n_right
        h_right = 1 - p1_right**2 - p0_right**2
        
        ginis_valid = -(n_left/n) * h_left - (n_right/n) * h_right
        
        best_idx = np.argmax(ginis_valid)
        threshold_best = valid_thresholds[best_idx]
        gini_best = ginis_valid[best_idx]
        
        return thresholds, ginis_valid, threshold_best, gini_best

    def _fit_node(self, sub_X, sub_y, node, depth):
        # Преобразуем в numpy массивы
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        
        # Stopping conditions
        is_pure = np.all(sub_y == sub_y[0])
        depth_exceeded = self.max_depth is not None and depth >= self.max_depth
        min_samples_split_reached = self.min_samples_split is not None and len(sub_y) < self.min_samples_split

        if is_pure or depth_exceeded or min_samples_split_reached:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types_[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                # Для категориальных признаков просто используем их как есть
                feature_vector = sub_X[:, feature].astype(float)
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = self._find_best_split(feature_vector, sub_y)

            # Если разделение не найдено, пропускаем признак
            if threshold is None:
                continue

            # Check min_samples_leaf condition
            if self.min_samples_leaf is not None and threshold is not None:
                left_count = np.sum(feature_vector < threshold)
                right_count = len(feature_vector) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

            if gini is not None and (gini_best is None or gini > gini_best):
                gini_best = gini
                feature_best = feature
                threshold_best = threshold
                if feature_type == "real":
                    split = feature_vector < threshold
                elif feature_type == "categorical":
                    split = feature_vector < threshold
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types_[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types_[feature_best] == "categorical":
            node["threshold"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        
        # Если feature_types не задан (например, при OHE), считаем все признаки вещественными
        if not self.feature_types_ or self.feature_types_[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:  # categorical
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        # Преобразуем в numpy массивы
        X = np.array(X)
        y = np.array(y)
        
        # Создаем "обученный" атрибут feature_types_
        if self.feature_types and len(self.feature_types) > 0:
            self.feature_types_ = self.feature_types
        else: # Если исходный список пуст (как в Pipeline)
            self.feature_types_ = ['real'] * X.shape[1]

        if np.any(list(map(lambda x: x != "real" and x != "categorical", self.feature_types_))):
            raise ValueError("There is unknown feature type")
        self.tree_ = {}
        self.classes_ = np.unique(y)
        self._fit_node(X, y, self.tree_, depth=0)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self.tree_))
        return np.array(predicted)
