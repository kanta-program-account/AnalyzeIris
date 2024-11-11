import mglearn
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import DBSCAN

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline



class AnalyzeIris:
    
    """This class has some methods to analyze Iris dataset.
    
    Attributes:
    X (numpy.ndarray): Feature data of Iris dataset.
    y (numpy.ndarray): Target labels of Iris dataset.
    models (dict): Dictionary of classification models used for analysis.
    scalers (dict): Dictionary of scaling methods for feature normalization.
    df_TestScores (pandas.core.frame.DataFrame): DataFrame containing 5 test scores for each model.
    test_scores_mean (list): List of mean test score for each models.
    """
    
    def __init__(self): 
        """Initializes the analyzeIris with the Iris dataset.
        """
        #Load the Iris dataset.
        iris_dataset = load_iris()

        # Get a feature names.
        self.feature_names = iris_dataset.feature_names

        # Get a data and taraget from the Iris dataset.
        self.X, self.y = iris_dataset.data, iris_dataset.target

        # Shuffle the raw X and y.
        self.shuffle_X, self.shuffle_y = shuffle(self.X, self.y, random_state=0)

        # Prepare dictionaries, a dataframe, and an array. Use in practice2.
        self.models = {}
        self.feature_importances = {}
        self.df_test_scores_all_models = pd.DataFrame()
        self.mean_scores = []

        # Prepare a dictionary.Use in practice3.
        self.scalers = {}
        
        
        
    def Get(self) -> pd.DataFrame:
        """Retrieve the iris dataset as a DataFrame.

        Returns:
            pandas.core.frame.DataFrame: A DataFrame containing the feature data of the iris dataset.
        """

        # Define X and y as a raw data.
        X, y = self.X, self.y

        # Convert X to a dataframe.
        iris_dataframe = pd.DataFrame(X, columns=self.feature_names)

        # Add a new column and data 'y'.
        iris_dataframe["label"] = y

        return iris_dataframe # Need to return or print. df.head() only shows a portion of the dataframe as a preview.
    

    def PairPlot(self, cmap: str = 'viridis') -> None:
        """Show scatter plots for each feature in the iris dataset. The diagonal components are histograms.
        
        Args:
            cmap (str, optional): The color map used for the scatter plots. Defaults to 'viridis'.
        """

        # Define X and y as a raw data.
        X, y = self.X, self.y

        # Convert a raw data to a dataframe.
        iris_dataframe = pd.DataFrame(X, columns=self.feature_names)

        # Create a pairplot graph.
        grr = pd.plotting.scatter_matrix(iris_dataframe, c=y, figsize=(15, 15), marker='o',hist_kwds={'bins':20}, s=60, alpha=.8, cmap=cmap) # s: size of a marker.

    def AllSupervised(self, n_neighbors: int) -> None:
        """Show lists of 5 train and test scores for each model.

            KNeghiborsClassifier: The new data point is classified with the label that corresponds to the nearest neighbor.
            LinearRegression: Least squares method.

        Args:
            n_neighbors (int): Determine the n_neighbors as a parameter of KNeighborsClassifier model.
        """

        # Define X and y as shuffled.
        X, y = self.shuffle_X, self.shuffle_y

        # Define dictionary of models.
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=2000),
            'LinearSVC': LinearSVC(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=n_neighbors),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(random_state=0),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=0),
            'MLPClassifier': MLPClassifier(max_iter=2000)
        }

        # Prepare for performing cross-validation.
        kfold = KFold(n_splits=5, shuffle=False) # "kfold" is just defines how to divide the Iris dataset. train_test_split() cannnot evaluate train scores because this method cannot divide X_train. 

        # Initialize for each feature importances.
        tree_importances = np.zeros(X.shape[1])
        forest_importances = np.zeros(X.shape[1])
        gbrt_importances = np.zeros(X.shape[1])

        # Calucurate train scores of each models.（cross_val_score only calculate test scores.）
        for model_name, model in self.models.items():
            print(f"=== {model_name} ===")

            # Dataframe preparation for (n_splits) test scores of each models.
            df_test_scores_model = pd.DataFrame() 

            # Split train and test datasets according to Stratified K-Fold (SKF) rules.
            for train_index, test_index in kfold.split(X, y):
                
                # Convert a index to data.
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Train a model.
                model.fit(X_train, y_train)
                
                # Caluculate feature importances of each decsision tree models (to get the mean feature importances later).
                if model_name == 'DecisionTreeClassifier':
                    tree_importances += model.feature_importances_
                elif model_name == 'RandomForestClassifier':
                    forest_importances += model.feature_importances_
                elif model_name == 'GradientBoostingClassifier':
                    gbrt_importances += model.feature_importances_

                # Return test score.
                test_score = model.score(X_test, y_test)

                # Return train score.
                train_score = model.score(X_train, y_train) 

                # Output.
                print("test score: {:.3f}   ".format(test_score), "train score: {:.3f}".format(train_score)) 
                
                # Add the current model's test scores.
                df_test_scores_model = pd.concat([df_test_scores_model, pd.DataFrame([test_score])], ignore_index=True) # pd.concat(default: axis=0): Cobine dataframes.

            # Caluculate mean of current model.
            mean_score = df_test_scores_model.mean().item()
            self.mean_scores.append(mean_score)

            # Add the all model's test scores.
            self.df_test_scores_all_models = pd.concat([self.df_test_scores_all_models, df_test_scores_model], ignore_index=True, axis=1)

        # Get the mean feature importances
        tree_importances /= kfold.get_n_splits() 
        forest_importances /= kfold.get_n_splits()
        gbrt_importances /= kfold.get_n_splits()

        # Assign 
        self.feature_importances = {
            'DecisionTreeClassifier': tree_importances,
            'RandomForestClassifier': forest_importances,
            'GradientBoostingClassifier': gbrt_importances
        }
    
    def GetSupervised(self) -> pd.DataFrame:
        """Show dataframe of 5 test scores for each model.

        Returns:
            pandas.core.frame.DataFrame: Dataframe of 5 test scores for each model.
        """
        # Set a column for a dataframe.
        self.df_test_scores_all_models.columns = self.models.keys()

        return self.df_test_scores_all_models
    
    def BestSupervised(self) -> tuple[str, np.float64]:
        """Show the best model and the test score. 
        "The best model" means that the model resulted in the maximum mean test scores.

        Returns:
            best_model_name (str): The best model's name.
            best_score (numpy.float64): The best model's mean test score. #X, yがnumpy配列の為、meanの計算も基本的にnumpy演算となる。その為numpy.float64型を取る。
        """

        # Extract the max mean test score.
        best_score = max(self.mean_scores)

        # Extract the index of best score.
        best_method_index = self.mean_scores.index(best_score)

        # Extract the name of best model by using the index.
        best_model_name = list(self.models.keys())[best_method_index]
        
        return best_model_name, best_score
    
    def PlotFeatureImportancesAll(self) -> None:
        """Show bar graphs which refers feature importances for each tree classifier.
        """

        # Get the number of features.
        n_features = len(self.feature_names)

        # Create a graph.
        for model_name, importance in self.feature_importances.items():
            
            plt.barh(range(n_features), importance, align='center')
            plt.yticks(np.arange(n_features), self.feature_names)
            plt.xlabel(f"Feature importance : {model_name}")
            plt.show()
            
    def VisualizeDecisionTree(self) -> graphviz.sources.Source:
        """Display the classification process in a tree-shaped format.

        Returns:
            graphviz.sources.Source: This is the classification process that has been visualized.
        """

        # Define X and y as a raw data. 
        X, y = self.X, self.y

        # Split X and y. 
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=0) # "kfold" used in AllSupervised() method cannot be used because each split of X_train and y_train is too small.

        # Define a model.
        tree = DecisionTreeClassifier(random_state=0)

        # Fit a model.
        tree.fit(X_train, y_train)

        # Export the tree model for a 'dot' file using graphviz.
        export_graphviz(tree, out_file="tree.dot", class_names=["L", "a", "b"], feature_names=self.feature_names, impurity=False, filled=True)

        # Read a tree.dot file.
        with open("tree.dot") as f:
            dot_graph = f.read()

        return graphviz.Source(dot_graph)
    
    def PlotScaledData(self) -> None:
        """Show a scaled data plot for each scaling model and each feature combination. 
        """
        # Define X and y as shuffled.
        X, y = self.shuffle_X, self.shuffle_y
        
        # Define dictionary of scalers.
        self.scalers = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
        }
        
        # Define the number of splits and features.
        kfold = 5
        number_of_features = len(self.feature_names)
        
        # Execute cross-validation for original data.
        cv_results = cross_validate(LinearSVC(), X, y, cv=kfold, return_train_score=True, return_indices=True)
        
        # --Plot "scaled data" after obtaining 5-fold train and test scores.--
        
        for i in range(kfold): # i:split index
            
            # # Create a subplots.
            fig, axes = plt.subplots(4, 5, figsize=(20, 20))
            
            # Extract a original data from cross_validate().
            X_train = X[cv_results["indices"]["train"][i]]
            X_test = X[cv_results["indices"]["test"][i]]
            
            # Output.
            print("{:<17}: test score: {:.3f}     train score: {:.3f}".format("Original", cv_results['test_score'][i], cv_results['train_score'][i]))

            for j, (scaler_name, scaler) in enumerate(self.scalers.items()):  # j:scaler index
            
                # Create a pipeline that scales the data and fits a model.
                pipeline = make_pipeline(scaler, LinearSVC())
                
                # Execute cross-validation for scaled data.
                scaled_cv_results = cross_validate(pipeline, X, y, cv=5, return_train_score=True, return_indices=True)

                # Extract a scaled data from cross_validate().
                X_train_scaled = scaler.fit_transform(X[scaled_cv_results["indices"]["train"][i]])
                X_test_scaled = scaler.fit_transform(X[scaled_cv_results["indices"]["test"][i]]) 

                print("{:<17}: test score: {:.3f}     train score: {:.3f}".format(scaler_name, scaled_cv_results['test_score'][i], scaled_cv_results['train_score'][i]))       

                for k in range(number_of_features): # k:feature index
                    # Replace rows and columns based on indexes.
                    row = k
                    column = j+1           
                        
                    if k != number_of_features-1:
                        if j == 0:
                            # Create a original data plot.
                            axes[row, 0].scatter(X_train[:, k], X_train[:, k+1], label="Training set")
                            axes[row, 0].scatter(X_test[:, k], X_test[:, k+1], marker='^', label="Test set")
                            axes[row, 0].legend(loc='upper left')
                            axes[row, 0].set_title("Original data")
                            axes[row, 0].set_xlabel(self.feature_names[k])
                            axes[row, 0].set_ylabel(self.feature_names[k+1])
                        
                        # Create a scaled data plots.
                        axes[row, column].scatter(X_train_scaled[:, k], X_train_scaled[:, k+1], label="Training set")
                        axes[row, column].scatter(X_test_scaled[:, k], X_test_scaled[:, k+1], marker='^', label="Test set")
                        axes[row, column].legend(loc='upper left')
                        axes[row, column].set_title(f"{scaler_name}")
                        axes[row, column].set_xlabel(self.feature_names[k])
                        axes[row, column].set_ylabel(self.feature_names[k+1])
                    else:
                        if j == 0:
                            # Create a original data plot.
                            axes[row, 0].scatter(X_train[:, k], X_train[:, 0], label="Training set")
                            axes[row, 0].scatter(X_test[:, k], X_test[:, 0], marker='^', label="Test set")
                            axes[row, 0].legend(loc='upper left')
                            axes[row, 0].set_title("Original data")
                            axes[row, 0].set_xlabel(self.feature_names[k])
                            axes[row, 0].set_ylabel(self.feature_names[0])

                        # Create a scaled data plots.
                        axes[row, column].scatter(X_train_scaled[:, k], X_train_scaled[:, 0], label="Training set")
                        axes[row, column].scatter(X_test_scaled[:, k], X_test_scaled[:, 0], marker='^', label="Test set")
                        axes[row, column].legend(loc='upper left')
                        axes[row, column].set_title(f"{scaler_name}")
                        axes[row, column].set_xlabel(self.feature_names[k])
                        axes[row, column].set_ylabel(self.feature_names[0])    
        
        # Adjust a plots.
        plt.tight_layout()
        plt.show()
    
    def PlotFeatureHistgram(self) -> None:
        """Show histgrams of the frequency for each feature in various range. 
        """
        X, y = self.X, self.y
        feature_names = self.iris_dataset.feature_names
        target_names = self.iris_dataset.target_names  # クラス名を修正
        colors = ['blue', 'red', 'green']  # クラスごとの色を指定
        
        # 各特徴量ごとにヒストグラムを描画
        for i in range(X.shape[1]):  # 4つの特徴量に対してループ
            plt.figure(figsize=(10, 6))  # 図のサイズ設定
            for j, color in enumerate(colors):  # クラスごとのループ
                plt.hist(X[y == j, i], bins=20, alpha=0.5, color=color, label=target_names[j])
            
            plt.xlabel(feature_names[i])  # x軸ラベル
            plt.ylabel("Frequency")  # y軸ラベル
            plt.legend()  # 凡例を表示
            plt.show()  # 各特徴量ごとに表示
    
    def PlotPCA(self, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, PCA]: # tuple: 変更できない配列。 list: 変更可能な配列。
        """Show  scatter plot of PCA scaled iris_data. 

        Args:
            n_components (int): Determine the numbar of dimensions as a parameter of PCA.

        Returns:
            X_scaled (pandas.core.frame.DataFrame): Dataframe of scaled iris data.
            df_pca (pandas.core.frame.DataFrame): Dataframe of scaled iris data after dimensionality reduction.
            pca (sklearn.decomposition._pca.PCA): A fitted PCA model that contains attributes to explain the results.
        """
        X, y = self.X, self.y
        feature_names = self.iris_dataset.feature_names
        
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)#スケール変換の適用
        
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        X_pca = pca.transform(X_scaled)
        
        df_pca = pd.DataFrame(X_pca)
        
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], y)
        plt.legend(self.iris_dataset.target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        
        plt.matshow(pca.components_, cmap='viridis')
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("principal components")
        
        return X_scaled, df_pca, pca
    
    def PlotNMF(self, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """Show  scatter plot of NMF scaled iris_data. 

        Args:
            n_components (_type_): Determine the numbar of components as a parameter of NMF.

        Returns:
            X_scaled_nmf (pandas.core.frame.DataFrame): Dataframe of scaled iris data.
            df_nmf (pandas.core.frame.DataFrame): Dataframe of scaled iris data after component extraction.
            nmf (sklearn.decomposition._pca.PCA): A fitted NMF model.
        """
        X, y = self.X, self.y
        feature_names = self.iris_dataset.feature_names
        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        
        # scaler = MinMaxScaler() #x軸, y軸いずれも0~1に収まるようにスケーリングされる。
        # scaler.fit(X)
        # X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)#スケール変換の適用
        
        nmf = NMF(n_components=n_components, random_state=0) #取り出される成分と選ばれ方がランダム
        nmf.fit(X) #X_trainをもとに成分を抽出。 nmfには成分の影響力に順番は無い。全て同等。
        X_scaled_nmf = nmf.transform(X) #X_trainをもとに選択された成分で、X_trainを再構成。
        
        df_nmf = pd.DataFrame(X_scaled_nmf)
        
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(X_scaled_nmf[:, 0], X_scaled_nmf[:, 1], y)
        plt.legend(self.iris_dataset.target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        
        plt.matshow(nmf.components_, cmap='viridis')
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("principal components")
        
        return X_scaled_nmf, df_nmf, nmf
    
    def PlotTSNE(self) -> None: #ラベルを使わずに、分類出来る。似ているものが近くに、違うものは遠くに配置するアルゴリズム。
        """Show t-SNE clustering. 

        This algorithm clusters the data without using labels, placing similar items close together and dissimilar items far apart.
        """
        X, y = self.X, self.y
        
        tsne = TSNE(random_state=42)
        iris_tsne = tsne.fit_transform(X)
        
        colors = ["#476A2A", "#7851B8", "#BD3430"]
        
        plt.figure(figsize=(10, 10))
        plt.xlim(iris_tsne[:, 0].min(), iris_tsne[:, 0].max() + 1)
        plt.ylim(iris_tsne[:, 1].min(), iris_tsne[:, 1].max() + 1)
        for i in range(len(X)):
            plt.text(iris_tsne[i, 0], iris_tsne[i, 1], str(y[i]), color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        
    def PlotKMeans(self) -> None:
        """Show kmeans clustering.

        This algorithm clusters the data without using labels, placing cluster centers by calculating the mean of the data.
        """
        X, y = self.X, self.y
        
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        
        print("KMeans法で予測したラベル:\n{}".format(kmeans.labels_))
        
        mglearn.discrete_scatter(X[:, 2], X[:, 3], kmeans.labels_, markers='o')
        mglearn.discrete_scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], [0, 1, 2], markers='^', markeredgewidth=2, c=['white','white','white'])
        plt.xlabel("petal length (cm)")
        plt.ylabel("petal width (cm)")
                
        print("実際のラベル:\n{}".format(y))
        
    def PlotDendrogram(self, truncate :bool=False) -> None:
        """Show dendrogram which visualizes the clustering process. 

        Args:
            truncate (bool, optional): Determine the type of dendrogram, original or reduced. Defaults to False.
        """
        X, y = self.X, self.y
        
        linkage_array = ward(X) # ward:凝集型クラスタリング クラスタ間距離を返す
        if truncate:
            dendrogram(linkage_array, p=10, truncate_mode='lastp')   # 最後に形成された10個のクラスタを表示
        else:
            dendrogram(linkage_array)  # 完全なデンドログラムを表示
        
        fig = plt.gcf()
        fig.set_size_inches(12, 8)  # 幅10インチ、高さ8インチ
        
        ax = plt.gca()
        bounds = ax.get_xbound()
        ax.plot(bounds, [10, 10], '--', c='k')
        ax.plot(bounds, [5.5, 5.5], '--', c='k')

        ax.text(bounds[1], 10, ' 3 clusters', va='center', fontdict={'size': 15})
        ax.text(bounds[1], 5.5, ' 4 clusters', va='center', fontdict={'size': 15})
        plt.xlabel("Sample")
        plt.ylabel("ClusterDistance")
        plt.show()
        
    def PlotDBSCAN(self, scaling :bool=False, eps :int=0.9, min_samples :int=5) -> None:
        """Show DBSCAN clustering.

        This algorism clusters the data based on density .

        Args:
            scaling (bool, optional): Determine whether to scale or not. Defaults to False.
            eps (float, optional): The parameter which decides the range of the same cluster. Defaults to 0.9.
            min_samples (int, optional): The parameter which decides the number of data points in one cluster. Defaults to 5.
        """
        X, y = self.X, self.y
        
        if scaling:
            scaler = StandardScaler()
            scaler.fit(X)
            X_scaled = scaler.transform(X)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        print("Cluster Memberships:\n{}".format(clusters))
        
        plt.scatter(X[:, 2], X[:, 3], c=clusters)
        plt.xlabel("Feature2")
        plt.ylabel("Feature3")