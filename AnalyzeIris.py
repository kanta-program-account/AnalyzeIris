import mglearn
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import DBSCAN





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

        # Get a target names.
        self.target_names = iris_dataset.target_names

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

        # Initialize feature importances.
        tree_importances = 0
        forest_importances = 0
        gbrt_importances = 0

        # Define dictionary of eature importances.
        self.importances_map = {
            'DecisionTreeClassifier': tree_importances,
            'RandomForestClassifier': forest_importances,
            'GradientBoostingClassifier': gbrt_importances
            }

        # Define the number of splits.
        kfold = 5

        # Calucurate train scores of each models.（cross_val_score only calculate test scores.）
        for model_name, model in self.models.items():
            print(f"=== {model_name} ===")

            # Dataframe preparation for (n_splits) test scores of each models.
            df_test_scores_model = pd.DataFrame() 

            # Execute cross-validation for original data.
            cv_results = cross_validate(model, X, y, cv=kfold, return_train_score=True, return_indices=True, return_estimator=True)

            # Split train and test datasets according to Stratified K-Fold (SKF) rules.
            for fold_index in range(kfold):

                # Replace.
                test_score_fold = cv_results['test_score'][fold_index]
                train_score_fold = cv_results['train_score'][fold_index]

                # Reassignment a fitted model.
                model = cv_results['estimator'][fold_index]

                # Output.
                print("test score: {:.3f}   ".format(test_score_fold), "train score: {:.3f}".format(train_score_fold)) 
                
                # Add the current model's test scores.
                df_test_scores_model = pd.concat([df_test_scores_model, pd.DataFrame([test_score_fold])], ignore_index=True) # pd.concat(default: axis=0): Cobine dataframes.

                # Caluculate feature importances (to get the mean of them later).
                if model_name in self.importances_map:
                    self.importances_map[model_name] += model.feature_importances_

            # Caluculate mean of current model.
            self.mean_scores.append(df_test_scores_model.mean().item())

            # Add to the DataFrame that stores all the model's test scores.
            self.df_test_scores_all_models = pd.concat([self.df_test_scores_all_models, df_test_scores_model], ignore_index=True, axis=1)

        # Get the mean of feature importances.
        for importances in self.feature_importances:
            importances /= kfold
    
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

        print(self.importances_map)

        # Create a graph.
        for model_name, importance in self.importances_map.items():
            
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=0) # "fold" used in AllSupervised() method cannot be used because each split of X_train and y_train is too small.

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
        
        def ScatterPlot(self, axes, row, column, X_train, X_test, index_feature_xaxis, index_feature_yaxis, title) -> None:
            """Helper function to create a scatter plot on given axes with training and test data.
            """
            axes[row, column].scatter(X_train[:, index_feature_xaxis], X_train[:, index_feature_yaxis], label="Training set")
            axes[row, column].scatter(X_test[:, index_feature_xaxis], X_test[:, index_feature_yaxis], marker='^', label="Test set")
            axes[row, column].legend(loc='upper left')
            axes[row, column].set_title(f"{title}")
            axes[row, column].set_xlabel(self.feature_names[index_feature_xaxis])
            axes[row, column].set_ylabel(self.feature_names[index_feature_yaxis]) 
        
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
        
        for fold_index in range(kfold): # i:fold index
            
            # Create subplots.
            fig, axes = plt.subplots(4, 5, figsize=(20, 20))
            
            # Extract a original data from cross_validate().
            X_train = X[cv_results["indices"]["train"][fold_index]]
            X_test = X[cv_results["indices"]["test"][fold_index]]
            
            # Output.
            print("{:<17}: test score: {:.3f}     train score: {:.3f}".format("Original", cv_results['test_score'][fold_index], cv_results['train_score'][fold_index]))

            # Plot an original data.
            for index_feature_xaxis in range(number_of_features): # index_feature_xaxis: The feature index of the x-axis.
                # Calculate the feature index of the y-axis.
                index_feature_yaxis = index_feature_xaxis+1 if (index_feature_xaxis != number_of_features-1) else 0         
                
                # Replace.
                row = index_feature_xaxis
                
                # Create a original data plot.
                ScatterPlot(self, axes=axes, row=row, column=0, X_train=X_train, X_test=X_test, index_feature_xaxis=index_feature_xaxis, index_feature_yaxis=index_feature_yaxis, title="Original")
            
            for column_axes, (scaler_name, scaler) in enumerate(self.scalers.items()):
            
                # Create a pipeline that scales the data and fits a model.
                pipeline = make_pipeline(scaler, LinearSVC())
                
                # Execute cross-validation for scaled data.
                scaled_cv_results = cross_validate(pipeline, X, y, cv=5, return_train_score=True, return_indices=True)

                # Extract a scaled data from cross_validate().
                X_train_scaled = scaler.fit_transform(X[scaled_cv_results["indices"]["train"][fold_index]])
                X_test_scaled = scaler.transform(X[scaled_cv_results["indices"]["test"][fold_index]]) 

                print("{:<17}: test score: {:.3f}     train score: {:.3f}".format(scaler_name, scaled_cv_results['test_score'][fold_index], scaled_cv_results['train_score'][fold_index]))       

                for index_feature_xaxis in range(number_of_features): # index_feature_xaxis: The feature index of the x-axis.
                    # Calculate the feature index of the y-axis.
                    index_feature_yaxis = index_feature_xaxis+1 if (index_feature_xaxis != number_of_features-1) else 0         
                    
                    # Replace.
                    row = index_feature_xaxis
                    column = column_axes+1           
                    
                    # Create a scaled data plots.
                    ScatterPlot(self, axes, row, column, X_train_scaled, X_test_scaled, index_feature_xaxis, index_feature_yaxis, scaler_name)
        
            # Display the current figure and then print the results for the fold.
            plt.tight_layout()
            plt.show()
            print("=" * 73)

    def PlotFeatureHistgram(self) -> None:
        """Show histgrams of the frequency for each feature in various range. 
        """
        # Define X and y as a raw data.
        X, y = self.X, self.y

        # Define a list of color.
        colors = ['blue', 'red', 'green']
        
        # Plot a histogram for each feature.
        for feature in range(len(self.feature_names)):

            # Create subplots.
            plt.figure(figsize=(10, 6))

            # Plot histograms for each target, with the same bins for all.
            # Find the minimum and maximum values of the feature across all targets.
            min_value = X[:, feature].min()
            max_value = X[:, feature].max()

            # Set bins.
            bins = 50  # Fixed number of bins for all targets

            # Plot the histogram with the same bins across all target labels.
            for target, color in zip(range(len(self.target_names)), colors):
                # Select data corresponding to the current target and plot it.
                plt.hist(X[y == target, feature], bins=bins, alpha=0.5, color=color, label=self.target_names[target], range=(min_value, max_value))
            
            plt.xlabel(self.feature_names[feature])
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()
    
    def PlotPCA(self, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, PCA]: # tuple: A non-modifiable array. list: A modifiabl array.
        """Show  scatter plot of PCA scaled iris_data. 

        Args:
            n_components (int): Determine the numbar of dimensions as a parameter of PCA.

        Returns:
            X_scaled (pandas.core.frame.DataFrame): Dataframe of scaled iris data.
            df_pca (pandas.core.frame.DataFrame): Dataframe of scaled iris data after dimensionality reduction.
            pca (sklearn.decomposition._pca.PCA): A fitted PCA model that contains attributes to explain the results.
        """
        # Define X and y as a raw data.
        X, y = self.X, self.y
        
        # Initialize the StandardScaler.
        scaler = StandardScaler()

        # Fit a scaler to X and transform.
        X_scaled = scaler.fit_transform(X)
        
        # Initialize a PCA.
        pca = PCA(n_components=n_components, random_state=0)

        # Fit a PCA to X_scaled data and transform using components extracted by PCA.
        X_pca = pca.fit_transform(X_scaled)
        
        # Create a dataframe.
        df_X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Create a dataframe.
        df_pca = pd.DataFrame(X_pca)
    
        # Create a plot.
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], y)
        plt.legend(self.target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        
        # Create a heat map.
        plt.matshow(pca.components_, cmap='viridis')
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)),self.feature_names, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("principal components")

        return df_X_scaled, df_pca, pca
    
    def PlotNMF(self, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """Show  scatter plot of NMF scaled iris_data. 

        Args:
            n_components (_type_): Determine the numbar of components as a parameter of NMF.

        Returns:
            X_scaled_nmf (pandas.core.frame.DataFrame): Dataframe of scaled iris data.
            df_nmf (pandas.core.frame.DataFrame): Dataframe of scaled iris data after component extraction.
            nmf (sklearn.decomposition._pca.PCA): A fitted NMF model.
        """
        # Define X and y as a raw data.
        X, y = self.X, self.y
        
        # Initialize a NMF.
        nmf = NMF(n_components=n_components, random_state=0)

        # Fit a NMF to X_scaled data and transform using components extracted by NMF.
        X_nmf = nmf.fit_transform(X)

        # Create a dataframe.
        df_X = pd.DataFrame(X, columns=self.feature_names)

        # Create a dataframe.
        df_nmf = pd.DataFrame(X_nmf)
        
        # Create a plot.
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(X_nmf[:, 0], X_nmf[:, 1], y)
        plt.legend(self.target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        
        # Create a heat map.
        plt.matshow(nmf.components_, cmap='viridis')
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)),self.feature_names, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("principal components")
        
        return df_X, df_nmf, nmf
    
    def PlotTSNE(self) -> None:
        """Show t-SNE clustering. 

        This algorithm clusters the data without using labels, placing similar items close together and dissimilar items far apart.
        """
        # Define X and y as a raw data.
        X, y = self.X, self.y
        
        # Initialize a TSNE.
        tsne = TSNE(random_state=0) 

        # Fit a TSNE and tranform.
        X_tsne = tsne.fit_transform(X)
        
        # Define a list of color.
        colors = ["#476A2A", "#7851B8", "#BD3430"]
        
        # Create a plot.
        plt.figure(figsize=(10, 10))
        plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
        plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
        for i in range(len(X)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]), color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        
    def PlotKMeans(self) -> None:
        """Show kmeans clustering.

        This algorithm clusters the data without using labels, placing cluster centers by calculating the mean of the data.
        """

        def ScatterPlot(self, cluster_labels) -> None:
            """Helper function to create a scatter plot on given axes with training and test data.
            """
            # Create a plot.
            for label, (marker, color) in enumerate(zip(markers, colors)):

                # Plot a data of each label.
                points = X[cluster_labels == label]
                mglearn.discrete_scatter(points[:, 2], points[:, 3], label, markers=marker, c=[color]*len(points))

            # Plot a cluster center.
            mglearn.discrete_scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], label, markers='^', markeredgewidth=2, c=['white']*len(self.target_names))

            # Set a graph in detail.
            plt.xlim(X[:, 2].min() - .2, X[:, 2].max() + .2)
            plt.ylim(X[:, 3].min() - .2, X[:, 3].max() + .2)
            plt.xlabel("petal length (cm)")
            plt.ylabel("petal width (cm)")
            plt.show()

        # Define X and y as a raw data.
        X, y = self.X, self.y

        # feature_xaxis = 
        
        # Initialize a kmeans.
        kmeans = KMeans(n_clusters=3, random_state=0)

        # Fit a kmeans.
        kmeans.fit(X)

        # Define a list of marker and color.
        markers = ["o", "*", "D"]
        colors = ['blue', 'orange', 'green']
        
        # Show predeicted clustering labels.
        print("KMeans法で予測したラベル:\n{}".format(kmeans.labels_)) 

        # Create a plot.
        ScatterPlot(self, kmeans.labels_)

        # Show correct clustering labels. 
        print("実際のラベル:\n{}".format(y))

        # Create a plot.
        ScatterPlot(self, kmeans.labels_)
        
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