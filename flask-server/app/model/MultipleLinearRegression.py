# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultipleLinearRegression:
    """Class for a multiple linear regression model.

    Extended Summary
    ----------------
    The model is desig\ned to be trained to predict multiple target variables using
    the same input features Hence, the model stores separate sets of parameter values
    for the prediction of each target variable.
    General user flow:
    1) Store data (expects pandas DataFrames) using store_data().
    2) Train the model to predict the specified target variable using train().
    3) Validate the model's performance using validate().

    Attributes
    ----------
    params : dict
        Dictionary that stores the different sets of parameter values for different
        target variables. The key is the name of a target variable (str) while
        the value is a numpy.ndarray containing the parameter values. New key-value
        pairs (i.e. new parameters) are added or updated when the model is trained to
        predict a specified target variable. Each numpy.ndarray (set of parameter values)
        has a shape of (n+1,1) where n is the number of features.
    features, targets : list of str
        A list of strings containing the column names of the features and targets respectively.
        Initialised when data is stored in the instance using store_data().
    train_features, test_features, train_targets, test_targets : pandas.DataFrames
        Pandas Dataframes containing the features/targets for training/testing. Initialised
        when data is stored in the instance using store_data().
    means, stds : float
        The mean and standard deviation of the training features, initialised during training.
        Used for normalization of input features for model predictions.

    Methods
    -------
    store_data(data, feature_names, target_names, random_state=None, test_size=0.5)
        Takes the given data and separates the features and targets. Then splits the data
        for training and testing before storing it in the model's attributes.
    add_transformed_feature(feature_name, new_feature_name, transform_func, replace=False)
        Transforms a specified feature and adds it to the stored data.
    train(target, alpha=0.01, epochs=1000, retrain=False, convergence_threshold=1e-6, show=True)
        Trains the model to predict the specified target variable with the input features. Stores
        the parameters in the model.
    predict(df_features, target)
        Given some input data of features, the model returns its predictions for the specified target
        variable using linear regression (with its stored parameters).
    validate(target, visualize=True)
        Evaluate the model's ability to predict the specified target variable (its performance) using the
        test data. Will show visualizations of the predicted values against the true values for the test data
        and return various performance metrics.
    """

    def __init__(self):
        """Constructs a new instance of MultipleLinearRegression"""

        self.params = {} # dictionary of weights (numpy arrays)
        self.features, self.targets = None, None # list of feature/target column names
        self.train_features, self.train_targets = None, None
        self.test_features, self.test_targets = None, None
        self.means, self.stds = None, None # for normalization (initialised during training)


    def store_data(self, data, feature_names, target_names, random_state=None, test_size=0.5):
        """Prepares and stores data in the model (instance).

        Extended Summary
        ----------------
        Will split the dataset into two: one portion for training and the other for testing (model validation).
        The features and targets for training/testing are stored in the instance as attributes.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame that contains all the relevant data (features and targets).
        feature_names, target_names : list
            Lists containing the column names for the features and targets respectively.
        random_state : int, optional
            Integer representing a seed for splitting of the data into training and test sets.
            Setting a seed to a specific value ensures reproducibility (default is None, which means
            that no seed will be specified and data splitting will be random).
        test_size : float, default 0.5
            Float value between 0 and 1 that represents the fraction of the data that will be in the test set.
        """

        # Split data into features and targets
        self.features = feature_names
        self.targets = target_names
        df_features = data.loc[:, feature_names]
        df_targets = data.loc[:, target_names]

        # Split data into training and testing sets, then store in instance
        self.train_features, self.test_features, self.train_targets, self.test_targets = self._split_data(df_features, df_targets, random_state, test_size)


    def add_transformed_feature(self, feature_name, new_feature_name, transform_func, replace=False):
        """Takes a feature in the stored data and stores a transformed version of it.

        Extended Summary
        ----------------
        Transforms the specified feature (already stored in the model) by applying the transformation
        function on each feature value. The new transformed feature values can be stored as a new
        feature or can replace the original feature which was transformed. Important to note: the model
        should be retrained after adding a transformed feature.

        Parameters
        ----------
        feature_name : str
            Column name of the feature to be transformed. The feature should exist in the model's
            stored data.
        new_feature_name : str
            The column name of the transformed feature to be added.
        transform_func:
            The name of the transformation function i.e. the function to be applied to the specified feature
            values to transform them.
        replace : bool, default False
            Determines whether transformed features will replace the original feature or be added as a separate
            feature. (Default is False, so the transformed features will be added as a new column with the
            column name specified with parameter new_feature_name).
        """

        if feature_name in self.features:
            # add new column of transformed features to both training and test features
            train_features_transformed = self.train_features[feature_name].apply(transform_func)
            test_features_transformed = self.test_features[feature_name].apply(transform_func)

            if replace == True:
                # Replace data:
                self.train_features[feature_name] = train_features_transformed
                self.test_features[feature_name] = test_features_transformed

                # Rename:
                self.train_features.rename(columns={feature_name: new_feature_name}, inplace=True)
                self.test_features.rename(columns={feature_name: new_feature_name}, inplace=True)
                self.features = [new_feature_name if item == feature_name else item for item in self.features]

            else:
                # Add as new column
                self.train_features[new_feature_name] = train_features_transformed
                self.test_features[new_feature_name] = test_features_transformed
                self.features.append(new_feature_name)
        else:
            raise ValueError(f"No feature called {feature_name} in stored data.")


    def train(self, target, alpha=0.01, epochs=1000, retrain=False, convergence_threshold=1e-6, show=True):
        """Train the model to predict the specified target variable (using the stored training data).

        Extended Summary
        ----------------
        Performs batch gradient descent with training data stored using the store_data method.
        Updates the 'param' attribute with new parameter values. Cost is defined as half the mean squared error.

        Parameters
        ----------
        target : str
            Column name of target variable to be predicted.
        alpha : float, default 0.01
            Learning rate for gradient descent.
        epochs : int, default 1000
            Number of gradient descent steps.
        retrain : bool, default False
            If True, reinitialize the model parameters. (Default is False, meaning
            training will start using current parameter values if they exist)
        convergence_threshold : float, default 1e-6
            Threshold for convergence based on the change in cost. Training will end
            prematurely if difference in cost between iterations is less than the threshold.
        show : bool, default True
            If True, will show change in cost during training.

        Returns
        -------
        self.params : numpy.ndarray
            NumPy array containing parameter values.
        J_storage : numpy.ndarray
            NumPy array containing the cost after each iteration of gradient descent.
        """

        # Check that there is data stored for training and validation:
        if any(data is None for data in (self.train_features, self.train_targets, self.test_features, self.test_targets)):
            raise ValueError("Data missing. Run the store_data() method first.")
        # Check that valid target name is specified:
        elif target not in self.targets:
            raise ValueError("Specified target not found in stored data.")


        # Normalize training features:
        train_features_norm, self.means, self.stds = self._normalize_z(self.train_features)

        # Prepare data in numpy arrays:
        X_train = self._prepare_X(train_features_norm)
        y_train = self._prepare_y(self.train_targets.loc[:, [target]])
        m, n = X_train.shape  # m = number of training examples, n = number of features (including intercept term)

        # Validate parameters
        params = self.params.setdefault(target, np.zeros((n,1)) ) # initialize if not exist
        if retrain or params.shape[0] != n: # check if dimensions match
            self.params[target] = np.zeros((n,1)) # reset

        # Gradient Descent
        J_storage = np.zeros(epochs)
        if show: print("TRAINING START")
        for i in range(epochs):
            y_hat = self._calc_linreg(X_train, self.params[target])
            err = y_hat - y_train

            # Measure cost J in each iteration (epoch):
            J = np.matmul(err.T, err) / (2*m)
            J_storage[i] = J[0][0] # scalar value of (1,1) array

            # Update parameters (according to partial derivatives of cost function J):
            deriv = np.matmul(X_train.T, err) / m
            self.params[target] = self.params[target] - alpha * deriv  # update stored parameters

            if show: print(f"Epoch {i + 1:<4} || Cost: {J[0][0]:.5f}")
            # Check for convergence:
            if i > 0 and abs(J_storage[i] - J_storage[i-1]) < convergence_threshold:
                print(f"Converged (threshold = {convergence_threshold}). Stopping training.")
                break

        if show:
            print("TRAINING END")
            plt.figure(figsize=(5, 3)) # size of plot
            sns.lineplot(x=range(1, len(J_storage) + 1), y=J_storage)
            plt.xlabel("Epochs")
            plt.ylabel("Cost (J)")
            plt.title("Training Cost Over Time")
            plt.show()

        return self.params, J_storage


    def predict(self, df_features, target):
        """Model returns predictions of specified target using the given input features.

        Parameters
        ----------
        df_features : pandas.DataFrame
            Data containing input features to predict with, where each row is a set of input features (x).
            It is a DataFrame with shape (m,n), where m is the number of data points to be predicted
            and n is the number of feature variables.
        target : str
            Column name of target variable to be predicted.

        Returns
        -------
        y_hat : numpy.ndarray
            NumPy array that has a shape of (m,1) and contains the model's predictions for each
            of the input data points.
        """

        self._target_valid(target)
        norm_features,_,_ = self._normalize_z(df_features, self.means, self.stds) # normalize input features using training feature means and stds
        X = self._prepare_X(norm_features)
        y_hat = self._calc_linreg(X, self.params[target])
        return y_hat


    def validate(self, target, visualize=True):
        """Evaluate the model's performance using testing data and return performance metrics.

        Extended Summary
        ----------------
        Using the testing data, the model will be evaluated by comparing its predictions (of the
        specified target variable) with the true target values. This comparison is visualised with
        scatter plots of the target variable against each input feature (optional). The following metrics,
        which evaluate the multiple linear regression model's predictions, are computed and returned:
        - Mean Square Error
        - Mean Absolute Error
        - R2 coefficient of determination
        - Adjusted R2 coefficient of determination

        Parameters
        ----------
        target : str
            Column name of target variable to be predicted. The model will be evaluated based on its
            predictions for this target variable.
        visualize : bool, default True
            If True, comparison between predictions and true values is visualized.

        Returns
        -------
        mse, mae, r2, r2_adj : float
            The numerical values of the model's performance metrics mentioned above.

        Notes
        -----
        - The R2 coefficient is usually a value between 0 and 1, in which 0 indicates that the model explains
        none of the variability and 1 indicates that the model explains all the variability. When there is more
        than 1 feature however, it is possible for R2 coefficient to be negative if the fit is particularly bad.
        - For multiple linear regression, the R2 coefficient is not a good performance metric, as increasing the
        number of features always increases the R2 value. The adjusted R2 value (r2_adj) is a metric that accounts
        for this "inflation" of the R2 value when the number of feature variables increases, and is therefore
        more appropriate for evaluating model fit.
        """

        self._target_valid(target)
        df_features = self.test_features
        df_targets = self.test_targets

        y_hat = self.predict(df_features, target)
        y = self._prepare_y(df_targets.loc[:, [target]])
        if visualize == True:
            self._visualize(df_features, df_targets, target, y_hat)
        mse, mae, r2, r2_adj = self._calc_metrics(y_hat, y)
        print(f"Mean Square Error: {mse:.5f} | Mean Absolute Error: {mae:.5f} | R2 Coefficient of Determination: {r2:.5f} | Adjusted R2: {r2_adj:.5f}")
        return mse, mae, r2, r2_adj


    """Helper (private) functions"""

    def _visualize(self, df_features, df_targets, target, y_hat, size=5, n=3):
        """Visualize comparison between model prediction and true target values with scatter plots.

        Parameters
        ----------
        df_features, df_targets : pandas.DataFrame
            Dataframes containing the features and targets.
        target : str
            Column name of target variable being predicted.
        y_hat : numpy.ndarray
            NumPy array containing model's predictions.
        size: int, default 5
            Determines the size of the scatterplots.
        n : int, default 3
            Maximum number of scatterplots in a row.
        """

        # Prepare data for plots (everything should be 1-dimensional)
        num_features = len(self.features)
        y = df_targets[target]
        y_hat = y_hat.flatten()

        # Calculate the number of rows and columns for subplots (n plots per row)
        num_rows = int(np.ceil(num_features / n))
        num_cols = min(num_features, n)
        width = size * num_cols
        height = size * num_rows

        # Set up subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(width, height))
        if num_features > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Subplot style parameters
        y_params = {
            'color': 'blue',
            'marker': 'o',
            'label': "True values"
        }

        y_hat_params = {
            'color': 'orange',
            'marker': 'o',
            'label': "Predictions"
        }

        # Plot target against each feature
        for i in range(num_features):
            feature = self.features[i]
            x = df_features[feature]
            sns.scatterplot(x=x, y=y, ax=axes[i], **y_params)
            sns.scatterplot(x=x, y=y_hat, ax=axes[i], **y_hat_params)

            # Add labels and title
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target)
            axes[i].set_title("  ")
            # axes[i].set_title(f"Scatter plot of {target} against {feature}")
            axes[i].legend()

        # Hide remaining empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_axis_off()

        plt.tight_layout()
        plt.show()


    def _calc_metrics(self, y_hat, y):
        """Calculate 4 metrics for model evaluation.

        Parameters
        ----------
        y_hat, y : numpy.ndarray
            Arrays containing model's predictions (y_hat) and the true values (y)
            for m training examples. Both should be of shape (m,1).
        """

        r2 = self.R2(y_hat, y)
        r2_adj = self.adjusted_R2(r2, y.shape[0], len(self.features))
        mse = self.mean_squared_error(y_hat, y)
        mae = self.mean_abs_error(y_hat, y)
        return mse, mae, r2, r2_adj


    def _target_valid(self, target):
        """Checks whether given target name is valid (found in stored data)
        and whether the model has been trained to predict it yet."""

        if target not in self.targets:
            raise ValueError("Specified target not found in stored data.")
        elif target not in self.params.keys():
            raise ValueError("Model has not been trained to predict this target variable yet. Run the train() method.")


    @staticmethod
    def _calc_linreg(X, params):
        """Calculates linear regression equation.

        Parameters
        ----------
        X : numpy.ndarray
            Array of shape (m,n+1) where m is the number of data examples (rows)
            and n is the number of feature variables (excluding the intercept term).
        params : numpy.ndarray
            Array of shape (n+1,1) i.e. column vector where the ith value is the
            coefficient for the ith feature (the 0th coefficient is for intercept term)

        Returns
        -------
        np.ndarray
            Array of shape (m,1).
        """

        return np.matmul(X, params)


    @staticmethod
    def _split_data(df_features, df_targets, random_state, test_size):
        """Splits the given feature and target dataframes into a test
        and train set randomly.

        Parameters
        ----------
        df_features, df_targets : pandas.DataFrame
            DataFrames to be split into test and train sets.
        random_state : int
            Integer representing a seed for splitting of the data into training and test sets.
            Setting a seed to a specific value ensures reproducibility. If parameter is set to None
            instead of an integer, no seed will be specified and data splitting will be random.
        test_size : float
            Fraction (between 0 to 1) of the data to be allocated as the test set.

        Returns
        -------
        df_features_train, df_targets_train : pandas.DataFrame
            Features and targets for training set.
        df_features_test, df_targets_test : pandas.DataFrame
            Features and targets for test set.
        """

        # Set the random seed
        if random_state is not None:
          np.random.seed(random_state)

        indexes = df_features.index
        k = int(len(indexes) * test_size)

        # Randomly shuffle the indices to create a random test set
        test_indexes = np.random.choice(indexes,k,replace=False)
        train_indexes = list(set(indexes) - set(test_indexes))
        df_features_train = df_features.loc[train_indexes, :]
        df_features_test = df_features.loc[test_indexes, :]
        df_targets_train = df_targets.loc[train_indexes, :]
        df_targets_test = df_targets.loc[test_indexes, :]

        return df_features_train, df_features_test, df_targets_train, df_targets_test


    @staticmethod
    def _normalize_z(dfin, columns_means=None, columns_stds=None):
        """Normalize input dataframe using Z normalization

        Parameters
        ----------
        dfin : pandas.DataFrame
            Input data to be normalized.
        columns_means, columns_stds : float, optional
            Mean and standard deviation values used for normalization. If not
            specified (default None), the values will be computed using the input data.

        Returns
        -------
        dfout : pandas.DataFrame
            DataFrame containing normalized data.
        columns_means, columns_stds : float
            Will return the mean and standard deviation calculated using input DataFrame
            or the values passed in as arguments (depending on whether values for
            columns_means and columns_stds were passed in).
        """

        if columns_means is None:
            columns_means = dfin.mean(axis=0) # mean of all rows in each column
        if columns_stds is None:
            columns_stds = dfin.std(axis=0)

        dfout = (dfin - columns_means) / columns_stds # broadcasting
        return dfout, columns_means, columns_stds


    @staticmethod
    def _prepare_X(df_features):
        """Turns DataFrame of feature values into design matrix X.

        Extended Summary
        ----------------
        Takes in DataFrame with shape (m,n) where m is the number of data
        examples and n is the number of features. It will be converted into
        a 2-dimensional NumPy array and a column of 1s (x0) will be added for
        the intercept term.

        Parameters
        ----------
        df_features : pandas.DataFrame
            DataFrame of feature values with shape (m,n). (If a pandas.Series is passed
            in instead, it will be converted to pandas DataFrame automatically.)

        Returns
        -------
        X : numpy.ndarray
            NumPy array representing design matrix X with shape (m,n+1).
        """

        if type(df_features) == pd.Series:
            df_features = pd.DataFrame(df_features)

        rows, cols = df_features.shape
        if type(df_features) == pd.DataFrame:
            features = df_features.to_numpy() # Convert dataframe to np array
        else:
            features = df_features
        features = features.reshape(-1, cols) # ensure it is 2-dimensional array

        # Add new column of features x0 = 1 (intercept term):
        x0 = np.ones((rows,1))
        X = np.concatenate((x0, features), axis=1)
        return X # (m, n+1)


    @staticmethod
    def _prepare_y(df_target):
        """Turns DataFrame of target values into column vector y.

        Extended Summary
        ----------------
        Takes in DataFrame with shape (m,1) where m is the number of data
        examples. It will be converted into a 2-dimensional NumPy array with shape (m,1).

        Parameters
        ----------
        df_target : pandas.DataFrame
            DataFrame of feature values with shape (m,1). (If a pandas.Series is passed
            in instead, it will be converted to pandas DataFrame automatically.)

        Returns
        -------
        y : numpy.ndarray
            NumPy array representing column vector y with shape (m,1).
        """

        if type(df_target) == pd.Series:
            df_target = pd.DataFrame(df_target)

        cols = df_target.shape[1]
        if type(df_target) == pd.DataFrame:
            target = df_target.to_numpy()
        else:
            target = df_target

        target = target.reshape(-1, cols) # reinforce that targets should be 2-d np array
        return target


    @staticmethod
    def mean_squared_error(y_hat, y):
        """Compute Mean Squared Error (MSE)."""

        mse =  np.mean(np.square(y_hat - y))
        return mse

        '''# Alternatively, using matrices:
        m = y.shape[0] # number of data examples
        err = y_hat - y
        mse = np.matmul(err.T, err) / m # no need half factor
        mse = mse[0][0]
        # is matrix approach better? check which is computationally faster
        '''


    @staticmethod
    def R2(y_hat, y):
        """Compute R-squared value."""

        y_bar = np.mean(y)
        err = y_hat - y
        ss_residual = np.sum(np.square(y - y_hat))
        ss_total = np.sum(np.square(y - y_bar)) # y_bar is broadcasted
        r2 = 1 - (ss_residual / ss_total)
        # It is possible for r2 to be negative (ss_res > ss_total) if not model not well fitted.
        return r2

        '''# Alternatively, using matrices:
        # err = y_hat - y
        # ss_res = np.matmul(err.T, err)
        # ss_tot = np.matmul((y - y_bar).T, (y - y_bar))
        '''


    @staticmethod
    def adjusted_R2(r2, num_examples, num_features):
        """Compute Adjusted R-squared value."""

        return 1 - ( (1-r2)*(num_examples-1) ) / (num_examples-num_features-1)


    @staticmethod
    def mean_abs_error(y_hat, y):
        """Compute Mean Absolute Error (MAE)."""

        return np.mean(np.abs(y_hat - y))
    

class LocallyWeightedRegression(MultipleLinearRegression):
    """Class for a locally weighted regression model.

    Extended Summary
    ----------------
    The model is a subclass of MultipleLinearRegression and inherits most
    of its functionality. Unlike in multiple linear regression, however, this
    model does not save any parameter values and are instead computed everytime
    the model makes a prediction (as it needs to fit to the locally weighted cost).
    Hence there is no training step. After training data is stored, the model can
    make predictions immediately.
    General user flow:
    1) Store data (expects pandas DataFrames) using store_data().
    2) Validate the model's performance using validate().

    Attributes
    ----------
    features, targets : list of str
        A list of strings containing the column names of the features and targets respectively.
        Initialised when data is stored in the instance using store_data().
    train_features, test_features, train_targets, test_targets : pandas.DataFrames
        Pandas Dataframes containing the features/targets for training/testing. Initialised
        when data is stored in the instance using store_data().

    Methods
    -------
    Inherited:
        store_data(data, feature_names, target_names, random_state=None, test_size=0.5)
            Takes the given data and separates the features and targets. Then splits the data
            for training and testing before storing it in the model's attributes.
        add_transformed_feature(feature_name, new_feature_name, transform_func, replace=False)
            Transforms a specified feature and adds it to the stored data.

    Overriden:
        train()
            This method is disabled. Parameter values are obtained for each
            prediction in locally weighted regression.
        predict(dfin_features, target, tau=1)
            Given some input data of features, the model returns its predictions for the specified target
            variable using locally weighted regression.
        validate(target, visualize=True, tau=1)
            Evaluate the model's ability to predict the specified target variable (its performance) using the
            test data. Will show visualizations of the predicted values against the true values for the test data
            and return various performance metrics.

    Notes
    -----
    Locally weighted regression is a non-parametric learning algorithm. The bandwidth parameter (tau) is very
    important value that has a significant impact on the model's performance.
    """

    def __init__(self):
        """Constructs a new instance of LocallyWeightedRegression.

        Notes
        -----
        Unlike the MultipleLinearRegression class, there is no need to store
        parameters nor the mean and standard deviation of the training features.
        In locally weighted regression, the parameter values are determined for
        each prediction.
        """

        self.features, self.targets = None, None # list of names
        self.train_features, self.train_targets = None, None
        self.test_features, self.test_targets = None, None

    def predict(self, dfin_features, target, tau=1):
        """Predict for the specified target with the given input features using
        locally weighted regression.

        Parameters
        ----------
        dfin_features : pandas.DataFrame
            Data containing input features to predict with, where each row is a set of input features (x).
            It is a DataFrame with shape (m,n), where m is the number of data points to be predicted
            and n is the number of feature variables.
        target : str
            Column name of target variable to be predicted.
        tau : float, default 1
            Bandwidth parameter value of the gaussian kernel (local weight).

        Returns
        -------
        y_hat : numpy.ndarray
            NumPy array that has a shape of (m,1) and contains the model's predictions for each
            of the input data points.

        Notes
        -----
        In locally weighted regression, the parameters for each prediction changes depending on the
        input features. Datapoints closer to the input feature values (point of prediction) are given
        more weightage than points further away. The bandwidth parameter (tau) determines the spread of
        the gaussian kernel (which determines the weightages of points). A smaller bandwidth parameter
        means a more concentrated gaussian kernel, making it more sensitive to neighbouring points as opposed
        to more distant ones.
        When computing the parameters using the matrix equation, this function finds the Moore-Penrose inverse
        rather than the regular inverse. This is because the square matrix in the equation (X_train.T @ W @ X_train)
        may be singular and therefore non-invertible. In this case, the Moore-Penrose inverse is sufficient for
        the purpose of optimising the locally-weighted cost function.
        """

        # Check that there is data stored for training and validation:
        if any(data is None for data in (self.train_features, self.train_targets, self.test_features, self.test_targets)):
            raise ValueError("Data missing. Run the store_data() method first.")
        # Check that valid target name is specified:
        elif target not in self.targets:
            raise ValueError("Specified target not found in stored data.")

        # Prepare training features and target:
        train_features_norm, means, stds = self._normalize_z(self.train_features)
        X_train = self._prepare_X(train_features_norm)
        y = self._prepare_y(self.train_targets.loc[:, [target]])

        # Normalize and prepare input features
        test_features_norm, _, _ = self._normalize_z(dfin_features, means, stds)
        X_in = self._prepare_X(test_features_norm)

        num_inputs = X_in.shape[0]
        y_hat = np.zeros(num_inputs) # for storing predictions

        # Make prediction for each input data (row in X_in):
        for i in range(num_inputs):
            # x is specific point at which you want to make prediction
            x = X_in[i,:] # (n,) 1D array for broadcasting purposes

            dist = np.sum((X_train - x)**2, axis=1)
            w = np.exp(-dist / (2 * tau**2)) # (m,)
            W = np.diag(w) # diagonal matrix (m,m)
            params = np.linalg.pinv(X_train.T @ W @ X_train) @ (X_train.T @ W @ y) # Moore-Penrose inverse
            pred = x.reshape(1,-1) @ params
            y_hat[i] = pred

        y_hat = y_hat.reshape(-1,1)
        return y_hat # (m,1)

    def validate(self, target, visualize=True, tau=1):
        """Evaluate the model's performance using testing data and return performance metrics.

        Extended Summary
        ----------------
        Using the testing data, the model will be evaluated by comparing its predictions (of the
        specified target variable) with the true target values. This comparison is visualised with
        scatter plots of the target variable against each input feature. The following metrics, which
        evaluate the multiple linear regression model's predictions, are computed and returned:
        - Mean Square Error
        - Mean Absolute Error
        - R2 coefficient of determination
        - Adjusted R2 coefficient of determination

        Parameters
        ----------
        target : str
            Column name of target variable to be predicted. The model will be evaluated based on its
            predictions for this target variable.
        visualize : float, default True
            If True, comparison between predictions and true values is visualized.
        tau : float, default 1
            Bandwidth parameter value of the gaussian kernel (local weight).

        Returns
        -------
        mse, mae, r2, r2_adj : float
            The numerical values of the model's performance metrics mentioned above.

        Notes
        -----
        - The R2 coefficient is usually a value between 0 and 1, in which 0 indicates that the model explains
        none of the variability and 1 indicates that the model explains all the variability. When there is more
        than 1 feature however, it is possible for R2 coefficient to be negative if the fit is particularly bad.
        - For multiple linear regression, the R2 coefficient is not a good performance metric, as increasing the
        number of features always increases the R2 value. The adjusted R2 value (r2_adj) is a metric that accounts
        for this "inflation" of the R2 value when the number of feature variables increases, and is therefore
        more appropriate for evaluating model fit.
        """

        self._target_valid(target)
        df_features = self.test_features
        df_targets = self.test_targets

        y_hat = self.predict(df_features, target, tau=tau)
        if visualize == True:
            self._visualize(df_features, df_targets, target, y_hat)
        y = self._prepare_y(df_targets.loc[:, [target]])
        mse, mae, r2, r2_adj = self._calc_metrics(y_hat, y)
        print(f"Mean Square Error: {mse:.5f} | Mean Absolute Error: {mae:.5f} | R2 Coefficient of Determination: {r2:.5f} | Adjusted R2: {r2_adj:.5f}")
        return mse, mae, r2, r2_adj


    def _target_valid(self, target):
        """Checks whether given target name is valid (found in stored data)"""

        if target not in self.targets:
            raise ValueError("Specified target not found in stored data.")


    @staticmethod
    def train():
        raise Exception("This method is non-functional for Locally Weighted Regression. Parameter values are computed during each prediction instead.")

if __name__ == '__main__':
    # Demo on how to use:
    df = pd.read_csv("sample_data/housing_processed.csv")
    model = MultipleLinearRegression()
    model.store_data(df, ["RM","DIS","INDUS"], ["MEDV"], random_state=100, test_size=0.5)
    model.add_transformed_feature("RM", "RM Square", np.square, replace=False)
    model.train("MEDV")
    model.validate("MEDV")