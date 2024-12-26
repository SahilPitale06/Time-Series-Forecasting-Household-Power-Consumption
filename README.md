# Time Series Forecasting: Household Power Consumption

This project performs time series forecasting on household power consumption data. It utilizes different methods such as ARIMA and LSTM (Long Short-Term Memory) neural networks to predict future energy consumption.

## Project Structure

- `complete_dataset.csv`: The dataset containing household power consumption data.
- `Time_Series_Forecasting.ipynb`: The main notebook for performing the analysis and forecasting.
- `README.md`: This file.

## Dependencies

This project requires the following Python libraries:

- `ucimlrepo`: To fetch the dataset.
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `seaborn`: For additional visualizations.
- `statsmodels`: For time series decomposition and ARIMA model.
- `scikit-learn`: For machine learning tools.
- `tensorflow`: For building and training the LSTM model.

## Dataset

The dataset used in this project is the "Individual Household Electric Power Consumption" dataset, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption). The dataset contains power consumption data collected from an electric meter in a single household over a period of several years. The data is sampled at a frequency of one minute.

## Steps

### Step 1: Load Dataset
The dataset is loaded from the UCI repository, where the features and target values are separated. The features are power consumption data, while the target is the energy usage.

### Step 2: Data Preprocessing
- The `Date` and `Time` columns are combined into a single `datetime` column.
- The `datetime` column is set as the index.
- The `Global_active_power` column is converted to numeric, and any missing values are handled.

### Step 3: Exploratory Data Analysis
- The data is resampled to a daily frequency, summing the power consumption for each day.
- Missing values are checked and handled by dropping any rows with NaN values.
- A time series plot is created to visualize the daily global active power consumption over time.
- The time series is decomposed into trend, seasonal, and residual components.

### Step 4: ARIMA Model
- The data is split into training and testing sets.
- An ARIMA (AutoRegressive Integrated Moving Average) model is built and trained on the training data.
- Forecasts are made for the test set, and the performance is evaluated using the RMSE (Root Mean Squared Error) metric.

### Step 5: LSTM Model
- The data is scaled using MinMaxScaler to prepare it for training with the LSTM model.
- Sequences of 30 days are created as input-output pairs for training.
- The LSTM model is built with two LSTM layers and dropout layers to prevent overfitting.
- The model is trained, and predictions are made for the test set.
- The predicted values are plotted and compared to the true values, and the model performance is evaluated using RMSE.

## How to Run
1. To install the necessary libraries, run:

```bash
pip install ucimlrepo pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow
```
2. Download the dataset:
The dataset is automatically fetched from the UCI repository in the notebook.

3. Run the Jupyter notebook ```bash (Time_Series_Forecasting.ipynb): ```
Open the notebook and run the cells one by one to execute the steps outlined above.

## Results
The results will include:

- A time series plot of global active power consumption.
- A decomposition of the time series into trend, seasonal, and residual components.
- Forecasted power consumption values using the ARIMA model.
- Forecasted power consumption values using the LSTM model.
- RMSE values for both ARIMA and LSTM models to evaluate their performance.

## Evaluation
The Root Mean Squared Error (RMSE) is used to evaluate the accuracy of the ARIMA and LSTM models. A lower RMSE value indicates better forecasting performance.

## Author
- **Name:** Sahil Pitale
- **Contact:** sp9328123456@gmail.com | [LinkedIn Profile](https://www.linkedin.com/in/sahil-pitale-56a5681bb/)
