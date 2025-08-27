from statsforecast.core import StatsForecast
from mlforecast.utils import PredictionIntervals
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals as NeuralPredictionIntervals


def train_statsforecast_model(model, df, freq, h, fitted=True):
    """
    Trains a StatsForecast model.

    Args:
        model (object): The StatsForecast model to train.
        df (pd.DataFrame): The training data.
        freq (str): The frequency of the time series.
        h (int): The forecast horizon.
        fitted (bool): Whether to return fitted values.

    Returns:
        tuple: A tuple containing the forecast and the fitted values.
    """
    fcst = StatsForecast(models=[model], freq=freq, n_jobs=-1)
    forecast = fcst.forecast(df=df, h=h, fitted=fitted)
    fitted_values = fcst.forecast_fitted_values()
    return forecast, fitted_values


def train_ml_model(model, df, h, level):
    """
    Trains an MLForecast or NeuralForecast model.

    Args:
        model (object): The model to train.
        df (pd.DataFrame): The training data.
        h (int): The forecast horizon.
        level (list): A list of prediction levels.

    Returns:
        tuple: A tuple containing the forecast and the in-sample predictions.
    """
    if "neuralforecast" in str(model.__class__).lower():
        nf = NeuralForecast(models=[model], freq="MS")
        prediction_intervals = NeuralPredictionIntervals()
        nf.fit(df=df, prediction_intervals=prediction_intervals)
        forecast = nf.predict(level=level)
        insample = nf.predict_insample(step_size=h)
    else:  # MLForecast
        model.fit(
            df, fitted=True, prediction_intervals=PredictionIntervals(n_windows=2, h=h)
        )
        forecast = model.predict(h, level=level)
        insample = model.forecast_fitted_values()
    return forecast, insample
