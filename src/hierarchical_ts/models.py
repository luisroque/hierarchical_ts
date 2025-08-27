from statsforecast.models import AutoETS
from mlforecast import MLForecast
import xgboost as xgb
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import MAE


def get_auto_ets_model(season_length, model="ZZA"):
    """
    Returns an AutoETS model.

    Args:
        season_length (int): The length of the season.
        model (str): The ETS model specification.

    Returns:
        AutoETS: An AutoETS model.
    """
    return AutoETS(season_length=season_length, model=model)


def get_xgboost_model(lags, date_features):
    """
    Returns an XGBoost model.

    Args:
        lags (list): A list of lags to use as features.
        date_features (list): A list of date features to use.

    Returns:
        MLForecast: An MLForecast model with an XGBoost regressor.
    """
    return MLForecast(
        models=[xgb.XGBRegressor(n_estimators=10)],
        freq="MS",
        lags=lags,
        date_features=date_features,
    )


def get_nbeats_model(horizon, input_size):
    """
    Returns an NBEATS model.

    Args:
        horizon (int): The forecast horizon.
        input_size (int): The input size for the model.

    Returns:
        NBEATS: An NBEATS model.
    """
    return NBEATS(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        scaler_type="standard",
        max_steps=50,
    )
