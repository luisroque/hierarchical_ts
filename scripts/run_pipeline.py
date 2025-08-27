import logging
import numpy as np
import pandas as pd
from hierarchical_ts.data import load_tourism_data, split_data
from hierarchical_ts.models import (
    get_auto_ets_model,
    get_xgboost_model,
    get_nbeats_model,
)
from hierarchical_ts.train import train_statsforecast_model, train_ml_model
from hierarchical_ts.predict import get_reconcilers, reconcile_forecasts
from hierarchical_ts.evaluate import evaluate_forecasts
from utilsforecast.losses import rmse, scaled_crps

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # Load and process data
    logging.info("Loading and processing data...")
    Y_df, S_df, tags = load_tourism_data()

    # Split data
    logging.info("Splitting data...")
    horizon = 12
    Y_train_df, Y_test_df = split_data(Y_df, horizon)
    Y_train_df = Y_train_df.copy()  # Address fragmentation warning

    # Train models
    # # ETS
    # logging.info("Training ETS model...")
    # ets_model = get_auto_ets_model(season_length=12, model="ZZZ")
    # Y_hat_ets, Y_fitted_ets = train_statsforecast_model(
    #     ets_model, Y_train_df, "MS", horizon
    # )

    # XGBoost
    logging.info("Training XGBoost model...")
    level = np.arange(0, 100, 2)
    xgb_model = get_xgboost_model(lags=[1, 2, 12, 24], date_features=["month"])
    Y_hat_xgb, Y_fitted_xgb = train_ml_model(xgb_model, Y_train_df, horizon, level)

    # NBEATS
    logging.info("Training NBEATS model...")
    nbeats_model = get_nbeats_model(horizon=horizon, input_size=2 * horizon)
    Y_hat_nbeats, Y_fitted_nbeats = train_ml_model(
        nbeats_model, Y_train_df, horizon, level
    )

    # Reconcile forecasts
    logging.info("Reconciling forecasts...")
    reconcilers = get_reconcilers()
    # Y_rec_ets = reconcile_forecasts(reconcilers, Y_hat_ets, Y_fitted_ets, S_df, tags)
    Y_rec_xgb = reconcile_forecasts(
        reconcilers, Y_hat_xgb, Y_fitted_xgb, S_df, tags, level
    )
    Y_rec_nbeats = reconcile_forecasts(
        reconcilers, Y_hat_nbeats, Y_fitted_nbeats, S_df, tags, level
    )

    # Combine reconciled forecasts
    logging.info("Combining reconciled forecasts...")
    Y_rec_df = Y_rec_xgb.merge(Y_rec_nbeats, on=["unique_id", "ds"])

    # Evaluate forecasts
    logging.info("Evaluating forecasts...")
    # ets_models = [col for col in Y_rec_df.columns if "AutoETS" in col]
    xgb_models = [col for col in Y_rec_df.columns if "XGBRegressor" in col]
    nbeats_models = [col for col in Y_rec_df.columns if "NBEATS" in col]

    # evaluation = evaluate_forecasts(Y_rec_df, Y_test_df, tags, [rmse], ets_models)
    evaluation_xgb = evaluate_forecasts(
        Y_rec_df, Y_test_df, tags, [scaled_crps], xgb_models, level
    )
    evaluation_nbeats = evaluate_forecasts(
        Y_rec_df, Y_test_df, tags, [scaled_crps], nbeats_models, level
    )

    # print("ETS Evaluation:")
    # print(evaluation)
    logging.info("XGBoost Evaluation:\n%s", evaluation_xgb)
    logging.info("NBEATS Evaluation:\n%s", evaluation_nbeats)

    # Save results
    logging.info("Saving results...")
    # evaluation.to_csv("assets/results/evaluation_ets.csv", index=False)
    evaluation_xgb.to_csv("assets/results/evaluation_xgb.csv", index=False)
    evaluation_nbeats.to_csv("assets/results/evaluation_nbeats.csv", index=False)
    logging.info("Pipeline finished. Results saved to assets/results/")


if __name__ == "__main__":
    main()
