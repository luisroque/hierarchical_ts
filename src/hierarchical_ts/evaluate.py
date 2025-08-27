from hierarchicalforecast.evaluation import evaluate
from utilsforecast.losses import rmse, scaled_crps


def evaluate_forecasts(Y_rec_df, Y_test_df, tags, metrics, models, level=None):
    """
    Evaluates the forecasts.

    Args:
        Y_rec_df (pd.DataFrame): The reconciled forecasts.
        Y_test_df (pd.DataFrame): The test data.
        tags (dict): A dictionary of tags for the hierarchy levels.
        metrics (list): A list of evaluation metrics.
        models (list): A list of model names to evaluate.
        level (list, optional): A list of prediction levels. Defaults to None.

    Returns:
        pd.DataFrame: The evaluation results.
    """
    df = Y_rec_df.merge(Y_test_df, on=["unique_id", "ds"])
    evaluation = evaluate(df=df, tags=tags, metrics=metrics, models=models, level=level)
    return evaluation
