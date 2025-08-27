from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace, ERM


def get_reconcilers():
    """
    Returns a list of reconciliation methods.

    Returns:
        list: A list of reconciliation methods.
    """
    return [
        BottomUp(),
        MinTrace(method="mint_shrink"),
        MinTrace(method="ols"),
        ERM(method="closed"),
    ]


def reconcile_forecasts(reconcilers, Y_hat_df, Y_df, S, tags, level=None):
    """
    Reconciles the forecasts.

    Args:
        reconcilers (list): A list of reconciliation methods.
        Y_hat_df (pd.DataFrame): The dataframe with the base forecasts.
        Y_df (pd.DataFrame): The dataframe with the fitted values.
        S (pd.DataFrame): The summing matrix.
        tags (dict): A dictionary of tags for the hierarchy levels.
        level (list, optional): A list of prediction levels. Defaults to None.

    Returns:
        pd.DataFrame: The reconciled forecasts.
    """
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_df, S=S, tags=tags, level=level)
    return Y_rec_df
