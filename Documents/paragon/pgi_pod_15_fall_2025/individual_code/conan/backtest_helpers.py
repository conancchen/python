import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from scipy import stats
import statsmodels.api as sm

def separate():
    print("="*50)

def set_date_index(df: pd.DataFrame, date_column_name: str, date_format_str: str) -> pd.DataFrame:
    df = df.rename(columns={date_column_name: "date"})
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index, format = date_format_str)
    return df

def compute_betas(portfolio_excess_returns, risk_factor_returns):
    '''
    args:
        portfolio_excess_returns: df of excess returns of portfolios
        risk_factor_returns: df of risk factor returns (used as X in regression)
    does:
        computes the beta, which conceptually is the covariance between the portfolio and risk_factor 
        over variance of the risk_factor, via time series regression. Note that if we use multiple risk
        factors, each will control for the others.
    returns:
        the betas for each of the portfolios 
    '''
    
    # We want the betas of each portfolio to go in here
    betas = []
    resids = []
    alphas = []
    const_t_stats = []
    
    # If we are only regressing on one factor, make sure it is structured as a df
    if isinstance(risk_factor_returns, pd.Series):
        factor_name = risk_factor_returns.name or "factor"
        risk_factor_returns = risk_factor_returns.to_frame(name=factor_name)

    X_clean = risk_factor_returns.dropna()
    
    # Time Series to estimate betas
    for portfolio in portfolio_excess_returns.columns:
        # Returns of the particular portfolio (rp stands for return portfolio)
        rp_excess_clean = portfolio_excess_returns[portfolio].dropna()
        
        common_index = X_clean.index.intersection(rp_excess_clean.index)
        
        # Align the risk free, factor returns, and portfolio returns
        aligned_excess_rp = rp_excess_clean.loc[common_index]
        aligned_X = X_clean.loc[common_index, :]
        
        # perform the regression
        X_format = np.array(aligned_X).reshape(-1, aligned_X.shape[1])
        X_format = sm.add_constant(X_format)  # add intercept
        model = sm.OLS(aligned_excess_rp, X_format).fit()

        # Include the alpha, beta, residuals, and t-stats on const for the security
        alphas.append(model.params['const'])
        betas.append(model.params.drop('const'))
        resids.append(model.resid)
        const_t_stats.append(model.tvalues['const'])
    
    # Formatting beta outputs
    betas = pd.concat(betas, axis=1).T
    betas.index = portfolio_excess_returns.columns
    betas.columns = [f"Beta on {col}" for col in risk_factor_returns.columns]
    
    # Formatting residuals outputs
    resids = pd.concat(resids, axis=1)
    resids.columns = portfolio_excess_returns.columns

    # Formatting alpha outputs
    alphas = pd.Series(alphas, index=portfolio_excess_returns.columns)
    alphas = pd.DataFrame(alphas, columns=['alpha'])
    alphas.index = portfolio_excess_returns.columns

    # Formatting t-stats outputs
    const_t_stats = pd.Series(const_t_stats, index=portfolio_excess_returns.columns)
    const_t_stats = pd.DataFrame(const_t_stats, columns=['t_stat_alpha'])
    const_t_stats.index = portfolio_excess_returns.columns
    
    # Return the betas, time series residuals, alphas, and t-stats of constant
    return betas, resids, alphas, const_t_stats

def cross_section_regression(all_returns, betas, char_dict=None, risk_free=None):
    '''
    args:
        all_returns: dataframe of excess returns (T x N)
        betas: dataframe of betas (N x F)
        char_dict: dict of (T x N)
        risk_free: risk free series to compute t-stat for gamma_0 (for self-financing, should be 0)
    does:
        performs a Fama-MacBeth cross sectional regression for each of the 
        days in the excess returns matrix. The design matrix is formed by
        taking the betas for the valid securities as well as the characteristics
        for the valid secuirites on that time period.
    returns:
        gammas: dataframe of the coefficients for each time period (T x (F + C))
        fm_mean: series of the mean of each coefficient across time ((F + C) x 1)
        fm_se: series of the standard error of each coefficient across time ((F + C) x 1)
        fm_t: series of the t-statistic of each coefficient across time ((F + C) x 1)
    '''

    per_date_coefs = []
    reg = LR()
    
    # Get the characteristic names
    if char_dict is not None:
        char_names = list(char_dict.keys())
    else:
        char_names = []
    
    # Make the labels for the coefficients in front of beta
    beta_cols = [f"gamma_{c}" for c in betas.columns]
    char_cols = [f"gamma_{name}" for name in char_names]
    r2_list = []
    
    # Iterate through the dates, each of which is a target
    for t, y_row in all_returns.iterrows():
        # Get assets that have a return
        mask = y_row.notna().copy()

        # A characteristic could be missing (but betas cannot)
        if char_dict is not None:
            for name, df in char_dict.items():
                mask &= df.loc[t].notna()

        # Subset to valid assets for this date
        assets_t = all_returns.columns[mask]

        # get the return information on day t for valid assets on day t
        y_t = y_row.loc[assets_t].to_numpy(dtype=float)
        
        # Build design matrix X
        X_parts = []
        
        # Only add betas if there are any
        if betas.shape[1] > 0:
            X_parts.append(betas.loc[assets_t, :].to_numpy(dtype=float))
        if char_dict is not None:
            for name in char_names:
                X_parts.append(char_dict[name].loc[t, assets_t].to_numpy(dtype=float).reshape(-1, 1))
        # Error check:
        if not X_parts:
            raise ValueError("No covariates were passed in on date " + 
                             f"{t} (no betas or characteristics). Preprocess data.")

        X_t = np.hstack(X_parts)

        # X_df = pd.DataFrame(X_t, index=assets_t)
        # y_df = pd.Series(y_t, index=assets_t)
        # display(pd.concat([y_df, X_df], axis=1))

        reg.fit(X_t, y_t)
        r2 = reg.score(X_t, y_t)
        r2_list.append(r2)
        
        coefs = [reg.intercept_] + list(reg.coef_)
        coef_index = ["gamma_0"] + beta_cols + char_cols
        per_date_coefs.append(pd.Series(coefs, index=coef_index, name=t))
    
    gammas = pd.DataFrame(per_date_coefs)
    
    # Summary stats
    fm_mean = gammas.mean(skipna=True)
    counts = gammas.count()
    fm_se = gammas.std(skipna=True, ddof=1) / np.sqrt(counts)

    # Subtract the null hypothesis mean (first element should be minus risk free mean
    # Null is 0 for factors)
    risk_free_aligned = risk_free.reindex(all_returns.index)
    risk_free_mean = risk_free_aligned.mean() if risk_free is not None else 0
    fm_t = (fm_mean - np.array([risk_free_mean] + [0] * (fm_mean.shape[0] - 1))) / fm_se
    
    return gammas, fm_mean, fm_se, fm_t, r2_list


def compute_mdd(return_series):
    '''
    args:
        returns_series: time series of returns for a strategy or asset
    does:
        computes the maximum drawdown (mdd) and start/end dates. mdd is defined
        as the maximum peak-to-trough decline in a price series. Trough is the lowest
        point before a new high-level is achieved
    returns:
        max drawdown, start date, and end date of mdd
    '''
    price_series = (1 + return_series).cumprod()
    price_series = price_series.dropna()

    # Append the value 1 to the start of the series
    prev_index = price_series.index[0] - pd.DateOffset(months=1)
    price_series = pd.concat([pd.Series([1], index=[prev_index]), price_series], axis=0)

    # Find peaks
    running_max = price_series.cummax()

    drawdown = (price_series - running_max) / running_max
    mdd = drawdown.min()
    mdd_end = drawdown.idxmin()
    mdd_start = price_series.loc[:mdd_end].idxmax()

    return mdd, mdd_start, mdd_end


def compute_sharpe_stderr(return_df, risk_free=None, 
                          ergodic=False, annualized=True):
    '''
    args:
        return_df: time series of returns for at least one strategy or asset (T x N)
        risk_free: time series of risk free returns (0 for self-financing)
        ergodic: dummy for whether to use ergodic or naive stderr formula
    does:
        computes the standard error of the Sharpe ratio, using one of two normal formulas
    returns:
        standard error of the Sharpe ratio
    '''
    scale = np.sqrt(12) if annualized else 1
    
    # T x N
    excess_return = return_df.sub(risk_free, axis=1) if risk_free is not None else return_df
    mean_return = excess_return.mean()
    std_return = excess_return.std()

    # Compute the number of dates that were used to compute sample stats
    T = excess_return.count(axis=0)

    # Compute Sharpe Ratio (unscaled)
    SR = mean_return / std_return
    
    if ergodic:
        # Compute higher moments for ergodic assumption (skewness and raw kurtosis)
        skewness = excess_return.apply(lambda x: stats.skew(x.dropna(), bias=False))
        kurtosis = excess_return.apply(lambda x: stats.kurtosis(x.dropna(), fisher=False, bias=False))

        SE = (1 / (T - 1) * (1 + 0.25 * (SR ** 2) * (kurtosis - 1) - SR * skewness)) ** 0.5
    else:
        SE = (1 / (T - 1) * (1 + 0.5 * SR ** 2)) ** 0.5
    
    return SE * scale
    