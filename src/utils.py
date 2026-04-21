import pandas as pd
import numpy as np

def simulate_future(df, features, scenario, sim_length):
    import numpy as np
    import pandas as pd

    SCENARIOS = {
        "best case": {"temp increase": 1.5, "precipitation change": 0.01},
        "middle of the road": {"temp increase": 4, "precipitation change": 0.1},
        "business as usual": {"temp increase": 8, "precipitation change": 0.22},
    }

    scenario = SCENARIOS[scenario]

    # ---------------------------------------
    # Baseline
    # ---------------------------------------
    baseline = df.groupby(['state', 'month']).agg(
        temp_anom_mean=('temp_anomaly', 'mean'),
        temp_anom_std=('temp_anomaly', 'std'),
        temp_baseline=('baseline_temp', 'max'),
        precip_anom_mean=('precip_prop_anomaly', 'mean'),
        precip_anom_std=('precip_prop_anomaly', 'std'),
        precip_baseline=('baseline_precip', 'max')
    ).reset_index()

    dfs = [baseline.assign(future_year_offset=i) for i in range(1, sim_length + 1)]
    future_weather = pd.concat(dfs, ignore_index=True)

    growth = future_weather['future_year_offset'] / future_weather['future_year_offset'].max()

    future_weather = future_weather.sort_values(
        ['state', 'future_year_offset', 'month']
    ).reset_index(drop=True)


    # ---------------------------------------
    # GENERATE CORRELATED WEATHER SHOCKS
    # ---------------------------------------

    n = len(future_weather)

    # probability of shock 
    # shock_prob = 0.04 # constant
    # shock_prob = 0.03 + 0.07 * growth # linear increase
    shock_prob = 0.03 + 0.20 * (growth ** 2) # nonlinear increase (acceleration)
    shock_flag = np.random.binomial(1, shock_prob, n)

    # shock type: 0 = heat/drought, 1 = wet storm
    shock_type = np.random.binomial(1, 0.5, n)

    # initialise
    temp_shock = np.zeros(n)
    precip_shock = np.zeros(n)

    # HEAT / DROUGHT shocks
    heat_idx = (shock_flag == 1) & (shock_type == 0)
    temp_shock[heat_idx] = np.random.normal(3, 1.2, heat_idx.sum())     # strong heat
    precip_shock[heat_idx] = np.random.normal(-0.4, 0.15, heat_idx.sum()) # dry

    # WET / STORM shocks
    wet_idx = (shock_flag == 1) & (shock_type == 1)
    temp_shock[wet_idx] = np.random.normal(0.5, 0.5, wet_idx.sum())       # mild temp
    precip_shock[wet_idx] = np.random.normal(0.8, 0.3, wet_idx.sum())     # very wet

    # ---------------------------------------
    # SHOCK PERSISTENCE
    # ---------------------------------------

    for state in future_weather['state'].unique():
        idx = np.where(future_weather['state'] == state)[0]

        for j in range(1, len(idx)):
            i = idx[j]
            prev_i = idx[j-1]

            if shock_flag[prev_i] == 1 and np.random.rand() < 0.5:
                shock_flag[i] = 1
                temp_shock[i] += temp_shock[prev_i] * 0.6
                precip_shock[i] += precip_shock[prev_i] * 0.6

    # ---------------------------------------
    # TEMPERATURE
    # ---------------------------------------
    yearly_temp_increase = scenario['temp increase'] / 76

    base_temp = (
        future_weather['temp_baseline']
        + yearly_temp_increase * future_weather['future_year_offset']
        + np.random.normal(0, 0.2 + 0.02 * future_weather['future_year_offset'], n)
        + temp_shock   #  add shock
    )

    future_weather['temp_anomaly'] = base_temp - future_weather['temp_baseline']

    temp_std_safe = future_weather['temp_anom_std'].clip(lower=0.1)

    future_weather['temp_anom_z'] = (
        (future_weather['temp_anomaly'] - future_weather['temp_anom_mean'])
        / temp_std_safe
    ).clip(-5, 10)

    # ---------------------------------------
    # PRECIPITATION
    # ---------------------------------------
    season = np.cos(2 * np.pi * (future_weather['month'] - 1) / 12)

    mean_drift = 0.03 * future_weather['future_year_offset']
    seasonal_strength = scenario['precipitation change'] * growth

    seasonal_component = seasonal_strength * season

    noise_scale = 0.12 + 0.02 * future_weather['future_year_offset']

    base_precip = (
        future_weather['precip_baseline']
        * (1 + mean_drift + seasonal_component + precip_shock)  # add shock
        * np.exp(np.random.normal(0, noise_scale, n))
    ).clip(lower=0)

    future_weather['precip_anomaly'] = (
        (base_precip - future_weather['precip_baseline'])
        / future_weather['precip_baseline']
    )

    precip_std_safe = future_weather['precip_anom_std'].clip(lower=0.1)

    future_weather['precip_anom_z'] = (
        (future_weather['precip_anomaly'] - future_weather['precip_anom_mean'])
        / precip_std_safe
    ).clip(-5, 10)

    # ---------------------------------------
    # PERSISTENCE
    # ---------------------------------------
    for col in ['temp_anom_z', 'precip_anom_z']:
        future_weather[col] = (
            0.4 * future_weather.groupby('state')[col].shift(1).fillna(0)
            + 0.6 * future_weather[col]
        )

    # ---------------------------------------
    # DROUGHT INDEX
    # ---------------------------------------
    future_weather['drought_index'] = (
        future_weather['temp_anom_z']
        - 0.7 * future_weather['precip_anom_z']
        + 0.5 * temp_shock
        - 0.5 * precip_shock
    )

    # ---------------------------------------
    # HOT DAYS
    # ---------------------------------------
    baseline_hot_days = (
        df.groupby(['state', 'month'])['n_hot_days']
        .mean()
        .reset_index()
        .rename(columns={'n_hot_days': 'baseline_hot_days'})
    )

    future_weather = future_weather.merge(baseline_hot_days, on=['state', 'month'], how='left')

    season_weight = np.clip(
        np.cos(2 * np.pi * (future_weather['month'] - 7) / 12),
        0,
        None
    ) ** 3

    hot_trend = 5 * future_weather['temp_anomaly']  + 2 * temp_shock

    future_weather['n_hot_days'] = (
        future_weather['baseline_hot_days']
        + hot_trend
    ) * (0.3 + 0.7 * season_weight * growth)

    future_weather['n_hot_days'] += np.random.normal(
        0,
        0.5 + 0.05 * future_weather['future_year_offset'],  # ↑ more variability
        size=n
    )

    future_weather['n_hot_days'] = future_weather['n_hot_days'].clip(0, 31).round()

    # ---------------------------------------
    # HEAVY RAIN DAYS
    # ---------------------------------------
    baseline_rain_days = df.loc[df['year'] == 2024, ['state', 'month', 'heavy_rain_days']]
    baseline_rain_days = baseline_rain_days.rename(columns={'heavy_rain_days': 'baseline_rain_days'})

    future_weather = future_weather.merge(baseline_rain_days, on=['state', 'month'], how='left')

    rain_season = (0.5 + 0.5 * np.cos(2 * np.pi * (future_weather['month'] - 1) / 12)) ** 2

    rain_signal = (
    12
    * (future_weather['precip_anomaly'] + precip_shock)  # link to shocks
    * rain_season
    * growth
)

    noise = np.random.normal(
        0,
        (0.2 + 0.03 * future_weather['future_year_offset']) * (0.5 + rain_season),
        size=n
    )

    future_weather['heavy_rain_days'] = (
        future_weather['baseline_rain_days']
        + rain_signal
        + 0.08 * future_weather['future_year_offset']
        + noise
    ).clip(0, 31).round()

    # ---------------------------------------
    # TIME FEATURES
    # ---------------------------------------
    latest_weather = df.loc[(df['year'] == 2024) & (df['month'] > 9)]

    future_weather['year'] = 2024 + future_weather['future_year_offset']
    future_weather['year_offset'] = 20 + future_weather['future_year_offset']

    future_weather['month_sin'] = np.sin(2 * np.pi * future_weather['month'] / 12)
    future_weather['month_cos'] = np.cos(2 * np.pi * future_weather['month'] / 12)

    recent = df[df['year'] >= df['year'].max() - 3]

    future_weather['log_n_obs'] = (
        recent.groupby(['state', 'month'])['log_n_obs']
        .max()
        .reindex(future_weather.set_index(['state', 'month']).index)
        .values
    )

    future_weather = pd.concat([latest_weather, future_weather])
    future_weather = future_weather.sort_values(['state', 'year', 'month'])

    # ---------------------------------------
    # LAGS + ROLLING
    # ---------------------------------------
    for col in [
        'temp_anom_z',
        'precip_anom_z',
        'n_hot_days',
        'drought_index',
        'heavy_rain_days'
    ]:
        future_weather[f'{col}_lag1'] = future_weather.groupby('state')[col].shift(1)

        future_weather[f'{col}_roll3'] = future_weather.groupby('state')[col].transform(
            lambda x: x.shift(1).rolling(3).mean()
        )

    # ---------------------------------------
    # FINAL
    # ---------------------------------------


    future_weather['temp_anom_z_sq'] = future_weather['temp_anom_z'] ** 2
    future_weather['drought_index_sq'] = future_weather['drought_index'] ** 2

    future_weather['int1'] = future_weather['temp_anom_z'] * future_weather['precip_anom_z']
    future_weather['int2'] = future_weather['temp_anom_z'] * future_weather['n_hot_days']
    future_weather['int3'] = future_weather['drought_index'] * future_weather['n_hot_days']


    features_extended = features.copy()
    features_extended.extend(['year', 'month'])

    future_weather = future_weather.loc[future_weather['year_offset'] > 20]
    future_weather = future_weather[features_extended]

    return future_weather


def forecast_biodiversity(df, model, features, freeze_year=True):

    X = df[features].copy()

    if freeze_year and 'year_offset' in X.columns:
        X['year_offset'] = 20

    df['pred_biodiv_z'] = model.predict(X)

    return df



def simulate_with_uncertainty_regression(
    df, state, scenario, features, model, sim_length=50, n_runs=50
):
    all_runs = []

    for i in range(n_runs):
        sim = simulate_future(
            df=df.loc[df['state'] == state],
            features=features,
            scenario=scenario,
            sim_length=sim_length
        )

        preds = forecast_biodiversity(sim, model, features)

        yearly = preds.groupby("year")["pred_biodiv_z"].mean()

        all_runs.append(yearly.values)

    all_runs = np.array(all_runs)

    return pd.DataFrame({
        "year": np.arange(1, all_runs.shape[1] + 1),
        "mean": all_runs.mean(axis=0),
        "std": all_runs.std(axis=0),
        "lower": all_runs.mean(axis=0) - all_runs.std(axis=0),
        "upper": all_runs.mean(axis=0) + all_runs.std(axis=0),
    })