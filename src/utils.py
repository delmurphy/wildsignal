import pandas as pd
import numpy as np

def simulate_future(df, features, scenario, sim_length):
    
    """
    Parameters
    ----------
    df : pandas DataFrame
        the full historic dataframe on which the model was trained
    features : list
        a list of the feature names that were used to train the model
    scenario : str
        Climate change scenarios based on IPCC projections by year 2100
        SCENARIOS = {
            "best case": {
                "temp increase":1.5,
                "precipitation change": 0.1
            },
            "middle of the road": {
                "temp increase":4,
                "precipitation change": 0.2
            },
            "business as usual": {
                "temp increase":7,
                "precipitation change": 0.3
            },
        }
    sim_length : num
        how many years to simulate into the future
    """

    SCENARIOS = {
        "best case": {
            "temp increase":1.5,
            "precipitation change": 0.1
        },
        "middle of the road": {
            "temp increase":4,
            "precipitation change": 0.2
        },
        "business as usual": {
            "temp increase":7,
            "precipitation change": 0.3
        },
    }

    scenario = SCENARIOS[scenario]

    # Get historic baseline, mean, std for temp and precip per state and month
    # for calculating z scores and anomalies for future data
    # note - all on original scale

    baseline = df.groupby(['state', 'month']).agg(
        temp_anom_mean = ('temp_anomaly', 'mean'),
        temp_anom_std = ('temp_anomaly', 'std'),
        temp_baseline = ('baseline_temp', 'max'),
        precip_anom_mean = ('precip_prop_anomaly', 'mean'),
        precip_anom_std = ('precip_prop_anomaly', 'std'),
        precip_baseline = ('baseline_precip', 'max')
    ).reset_index()

    # expand baseline df to sim_length years 
    dfs = [baseline.assign(future_year_offset=i) for i in range(1, sim_length+1)]
    future_weather = pd.concat(dfs, ignore_index=True)

    # Set projected temp & precip change by 2100
    projected_temp_increase = scenario['temp increase'] #in degrees C
    projected_precip_change = scenario['precipitation change'] #proportional change e.g 20% wetter/drier (amplitude in sinusoidal calculation)

    # ---------------------------------------
    # Simulate future temperature anomalies
    #---------------------------------------
    # calculate yearly projected temp increase
    yearly_temp_increase = projected_temp_increase/(2100-2024)
    # calculate future temps with random noise (increasing noise variability over time)
    future_weather['simulated_temp'] = (future_weather['temp_baseline'] 
                                + (yearly_temp_increase * future_weather['future_year_offset'])
                                + np.random.normal(0, 0.5 + 0.05 * future_weather['future_year_offset'], 
                                                    size=len(future_weather)))

    # calculate z-scores based on baseline data
    future_weather['temp_anomaly'] = future_weather['simulated_temp'] - future_weather['temp_baseline']
    future_weather['temp_anom_z'] = ((future_weather['temp_anomaly'] - future_weather['temp_anom_mean'])
                                    / future_weather['temp_anom_std'])

    # ---------------------------------------
    # Simulate future precipitation anomalies
    #---------------------------------------
    # Define simulation horizon
    max_years = future_weather['future_year_offset'].max()

    # Seasonal amplitude grows over simulation period
    growth = future_weather['future_year_offset'] / max_years

    # Seasonal pattern (stronger over time)
    future_weather['seasonal_precip'] = (
        projected_precip_change
        * growth
        * np.cos(2 * np.pi * (future_weather['month'] - 1) / 12)
    )

    # simulate future precipitation with increasing noise variability over time
    future_weather['simulated_precip'] = (
        future_weather['precip_baseline']
        * (1 + future_weather['seasonal_precip'])
        * np.exp(np.random.normal(
            0,
            0.15 + 0.05 * future_weather['future_year_offset'],  # slightly reduced base noise
            size=len(future_weather)
        ))
    ).clip(lower=0)


    # note for precipitation, calculate relative anomaly (e.g. 40% wetter/drier than normal)
    future_weather['precip_anomaly'] = (
        (future_weather['simulated_precip'] - future_weather['precip_baseline'])
        / future_weather['precip_baseline']
    )

    future_weather['precip_anom_z'] = (
        (future_weather['precip_anomaly'] - future_weather['precip_anom_mean'])
        / future_weather['precip_anom_std']
    )


    # ---------------------------------------
    # Simulate future drought risk
    #---------------------------------------
    # simulate drought index
    future_weather['drought_index'] = future_weather['temp_anom_z'] - future_weather['precip_anom_z']

    # ---------------------------------------
    # Simulate future hot days per month
    #---------------------------------------
    #baseline_hot_days = df.loc[df['year']==2024, ['state', 'month', 'n_hot_days']]
    baseline_hot_days = df.groupby(['state', 'month'])['n_hot_days'].mean().reset_index()
    baseline_hot_days = baseline_hot_days.rename(columns={'n_hot_days': 'baseline_hot_days'})
    hot_day_sensitivity = 5 # extra hot days per +1°C
    future_weather = future_weather.merge(baseline_hot_days)
    # Seasonal pattern (more likely to have hot days in summer than winter, increasing in severity with time)
    # amplify summer hot days and suppress winter hot days
    season_weight = np.clip(
        np.cos(2 * np.pi * (future_weather['month'] - 7) / 12),
        0,
        None
    ) ** 3 #cube the result to constrain winter values even more while keeping a strong summer signal
    # simulte n_hot_days with increasing noise variability over time
    future_weather['n_hot_days'] = (
        (
            future_weather['baseline_hot_days']
            + hot_day_sensitivity * (future_weather['temp_anomaly']) #using simulated temp here takes account of long-term trend
        )
        * season_weight
        + np.random.normal(0, 0.3 + 0.05 * future_weather['future_year_offset'], 
                                                    size=len(future_weather))
    ).clip(0, 31).round()

    # ---------------------------------------
    # Simulate future heavy rain days per month
    #---------------------------------------
    baseline_rain_days = df.loc[df['year'] == 2024, ['state', 'month', 'heavy_rain_days']]
    baseline_rain_days = baseline_rain_days.rename(columns={'heavy_rain_days': 'baseline_rain_days'})
    future_weather = future_weather.merge(baseline_rain_days, on=['state', 'month'], how='left')
    rain_season_weight = (
        # 0.5 + 0.5 * cos -> rescales cosine from [-1, 1] to [0, 1]
        0.5 + 0.5 * np.cos(2 * np.pi * (future_weather['month'] - 1) / 12)
    ) ** 2

    rain_sensitivity = 12  # higher for more extreme effects - wetter winters, drier summers
    noise = np.random.normal(
        0,
        (0.2 + 0.02 * future_weather['future_year_offset'])  # grows over time
        * (0.5 + rain_season_weight),
        len(future_weather)
    )
    year_trend = 0.1 * future_weather['future_year_offset'] 

    future_weather['heavy_rain_days'] = (
        future_weather['baseline_rain_days']
        + rain_sensitivity * (future_weather['precip_anomaly'] ** 1.3) * rain_season_weight
        + year_trend
        + noise
    ).clip(0, 31).round()

    #-----------------------------------------
    # Calculate lagged features, cyclical months and log_n_obs
    #-----------------------------------------
    #get last 3 months of 2024
    latest_weather = df.loc[(df['year']==2024) & (df['month']>9)]

    #drop columns not needed for model
    #latest_weather = latest_weather[features]
    #latest_weather

    #concat future weather
    future_weather['year'] = 2024 + future_weather['future_year_offset']
    future_weather['year_offset'] = 20 + future_weather['future_year_offset']
    future_weather['month_sin'] = np.sin(2*np.pi * future_weather['month'] / 12)
    future_weather['month_cos'] = np.cos(2*np.pi * future_weather['month'] / 12)
    #add log_n_obs as mean of last 3 years per month per state
    recent = df[df['year'] >= df['year'].max() - 3]
    baseline_log_n_obs = (recent.groupby(['state', 'month'])['log_n_obs']
                        .mean()
                        .reindex(future_weather.set_index(['state', 'month']).index)
                        .values)
    future_weather['log_n_obs'] = (baseline_log_n_obs)
    future_weather = pd.concat([latest_weather, future_weather])

    future_weather = future_weather.sort_values(['state', 'year', 'month'])

    #calculate lagged features
    future_weather['temp_anom_z_lag1'] = future_weather.groupby('state')['temp_anom_z'].shift(1)
    future_weather['precip_anom_z_lag1'] = future_weather.groupby('state')['precip_anom_z'].shift(1)
    future_weather['n_hot_days_lag1'] = future_weather.groupby('state')['n_hot_days'].shift(1)
    future_weather['drought_index_lag1'] = future_weather.groupby('state')['drought_index'].shift(1)
    future_weather['heavy_rain_days_lag1'] = future_weather.groupby('state')['heavy_rain_days'].shift(1)


    #calculate rolling features
    # 3 month rolling mean (shift back one month to avoid including current month in mean)
    future_weather['temp_anom_z_roll3'] = future_weather.groupby('state')['temp_anom_z'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    future_weather['precip_anom_z_roll3'] = future_weather.groupby('state')['precip_anom_z'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    future_weather['n_hot_days_roll3'] = future_weather.groupby('state')['n_hot_days'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    future_weather['drought_index_roll3'] = future_weather.groupby('state')['drought_index'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    future_weather['heavy_rain_days_roll3'] = future_weather.groupby('state')['heavy_rain_days'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    #keep year and month in df for later reporting
    features_extended = features.copy()
    features_extended.extend(['year', 'month'])

    future_weather = future_weather[features_extended]

    #Remove rows from 2024 (year_offset == 20)
    future_weather = future_weather.loc[future_weather['year_offset']>20]
                                     


    return future_weather



def forecast_bioshocks(df, model, features, threshold):
    """
    Apply the model to the future weather data simulated with simulate_future() to predict biodiversity anomalies
    Return df with model predictions as proba values and presence/absence biodiversity_anomaly based on proba threshold

    Parameters
    ----------
    df : pandas DataFrame
        the simulated future weather dataframe returned by simulate_future()
    model : the trained model
    features : list
        a list of the feature names that were used to train the model
    threshold : float
        the optimised threshold for proba values (saved in the trained model bundle)
    
    """

    # Dataframe with only necessary features
    X = df[features]

    # Predict probability
    proba = model.predict_proba(X)[:, 1]

    #add predictions to df
    df['proba'] = proba
    df['biodiversity_anomaly'] = (df['proba'] >= threshold).astype(int)

    return df



def simulate_with_uncertainty(
    df,
    state,
    scenario,
    features,
    model,
    threshold,
    sim_length = 50,
    n_runs=50
):
    """
    
    """
    all_runs = []

    for i in range(n_runs):

        # 1. simulate weather
        sim = simulate_future(
            df=df.loc[df['state'] == state],
            features = features,
            scenario=scenario,
            sim_length = sim_length,
            #seed=i  # important for variation control (if supported)
        )

        # 2. predict anomalies
        preds = forecast_bioshocks(df = sim, model = model, features = features, threshold = threshold)

        # 3. aggregate per year
        yearly = preds.groupby("year")["biodiversity_anomaly"].sum()

        all_runs.append(yearly.values)

    # convert to array: shape (runs, years)
    all_runs = np.array(all_runs)

    mean = all_runs.mean(axis=0)
    std = all_runs.std(axis=0)

    years = np.arange(1, mean.shape[0] + 1)

    return pd.DataFrame({
        "year": years,
        "mean_anomalies": mean,
        "std_anomalies": std,
        "lower": mean - std,
        "upper": mean + std
    })