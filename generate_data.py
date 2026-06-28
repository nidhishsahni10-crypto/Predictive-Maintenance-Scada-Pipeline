import numpy as np
import pandas as pd

def generate_scada_dataset(n_samples=5000, sequence_length=10, drift=False):
    """
    Generates high-fidelity synthetic industrial SCADA data representing sensor loops.
    Injects anomalies, missing rows, and data distribution drift.
    """
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-05-01", periods=n_samples, freq="T")
    
    # Generate industrial telemetry streams using normal distributions
    vibration = np.random.normal(loc=50.0, scale=5.0, size=n_samples)      # Hz
    temperature = np.random.normal(loc=75.0, scale=3.0, size=n_samples)    # °C
    pressure = np.random.normal(loc=200.0, scale=12.0, size=n_samples)     # kPa
    energy_rate = np.random.normal(loc=15.0, scale=1.5, size=n_samples)    # kWh per unit
    
    # Introduce structural product variability tied to vibration anomalies
    yield_error = 0.05 * vibration + np.random.normal(0, 0.8, size=n_samples)
    
    # Target label for 5-classifier tuning (0 = High Quality, 1 = Out of Spec/High Variability)
    # High variability defined as yield error deviating past 3 standard deviations
    yield_variability_label = (np.abs(yield_error) > 3.2).astype(int)

    if drift:
        # Simulate gradual machine wear and sensor degradation (Data Drift)
        temperature += np.linspace(0.0, 10.0, n_samples)
        vibration += np.linspace(0.0, 6.0, n_samples)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'vibration': vibration,
        'temperature': temperature,
        'pressure': pressure,
        'energy_rate': energy_rate,
        'yield_error': yield_error,
        'high_variability': yield_variability_label
    })
    
    # Create rule-based thermal/vibration triggers to simulate an impending failure state
    # This gives the PyTorch LSTM explicit, learnable sequential correlations
    failure_probability = (df['temperature'] > 82).astype(int) + (df['vibration'] > 57).astype(int)
    df['failure_unplanned'] = (failure_probability >= 1).astype(int)
    
    # Inject 2.5% random NaN values into sensory inputs to test the cleaning pipeline
    nan_mask_vibration = np.random.rand(n_samples) < 0.025
    nan_mask_pressure = np.random.rand(n_samples) < 0.025
    df.loc[nan_mask_vibration, 'vibration'] = np.nan
    df.loc[nan_mask_pressure, 'pressure'] = np.nan
    
    return df

if __name__ == "__main__":
    print("[Executing Data Generator]")
    baseline_df = generate_scada_dataset(drift=False)
    production_df = generate_scada_dataset(drift=True)
    
    baseline_df.to_csv("scada_baseline.csv", index=False)
    production_df.to_csv("scada_production.csv", index=False)
    
    print(f" -> Baseline shape: {baseline_df.shape} | Anomalies: {baseline_df['failure_unplanned'].sum()}")
    print(f" -> Production shape: {production_df.shape} | Drift injected successfully.")