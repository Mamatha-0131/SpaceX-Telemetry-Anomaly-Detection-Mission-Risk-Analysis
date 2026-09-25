import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the dataframe for fuel burn anomaly analysis.
    Ensures the presence of required columns, sorts by time,
    and derives burn rate if needed.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]  # Normalize column names

    # Ensure 'time_s' column exists
    if "time_s" not in df.columns:
        for alt in ["time", "t", "seconds"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "time_s"})
                break
    if "time_s" not in df.columns:
        raise ValueError("CSV must contain a 'time_s' column.")

    # Sort by time
    df = df.sort_values("time_s").reset_index(drop=True)

    # Derive burn rate if missing
    if "burn_rate_kg_s" not in df.columns:
        if "fuel_mass_kg" in df.columns:
            t = df["time_s"].values
            m = df["fuel_mass_kg"].values
            dm_dt = np.gradient(m, t)  # rate of change of mass
            burn_rate = -dm_dt         # burn rate is negative slope
            df["burn_rate_kg_s"] = burn_rate
        else:
            raise ValueError("Need either 'burn_rate_kg_s' or 'fuel_mass_kg' to compute burn rate.")

    # Add default stage column if missing
    if "stage" not in df.columns:
        df["stage"] = 1

    # Clean NaNs and infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "burn_rate_kg_s"])
    return df


def compute_expected_curve(df: pd.DataFrame, smoothing_window: int = 8) -> pd.DataFrame:
    """
    Compute the nominal burn rate curve per stage using rolling median
    and rolling mean smoothing.
    """
    out = df.copy()
    win = max(int(smoothing_window), 1)
    pieces = []

    for stage_val, g in out.groupby("stage", sort=True):
        g = g.sort_values("time_s").copy()
        med = g["burn_rate_kg_s"].rolling(window=win, min_periods=1, center=True).median()
        exp = med.rolling(window=win, min_periods=1, center=True).mean()
        g["expected_burn_rate"] = exp
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True).sort_values("time_s").reset_index(drop=True)
    out["residual"] = out["burn_rate_kg_s"] - out["expected_burn_rate"]
    out["zscore_stage"] = out.groupby("stage")["residual"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )
    return out


def detect_anomalies(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    use_iforest: bool = True,
    contamination: float = 0.02
) -> pd.DataFrame:
    """
    Detect anomalies based on Z-score and optionally IsolationForest.
    """
    out = df.copy()

    # Z-score detection
    z_flag = out["zscore_stage"].abs() >= z_thresh

    # IsolationForest detection
    if use_iforest:
        iso_flags = np.zeros(len(out), dtype=bool)
        for stage_val, g in out.groupby("stage", sort=True):
            X = np.column_stack([g["time_s"].values, g["residual"].values])
            if len(g) >= 10:
                clf = IsolationForest(
                    n_estimators=200,
                    contamination=min(max(contamination, 1e-6), 0.5),
                    random_state=42,
                )
                y = clf.fit_predict(X)  # -1 = anomaly
                stage_flags = (y == -1)
            else:
                stage_flags = np.zeros(len(g), dtype=bool)
            iso_flags[g.index] = stage_flags
    else:
        iso_flags = np.zeros(len(out), dtype=bool)

    out["is_anomaly"] = z_flag | iso_flags
    return out


def assign_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a severity level to each telemetry point based on
    anomaly status, Z-score and percentage deviation.

    Severity levels:
        Normal
        Minor
        Moderate
        High
        Critical
    """

    out = df.copy()

    # Calculate percentage deviation
    out["deviation_%"] = 100.0 * (
        (out["burn_rate_kg_s"] - out["expected_burn_rate"])
        / out["expected_burn_rate"].replace(0, np.nan)
    )

    # Absolute values are used because both positive and
    # negative deviations can be abnormal.
    abs_z = out["zscore_stage"].abs()
    abs_deviation = out["deviation_%"].abs()

    # Start every point as Normal
    out["severity"] = "Normal"

    # Only anomaly points can receive higher severity
    anomaly_mask = out["is_anomaly"]

    # Minor anomaly
    out.loc[
        anomaly_mask,
        "severity"
    ] = "Minor"

    # Moderate anomaly
    moderate_mask = anomaly_mask & (
        (abs_z >= 3.0) |
        (abs_deviation >= 10.0)
    )

    out.loc[
        moderate_mask,
        "severity"
    ] = "Moderate"

    # High anomaly
    high_mask = anomaly_mask & (
        (abs_z >= 4.0) |
        (abs_deviation >= 15.0)
    )

    out.loc[
        high_mask,
        "severity"
    ] = "High"

    # Critical anomaly
    critical_mask = anomaly_mask & (
        (abs_z >= 5.0) |
        (abs_deviation >= 20.0)
    )

    out.loc[
        critical_mask,
        "severity"
    ] = "Critical"

    return out

# ============================================================
# AUTOMATIC ANOMALY EXPLANATION
# ============================================================

def generate_anomaly_explanation(row) -> str:
    """
    Generate a human-readable explanation for an anomaly
    using severity, fuel burn deviation, Z-score and stage.
    """

    severity = row.get("severity", "Normal")
    deviation = row.get("deviation_%", 0)
    zscore = row.get("zscore_stage", 0)
    stage = row.get("stage", "Unknown")
    time_s = row.get("time_s", 0)

    # Normal telemetry
    if severity == "Normal":
        return "Telemetry is within the expected operating profile."

    # Determine whether fuel consumption is higher or lower
    if deviation > 0:
        direction = "higher"
        effect = "increased fuel consumption"
    else:
        direction = "lower"
        effect = "reduced fuel consumption"

    deviation_abs = abs(deviation)
    zscore_abs = abs(zscore)

    # Generate explanation based on severity
    if severity == "Minor":

        explanation = (
            f"Minor anomaly detected during Stage {stage}. "
            f"Fuel burn is {deviation_abs:.1f}% {direction} "
            f"than expected at {time_s:.1f}s, indicating "
            f"a small deviation from the nominal fuel profile."
        )

    elif severity == "Moderate":

        explanation = (
            f"Moderate anomaly detected during Stage {stage}. "
            f"Fuel burn is {deviation_abs:.1f}% {direction} "
            f"than expected. The deviation may indicate "
            f"{effect} or an unusual operating condition."
        )

    elif severity == "High":

        explanation = (
            f"High-severity anomaly detected during Stage {stage}. "
            f"Fuel burn is {deviation_abs:.1f}% {direction} "
            f"than expected with a Z-score of {zscore_abs:.2f}. "
            f"This represents a significant deviation from "
            f"the nominal mission profile and requires investigation."
        )

    else:

        explanation = (
            f"Critical anomaly detected during Stage {stage}. "
            f"Fuel burn is {deviation_abs:.1f}% {direction} "
            f"than expected with a Z-score of {zscore_abs:.2f}. "
            f"This extreme deviation may indicate a serious "
            f"fuel-system or propulsion-related abnormality."
        )

    return explanation

def calculate_risk_score(df: pd.DataFrame) -> dict:
    """
    Calculate an overall mission risk score from 0 to 100.

    Risk contribution:
        Normal   = 0
        Minor    = 1
        Moderate = 3
        High     = 7
        Critical = 12

    The score is normalized using the total number of
    telemetry points.
    """

    if df.empty:
        return {
            "risk_score": 0,
            "risk_level": "Low",
            "raw_risk": 0,
            "minor": 0,
            "moderate": 0,
            "high": 0,
            "critical": 0,
        }

    # Count severity levels
    minor = int(
        (df["severity"] == "Minor").sum()
    )

    moderate = int(
        (df["severity"] == "Moderate").sum()
    )

    high = int(
        (df["severity"] == "High").sum()
    )

    critical = int(
        (df["severity"] == "Critical").sum()
    )

    # Risk weights
    raw_risk = (
        minor * 1
        + moderate * 3
        + high * 7
        + critical * 12
    )

    # Number of telemetry observations
    total_points = len(df)

    # Normalize risk to 0–100.
    #
    # 12 points per observation represents the
    # maximum theoretical risk.
    risk_score = (
        raw_risk / (total_points * 12)
    ) * 100

    risk_score = round(
        min(max(risk_score, 0), 100)
    )

    # Determine risk level
    if risk_score < 20:

        risk_level = "Low"

    elif risk_score < 40:

        risk_level = "Moderate"

    elif risk_score < 70:

        risk_level = "High"

    else:

        risk_level = "Critical"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "raw_risk": raw_risk,
        "minor": minor,
        "moderate": moderate,
        "high": high,
        "critical": critical,
    }

def summarize_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary dataframe of detected anomalies.
    """
    anom = df[df["is_anomaly"]].copy()
    if anom.empty:
     return pd.DataFrame(
    columns=[
        "time_s",
        "stage",
        "actual_burn",
        "expected_burn",
        "deviation_%",
        "zscore",
        "severity",
        "explanation"
    ]
)

    anom["deviation_%"] = 100.0 * (
        (anom["burn_rate_kg_s"] - anom["expected_burn_rate"]) /
        anom["expected_burn_rate"].replace(0, np.nan)
    )

    anom = anom[[
    "time_s",
    "stage",
    "burn_rate_kg_s",
    "expected_burn_rate",
    "deviation_%",
    "zscore_stage",
    "severity",
    "explanation"
]]
    anom.columns = [
    "time_s",
    "stage",
    "actual_burn",
    "expected_burn",
    "deviation_%",
    "zscore",
    "severity",
    "explanation"
]

    return anom.sort_values("time_s").reset_index(drop=True)

# ============================================================
# MULTIVARIATE TELEMETRY SIMULATION
# ============================================================

def generate_multivariate_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simulated/derived telemetry parameters
    for project demonstration.

    IMPORTANT:
    These are simulated values, NOT real SpaceX telemetry.
    """

    out = df.copy()

    # Reproducible random generator
    rng = np.random.default_rng(42)

    # Time
    time = pd.to_numeric(
        out["time_s"],
        errors="coerce"
    ).fillna(0)

    # Fuel burn
    burn = pd.to_numeric(
        out["burn_rate_kg_s"],
        errors="coerce"
    ).fillna(0)

    # Normalize fuel burn
    burn_min = burn.min()
    burn_max = burn.max()

    if burn_max != burn_min:
        burn_norm = (
            (burn - burn_min)
            / (burn_max - burn_min)
        )
    else:
        burn_norm = pd.Series(
            0.5,
            index=out.index
        )

    # --------------------------------------------------------
    # NORMAL TELEMETRY
    # --------------------------------------------------------

    out["thrust_kN"] = (
        800
        + burn_norm * 500
        + rng.normal(0, 8, len(out))
    )

    out["temperature_K"] = (
        290
        + burn_norm * 80
        + rng.normal(0, 2, len(out))
    )

    out["pressure_bar"] = (
        45
        + burn_norm * 15
        + rng.normal(0, 0.8, len(out))
    )

    altitude = (
        0.5 * time ** 2
        + rng.normal(0, 30, len(out))
    )

    out["altitude_m"] = np.maximum(
        altitude,
        0
    )

    velocity = (
        5 * time
        + rng.normal(0, 5, len(out))
    )

    out["velocity_m_s"] = np.maximum(
        velocity,
        0
    )

    out["acceleration_m_s2"] = (
        5
        + burn_norm * 2
        + rng.normal(0, 0.15, len(out))
    )

    # --------------------------------------------------------
    # SIMULATED MULTIVARIATE ABNORMAL EVENTS
    # --------------------------------------------------------

    # Select a few telemetry points for demonstration
    anomaly_indices = [
        int(len(out) * 0.25),
        int(len(out) * 0.50),
        int(len(out) * 0.75),
    ]

    for idx in anomaly_indices:

        if idx < len(out):

            # Event 1: Increased thrust + temperature
            out.loc[
                out.index[idx],
                "thrust_kN"
            ] *= 1.25

            out.loc[
                out.index[idx],
                "temperature_K"
            ] *= 1.15

            # Event 2: Pressure abnormality
            out.loc[
                out.index[idx],
                "pressure_bar"
            ] *= 1.20

            # Event 3: Acceleration abnormality
            out.loc[
                out.index[idx],
                "acceleration_m_s2"
            ] *= 1.30

    return out

# ============================================================
# MULTIVARIATE ANOMALY DETECTION
# ============================================================

def detect_multivariate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect abnormal behavior across multiple telemetry
    parameters using Z-score analysis.

    This is a demonstration layer using simulated telemetry.
    """

    out = df.copy()

    telemetry_columns = [
        "thrust_kN",
        "temperature_K",
        "pressure_bar",
        "altitude_m",
        "velocity_m_s",
        "acceleration_m_s2",
    ]

    # Calculate Z-score for each telemetry parameter
    for column in telemetry_columns:

        mean_value = out[column].mean()
        std_value = out[column].std()

        if std_value == 0 or pd.isna(std_value):
            out[f"{column}_zscore"] = 0
        else:
            out[f"{column}_zscore"] = (
                (out[column] - mean_value)
                / std_value
            )

        # --------------------------------------------------------
    # Identify which parameters are abnormal
    # --------------------------------------------------------

    zscore_columns = [
        f"{column}_zscore"
        for column in telemetry_columns
    ]

    abnormal_parameter_lists = []

    for _, row in out.iterrows():

        abnormal_parameters = []

        for column, z_column in zip(
            telemetry_columns,
            zscore_columns
        ):

            if abs(row[z_column]) >= 2.5:

                parameter_name = {
                    "thrust_kN": "Thrust",
                    "temperature_K": "Temperature",
                    "pressure_bar": "Pressure",
                    "altitude_m": "Altitude",
                    "velocity_m_s": "Velocity",
                    "acceleration_m_s2": "Acceleration",
                }[column]

                abnormal_parameters.append(
                    parameter_name
                )

        abnormal_parameter_lists.append(
            abnormal_parameters
        )

    # Store readable parameter names
    out["abnormal_parameters"] = [
        ", ".join(parameters)
        if parameters
        else "None"
        for parameters in abnormal_parameter_lists
    ]

    # Count abnormal parameters
    out["multivariate_anomaly_count"] = (
        out[zscore_columns].abs() >= 2.5
    ).sum(axis=1)

    # At least 2 abnormal parameters
    # means multivariate anomaly
    out["multivariate_anomaly"] = (
        out["multivariate_anomaly_count"] >= 2
    )

    return out