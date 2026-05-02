from prefect import flow, task
from prefect.logging import get_run_logger
import subprocess
import sys
import os

BASE = os.path.join(os.path.dirname(__file__), '..')

@task(name="Data Validation", retries=2, retry_delay_seconds=10)
def validate_data():
    logger = get_run_logger()
    logger.info("Validating data in PostgreSQL...")
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(
        'postgresql://postgres:1234@localhost:5432/vitalwatch_db'
    )
    count = pd.read_sql(
        "SELECT COUNT(*) as c FROM patient_features", engine
    ).iloc[0]['c']
    assert count > 700000, f"Not enough rows: {count}"
    logger.info(f"Data valid: {count:,} rows found")
    return count

@task(name="Feature Engineering", retries=1)
def run_feature_engineering():
    logger = get_run_logger()
    logger.info("Running feature engineering...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'feature_engineering.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Feature engineering failed: {result.stderr}")
    logger.info("Feature engineering complete")
    return True

@task(name="Model Training", retries=1)
def run_model_training():
    logger = get_run_logger()
    logger.info("Training all ML models...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'train_models.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")
    logger.info("Model training complete")
    return True

@task(name="Time Series Training", retries=1)
def run_timeseries():
    logger = get_run_logger()
    logger.info("Running time series training...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'time_series.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Time series failed: {result.stderr}")
    logger.info("Time series complete")
    return True

@task(name="Recommendation System", retries=1)
def run_recommendation():
    logger = get_run_logger()
    logger.info("Running recommendation system...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'recomendations.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Recommendation failed: {result.stderr}")
    logger.info("Recommendation system complete")
    return True

@task(name="Association Rules", retries=1)
def run_association_rules():
    logger = get_run_logger()
    logger.info("Running association rules...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'association_rules.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Association rules failed: {result.stderr}")
    logger.info("Association rules complete")
    return True

@task(name="ML Tests", retries=1)
def run_tests():
    logger = get_run_logger()
    logger.info("Running ML pipeline tests...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'tests', 'test_pipeline.py')],
        capture_output=True, text=True
    )
    logger.info(result.stdout[-500:])
    if result.returncode != 0:
        raise Exception(f"Tests failed: {result.stderr}")
    logger.info("All tests passed")
    return True

@task(name="Send Notification")
def send_notification(success: bool, message: str):
    logger = get_run_logger()
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Pipeline {status}: {message}")
    # Discord webhook would go here
    # For now just logs
    print(f"\nVitalWatch Pipeline {status}")
    print(f"Message: {message}")

# ══════════════════════════════════════════════
# MAIN PIPELINE FLOW
# ══════════════════════════════════════════════
@flow(name="VitalWatch ML Pipeline",
      description="Full ML training pipeline for sepsis prediction")
def vitalwatch_pipeline(
    run_training: bool = True,
    run_feature_eng: bool = True
):
    logger = get_run_logger()
    logger.info("Starting VitalWatch ML Pipeline...")

    try:
        # Step 1: Validate data
        row_count = validate_data()

        # Step 2: Feature engineering (optional)
        if run_feature_eng:
            run_feature_engineering()

        # Step 3: Train all models (parallel where possible)
        if run_training:
            run_model_training()
            run_timeseries()
            run_recommendation()
            run_association_rules()

        # Step 4: Run tests
        run_tests()

        # Step 5: Notify success
        send_notification(
            True,
            f"All models trained successfully on {row_count:,} rows"
        )
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        send_notification(False, str(e))
        raise

# ── Quick test flow (runs tests only, no retraining) ──
@flow(name="VitalWatch Test Only")
def test_only_flow():
    validate_data()
    run_tests()
    send_notification(True, "Tests passed — no retraining")

if __name__ == "__main__":
    vitalwatch_pipeline()