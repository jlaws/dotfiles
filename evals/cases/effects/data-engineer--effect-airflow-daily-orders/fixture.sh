set -euo pipefail
mkdir -p dags
cat > dags/daily_orders.py <<'PY'
from datetime import datetime

import psycopg2
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

conn = psycopg2.connect("dbname=warehouse")
cur = conn.cursor()
cur.execute("SELECT max(order_date) FROM daily_orders")
LAST_LOADED = cur.fetchone()[0]


def load_orders():
    cur.execute(
        "INSERT INTO daily_orders "
        "SELECT * FROM staging_orders WHERE order_date = '2026-11-01'"
    )
    conn.commit()


with DAG(
    "daily_orders",
    start_date=datetime.now(),
    schedule="@daily",
    default_args={"depends_on_past": True, "retries": 3},
) as dag:
    wait = FileSensor(
        task_id="wait_for_file",
        filepath="/data/orders.csv",
        poke_interval=30,
        mode="poke",
    )
    load = PythonOperator(task_id="load", python_callable=load_orders)
    wait >> load
PY
