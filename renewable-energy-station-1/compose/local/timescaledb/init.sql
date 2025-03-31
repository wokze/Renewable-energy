-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create tables first
CREATE TABLE IF NOT EXISTS battery_instances (
    id SERIAL PRIMARY KEY,
    battery_type VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    capacity FLOAT NOT NULL,
    max_charge_rate FLOAT,
    max_discharge_rate FLOAT,
    manufactured_date VARCHAR,
    installation_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS battery_metrics (
    time TIMESTAMPTZ NOT NULL,
    battery_id INTEGER NOT NULL,
    current_charge FLOAT DEFAULT 0.0,
    voltage FLOAT,
    temperature FLOAT,
    cycle_count INTEGER,
    health FLOAT,
    status TEXT,
    last_eoc TIMESTAMPTZ,
    next_eoc TIMESTAMPTZ,
    energy_in FLOAT DEFAULT 0.0,
    energy_out FLOAT DEFAULT 0.0,
    efficiency FLOAT
);

-- Create hypertable
SELECT create_hypertable('battery_metrics', 'time', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Add indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_battery_metrics_battery_id 
    ON battery_metrics(battery_id);

CREATE INDEX IF NOT EXISTS idx_battery_metrics_time_battery_id 
    ON battery_metrics(time DESC, battery_id);

-- Add foreign key constraint
ALTER TABLE battery_metrics 
    ADD CONSTRAINT fk_battery_metrics_instance 
    FOREIGN KEY (battery_id) 
    REFERENCES battery_instances(id);

-- Create continuous aggregate view
CREATE MATERIALIZED VIEW IF NOT EXISTS battery_metrics_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    battery_id,
    avg(current_charge) as avg_charge_level,
    avg(temperature) as avg_temperature,
    sum(energy_in) as total_energy_in,
    sum(energy_out) as total_energy_out,
    avg(efficiency) as avg_efficiency,
    avg(health) as avg_health,
    max(cycle_count) as max_cycle_count
FROM battery_metrics
GROUP BY bucket, battery_id;

-- Set retention policy
SELECT add_retention_policy('battery_metrics', INTERVAL '1 year');

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('battery_metrics_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');