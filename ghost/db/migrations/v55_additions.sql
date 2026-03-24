-- Ghost v5.5 Database Migration
-- Run against existing Ghost PostgreSQL database

-- Trade Quality Score columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tqs_total FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tqs_grade VARCHAR(20);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tqs_weakest_dimension VARCHAR(50);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tqs_dimension_scores JSONB;

-- Stop calibration columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS sweep_ticks_actual FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS stop_structural_level FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS sweep_buffer_used_ticks FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS lowest_price_after_stop FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS did_price_recover BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS hunt_predicted BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS hunt_occurred BOOLEAN;

-- TP calibration columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp1_hit BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp2_hit BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp3_hit BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp1_overshoot_ticks FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS highest_price_reached FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS lowest_price_reached FLOAT;

-- Entry calibration columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS optimal_entry_sources JSONB;

-- Footprint columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS footprint_composite FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS footprint_signals JSONB;

-- Ensemble columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ensemble_consensus VARCHAR(30);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ensemble_mm_rate FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ensemble_qwen_rate FLOAT;

-- Payout columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS payout_phase VARCHAR(30);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS payout_risk_multiplier FLOAT;

-- Checklist columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS checklist_passed BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS checklist_failed_conds JSONB;

-- Additional context columns
ALTER TABLE trades ADD COLUMN IF NOT EXISTS amd_phase VARCHAR(20);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS weekly_profile_type VARCHAR(30);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS void_present BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS pyramid_strength FLOAT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS mtf_aligned BOOLEAN;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS setup_type VARCHAR(30);

-- Shadow signals table (TQS 80-84 trades logged for analysis)
CREATE TABLE IF NOT EXISTS shadow_signals (
    id SERIAL PRIMARY KEY,
    signal_time TIMESTAMPTZ DEFAULT NOW(),
    instrument VARCHAR(10),
    direction VARCHAR(10),
    tqs_total FLOAT,
    tqs_grade VARCHAR(20),
    rejection_reason VARCHAR(100),
    checklist_failed JSONB,
    would_have_outcome VARCHAR(20),
    would_have_pnl FLOAT
);

-- Calibration audit trail
CREATE TABLE IF NOT EXISTS calibration_runs (
    id SERIAL PRIMARY KEY,
    run_time TIMESTAMPTZ DEFAULT NOW(),
    instrument VARCHAR(10),
    sweep_buffer_before FLOAT,
    sweep_buffer_after FLOAT,
    tp_extension_before FLOAT,
    tp_extension_after FLOAT,
    session_bucket INT,
    sample_size INT,
    change_made BOOLEAN DEFAULT FALSE
);

-- Weekly profile history
CREATE TABLE IF NOT EXISTS weekly_profiles (
    id SERIAL PRIMARY KEY,
    week_start DATE,
    instrument VARCHAR(10),
    profile_type VARCHAR(30),
    bias VARCHAR(10),
    weekly_draw VARCHAR(20),
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_trades_instrument_outcome ON trades(instrument, outcome);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_shadow_instrument ON shadow_signals(instrument, signal_time);
