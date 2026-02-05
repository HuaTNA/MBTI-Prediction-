-- MBTI Prediction System Database Schema
-- For collecting new user data to improve models

-- ================ QUESTION MANAGEMENT ================

-- Question bank table
CREATE TABLE question_bank (
    question_id INT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    dimension VARCHAR(3) NOT NULL CHECK (dimension IN ('I/E', 'S/N', 'T/F', 'J/P')),
    version VARCHAR(20) DEFAULT '1.0',
    language VARCHAR(10) DEFAULT 'en',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Question usage tracking (which questions were shown to which user)
CREATE TABLE question_usage (
    usage_id VARCHAR(36) PRIMARY KEY,
    prediction_id VARCHAR(36),  -- Will reference predictions table (defined below)
    question_id INT REFERENCES question_bank(question_id),
    question_order INT,  -- Order in which question was presented (1, 2, 3...)
    user_response_text TEXT,  -- User's actual response
    response_length INT,  -- Length of response for analytics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Question performance metrics (A/B testing support)
CREATE TABLE question_metrics (
    metric_id VARCHAR(36) PRIMARY KEY,
    question_id INT REFERENCES question_bank(question_id),
    total_usage INT DEFAULT 0,
    avg_response_length FLOAT,
    prediction_correlation FLOAT,  -- How well responses correlate with final prediction
    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================ USER MANAGEMENT ================

-- Users table
CREATE TABLE users (
    user_id VARCHAR(36) PRIMARY KEY,  -- UUID for anonymity
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    consent_given BOOLEAN DEFAULT FALSE,  -- Whether user consents to data being used for training
    consent_timestamp TIMESTAMP,
    last_active TIMESTAMP,
    -- Do not store real names/emails or other sensitive information
    user_hash VARCHAR(64) UNIQUE  -- For identifying repeat users while protecting privacy
);

-- Predictions table
CREATE TABLE predictions (
    prediction_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id) ON DELETE CASCADE,
    session_id VARCHAR(36),  -- Multiple predictions in the same session
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Prediction results
    predicted_mbti VARCHAR(4) NOT NULL,  -- e.g., 'INTJ'
    confidence_score FLOAT,  -- 0-1

    -- Dimension scores
    e_i_score FLOAT,  -- Extraversion vs Introversion
    s_n_score FLOAT,  -- Sensing vs Intuition
    t_f_score FLOAT,  -- Thinking vs Feeling
    j_p_score FLOAT,  -- Judging vs Perceiving

    -- Model version tracking
    text_model_version VARCHAR(50),
    emotion_model_version VARCHAR(50),

    -- Data quality flags
    is_valid_for_training BOOLEAN DEFAULT FALSE,
    quality_score FLOAT  -- Data quality score (optional)
);

-- Text inputs table
CREATE TABLE text_inputs (
    text_id VARCHAR(36) PRIMARY KEY,
    prediction_id VARCHAR(36) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Text content (encrypted or hashed)
    text_content TEXT,  -- Consider encrypted storage
    text_length INT,
    language VARCHAR(10),  -- 'en', 'zh', etc.

    -- Text features (for quick queries)
    sentiment_score FLOAT,
    word_count INT,

    -- Preprocessed text (optional)
    cleaned_text TEXT
);

-- Emotion records table
CREATE TABLE emotion_records (
    emotion_id VARCHAR(36) PRIMARY KEY,
    prediction_id VARCHAR(36) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Detected emotion
    primary_emotion VARCHAR(20),  -- Anger, Happiness, Neutral, etc.
    emotion_confidence FLOAT,

    -- Probability distribution for all emotions (JSON format)
    emotion_distribution JSON,  -- {"Happiness": 0.8, "Neutral": 0.15, ...}

    -- Face image metadata (not storing actual images to save space and protect privacy)
    face_detected BOOLEAN,
    face_count INT,
    image_quality FLOAT,  -- Image quality score

    -- Optional: store thumbnail (low resolution) for debugging
    -- thumbnail_blob BLOB  -- 32x32 or 64x64 for verification only
);

-- User feedback table (critical!)
CREATE TABLE user_feedback (
    feedback_id VARCHAR(36) PRIMARY KEY,
    prediction_id VARCHAR(36) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- User's self-reported actual MBTI
    actual_mbti VARCHAR(4),  -- User's "correct" answer

    -- Feedback type
    feedback_type VARCHAR(20),  -- 'accurate', 'partially_correct', 'incorrect', 'unsure'
    accuracy_rating INT CHECK (accuracy_rating BETWEEN 1 AND 5),  -- 1-5 star rating

    -- Detailed feedback
    dimension_feedback JSON,  -- User feedback for each dimension
    comments TEXT,  -- User comments

    -- Whether user has taken a standard test
    tested_before BOOLEAN,
    test_source VARCHAR(50),  -- '16Personalities', 'Official MBTI', etc.

    -- Data usage permission
    allow_training_use BOOLEAN DEFAULT FALSE
);

-- Training batches table (for tracking retraining)
CREATE TABLE training_batches (
    batch_id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Training data statistics
    total_samples INT,
    text_samples INT,
    emotion_samples INT,
    multimodal_samples INT,

    -- Data source
    source_start_date TIMESTAMP,
    source_end_date TIMESTAMP,

    -- Model performance
    model_type VARCHAR(50),  -- 'text_ml', 'text_bert', 'emotion_cnn', etc.
    accuracy FLOAT,
    f1_score FLOAT,

    -- Model file path
    model_path VARCHAR(255),

    -- Status
    status VARCHAR(20),  -- 'training', 'completed', 'failed'
    notes TEXT
);

-- Data quality audit table
CREATE TABLE data_quality_audit (
    audit_id VARCHAR(36) PRIMARY KEY,
    prediction_id VARCHAR(36) REFERENCES predictions(prediction_id),
    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    audited_by VARCHAR(50),  -- 'auto' or auditor ID

    -- Quality checks
    text_quality_pass BOOLEAN,  -- Whether text is meaningful
    emotion_quality_pass BOOLEAN,  -- Whether emotion recognition is reliable
    label_quality_pass BOOLEAN,  -- Whether label is trustworthy

    -- Issue flags
    issues JSON,  -- ['spam', 'low_quality_image', 'contradictory_label']

    -- Approval status
    approved_for_training BOOLEAN DEFAULT FALSE
);

-- Index optimization
CREATE INDEX idx_predictions_user ON predictions(user_id);
CREATE INDEX idx_predictions_created ON predictions(created_at);
CREATE INDEX idx_predictions_mbti ON predictions(predicted_mbti);
CREATE INDEX idx_feedback_prediction ON user_feedback(prediction_id);
CREATE INDEX idx_feedback_actual_mbti ON user_feedback(actual_mbti);
CREATE INDEX idx_text_prediction ON text_inputs(prediction_id);
CREATE INDEX idx_emotion_prediction ON emotion_records(prediction_id);

-- View: Training-ready data
CREATE VIEW training_ready_data AS
SELECT
    p.prediction_id,
    p.predicted_mbti,
    f.actual_mbti,
    f.accuracy_rating,
    t.text_content,
    e.emotion_distribution,
    p.created_at
FROM predictions p
JOIN user_feedback f ON p.prediction_id = f.prediction_id
LEFT JOIN text_inputs t ON p.prediction_id = t.prediction_id
LEFT JOIN emotion_records e ON p.prediction_id = e.prediction_id
WHERE f.allow_training_use = TRUE
  AND p.is_valid_for_training = TRUE
  AND f.actual_mbti IS NOT NULL;

-- View: Model performance tracking over time
CREATE VIEW model_performance_over_time AS
SELECT
    DATE(created_at) as date,
    predicted_mbti,
    COUNT(*) as prediction_count,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN is_valid_for_training THEN 1 END) as valid_samples
FROM predictions
GROUP BY DATE(created_at), predicted_mbti
ORDER BY date DESC;
