CREATE OR REPLACE VIEW churn_inference_view_example AS
WITH
snapshot_days AS (
    SELECT
        date_add('day', i, DATE '2024-01-01') AS d
    FROM UNNEST(
        sequence(
            0,
            date_diff('day', DATE '2024-01-01', current_date)
        )
    ) t(i)
),

snapshot_weeks AS (
    SELECT DISTINCT
        CAST(date_trunc('week', d) AS date) AS snapshot_date
    FROM snapshot_days
    WHERE d <= current_date
),

filtered_accounts AS (
    SELECT
        CAST(a.account_id AS varchar) AS account_id,
        a.account_type,
        a.account_status,
        a.customer_start_date,
        a.customer_end_date,
        CAST(a.total_enrollment AS double) AS total_enrollment,
        CAST(a.site_count AS double) AS site_count
    FROM analytics.account_snapshot a
    WHERE
        COALESCE(a.account_type, '') <> 'Parent'
        AND COALESCE(a.account_status, '') <> 'Prospect'
        AND a.account_status IS NOT NULL
),

accounts AS (
    SELECT DISTINCT
        account_id
    FROM filtered_accounts
),

account_static AS (
    SELECT
        account_id,
        total_enrollment,
        site_count,
        CASE
            WHEN site_count IS NULL AND total_enrollment IS NULL THEN NULL
            ELSE
                COALESCE(site_count, 0)
                + CASE WHEN COALESCE(total_enrollment, 0) > 3000 THEN 1 ELSE 0 END
                + CASE WHEN COALESCE(total_enrollment, 0) > 10000 THEN 1 ELSE 0 END
        END AS account_complexity_score
    FROM filtered_accounts
),

account_week_spine AS (
    SELECT
        a.account_id,
        w.snapshot_date
    FROM accounts a
    CROSS JOIN snapshot_weeks w
),

call_base AS (
    SELECT
        CAST(c.account_id AS varchar) AS account_id,
        CAST(c.call_end_time AS timestamp) AS call_end_time,
        CAST(c.handle_time AS double) AS handle_time,
        CAST(c.quality_score_pct AS double) AS quality_score_pct
    FROM analytics.support_calls c
    WHERE
        c.account_id IS NOT NULL
        AND c.call_end_time IS NOT NULL
),

call_features AS (
    SELECT
        s.account_id,
        s.snapshot_date,

        COUNT_IF(
            cb.call_end_time >= CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
            AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
        ) AS calls_30d,

        COUNT_IF(
            cb.call_end_time >= CAST(date_add('day', -90, s.snapshot_date) AS timestamp)
            AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
        ) AS calls_90d,

        COUNT_IF(
            cb.call_end_time >= CAST(date_add('day', -180, s.snapshot_date) AS timestamp)
            AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
        ) AS calls_180d,

        COUNT_IF(
            cb.call_end_time >= CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
            AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
        )
        -
        COUNT_IF(
            cb.call_end_time >= CAST(date_add('day', -60, s.snapshot_date) AS timestamp)
            AND cb.call_end_time < CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
        ) AS calls_trend_30d,

        AVG(
            CASE
                WHEN cb.call_end_time >= CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
                     AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
                THEN cb.handle_time
            END
        ) AS avg_handle_time_30d,

        AVG(
            CASE
                WHEN cb.call_end_time >= CAST(date_add('day', -90, s.snapshot_date) AS timestamp)
                     AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
                THEN cb.handle_time
            END
        ) AS avg_handle_time_90d,

        AVG(
            CASE
                WHEN cb.call_end_time >= CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
                     AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
                THEN cb.quality_score_pct
            END
        ) AS avg_eval_score_30d,

        AVG(
            CASE
                WHEN cb.call_end_time >= CAST(date_add('day', -90, s.snapshot_date) AS timestamp)
                     AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
                THEN cb.quality_score_pct
            END
        ) AS avg_eval_score_90d,

        COUNT_IF(
            cb.call_end_time >= CAST(date_add('day', -90, s.snapshot_date) AS timestamp)
            AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
            AND cb.quality_score_pct < 80
        ) AS low_eval_count_90d,

        COALESCE(
            date_diff('day', CAST(MAX(CAST(cb.call_end_time AS date)) AS date), s.snapshot_date),
            9999
        ) AS days_since_last_call

    FROM account_week_spine s
    LEFT JOIN call_base cb
        ON cb.account_id = s.account_id
        AND cb.call_end_time < CAST(s.snapshot_date AS timestamp)
        AND cb.call_end_time >= CAST(date_add('day', -180, s.snapshot_date) AS timestamp)
    GROUP BY
        s.account_id,
        s.snapshot_date
),

survey_features AS (
    SELECT
        s.account_id,
        s.snapshot_date,

        COUNT_IF(
            cs.survey_date >= date_add('day', -30, s.snapshot_date)
            AND cs.survey_date < s.snapshot_date
        ) AS surveys_30d,

        COUNT_IF(
            cs.survey_date >= date_add('day', -90, s.snapshot_date)
            AND cs.survey_date < s.snapshot_date
        ) AS surveys_90d,

        COUNT_IF(
            cs.survey_date >= date_add('day', -180, s.snapshot_date)
            AND cs.survey_date < s.snapshot_date
        ) AS surveys_180d,

        COUNT_IF(
            cs.survey_date >= date_add('day', -30, s.snapshot_date)
            AND cs.survey_date < s.snapshot_date
        )
        -
        COUNT_IF(
            cs.survey_date >= date_add('day', -60, s.snapshot_date)
            AND cs.survey_date < date_add('day', -30, s.snapshot_date)
        ) AS surveys_trend_30d,

        MAX(cs.survey_date) AS last_survey_date,

        AVG(
            CASE
                WHEN cs.survey_date >= date_add('day', -30, s.snapshot_date)
                     AND cs.survey_date < s.snapshot_date
                THEN CAST(cs.survey_score AS double)
            END
        ) AS avg_quality_score_30d,

        AVG(
            CASE
                WHEN cs.survey_date >= date_add('day', -90, s.snapshot_date)
                     AND cs.survey_date < s.snapshot_date
                THEN CAST(cs.survey_score AS double)
            END
        ) AS avg_quality_score_90d,

        SUM(
            CASE
                WHEN cs.survey_date >= date_add('day', -30, s.snapshot_date)
                     AND cs.survey_date < s.snapshot_date
                THEN CAST(cs.fcr_eligible AS double)
                ELSE 0
            END
        ) AS fcr_eligible_30d,

        SUM(
            CASE
                WHEN cs.survey_date >= date_add('day', -30, s.snapshot_date)
                     AND cs.survey_date < s.snapshot_date
                THEN CAST(cs.fcr_positive AS double)
                ELSE 0
            END
        ) AS fcr_positive_30d,

        SUM(
            CASE
                WHEN cs.survey_date >= date_add('day', -90, s.snapshot_date)
                     AND cs.survey_date < s.snapshot_date
                THEN CAST(cs.fcr_eligible AS double)
                ELSE 0
            END
        ) AS fcr_eligible_90d,

        SUM(
            CASE
                WHEN cs.survey_date >= date_add('day', -90, s.snapshot_date)
                     AND cs.survey_date < s.snapshot_date
                THEN CAST(cs.fcr_positive AS double)
                ELSE 0
            END
        ) AS fcr_positive_90d

    FROM account_week_spine s
    LEFT JOIN analytics.customer_surveys cs
        ON cs.account_id = s.account_id
        AND cs.survey_date < s.snapshot_date
        AND cs.survey_date >= date_add('day', -180, s.snapshot_date)
    GROUP BY
        s.account_id,
        s.snapshot_date
),

survey_final AS (
    SELECT
        account_id,
        snapshot_date,
        surveys_30d,
        surveys_90d,
        surveys_180d,
        surveys_trend_30d,
        avg_quality_score_30d,
        avg_quality_score_90d,
        fcr_eligible_30d,
        fcr_positive_30d,
        CASE
            WHEN fcr_eligible_30d > 0 THEN fcr_positive_30d / fcr_eligible_30d
        END AS fcr_rate_30d,
        fcr_eligible_90d,
        fcr_positive_90d,
        CASE
            WHEN fcr_eligible_90d > 0 THEN fcr_positive_90d / fcr_eligible_90d
        END AS fcr_rate_90d,
        COALESCE(
            date_diff('day', CAST(last_survey_date AS date), snapshot_date),
            9999
        ) AS days_since_last_survey
    FROM survey_features
),

case_base AS (
    SELECT
        CAST(c.account_id AS varchar) AS account_id,
        TRY(
            CAST(
                from_iso8601_timestamp(
                    regexp_replace(
                        CAST(c.created_at AS varchar),
                        '([+-]\\d{2})(\\d{2})$',
                        '\\1:\\2'
                    )
                ) AS timestamp
            )
        ) AS created_ts
    FROM analytics.support_cases c
    WHERE
        c.account_id IS NOT NULL
        AND c.created_at IS NOT NULL
),

case_features AS (
    SELECT
        s.account_id,
        s.snapshot_date,

        COUNT_IF(
            cb.created_ts >= CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
            AND cb.created_ts < CAST(s.snapshot_date AS timestamp)
        ) AS cases_30d,

        COUNT_IF(
            cb.created_ts >= CAST(date_add('day', -90, s.snapshot_date) AS timestamp)
            AND cb.created_ts < CAST(s.snapshot_date AS timestamp)
        ) AS cases_90d,

        COUNT_IF(
            cb.created_ts >= CAST(date_add('day', -180, s.snapshot_date) AS timestamp)
            AND cb.created_ts < CAST(s.snapshot_date AS timestamp)
        ) AS cases_180d,

        COUNT_IF(
            cb.created_ts >= CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
            AND cb.created_ts < CAST(s.snapshot_date AS timestamp)
        )
        -
        COUNT_IF(
            cb.created_ts >= CAST(date_add('day', -60, s.snapshot_date) AS timestamp)
            AND cb.created_ts < CAST(date_add('day', -30, s.snapshot_date) AS timestamp)
        ) AS cases_trend_30d,

        COALESCE(
            date_diff('day', CAST(MAX(CAST(cb.created_ts AS date)) AS date), s.snapshot_date),
            9999
        ) AS days_since_last_case

    FROM account_week_spine s
    LEFT JOIN case_base cb
        ON cb.account_id = s.account_id
        AND cb.created_ts < CAST(s.snapshot_date AS timestamp)
        AND cb.created_ts >= CAST(date_add('day', -180, s.snapshot_date) AS timestamp)
    GROUP BY
        s.account_id,
        s.snapshot_date
),

feature_frame AS (
    SELECT
        s.account_id,
        s.snapshot_date AS data_snapshot_date,

        cf.calls_30d,
        cf.calls_90d,
        cf.calls_180d,
        cf.calls_trend_30d,
        cf.avg_handle_time_30d,
        cf.avg_handle_time_90d,
        cf.avg_eval_score_30d,
        cf.avg_eval_score_90d,
        cf.low_eval_count_90d,
        cf.days_since_last_call,

        sf.surveys_30d,
        sf.surveys_90d,
        sf.surveys_180d,
        sf.surveys_trend_30d,
        sf.avg_quality_score_30d,
        sf.avg_quality_score_90d,
        sf.fcr_eligible_30d,
        sf.fcr_positive_30d,
        sf.fcr_rate_30d,
        sf.fcr_eligible_90d,
        sf.fcr_positive_90d,
        sf.fcr_rate_90d,
        sf.days_since_last_survey,

        kf.cases_30d,
        kf.cases_90d,
        kf.cases_180d,
        kf.cases_trend_30d,
        kf.days_since_last_case,

        a.total_enrollment,
        a.site_count,
        a.account_complexity_score

    FROM account_week_spine s
    LEFT JOIN call_features cf
        ON cf.account_id = s.account_id
        AND cf.snapshot_date = s.snapshot_date
    LEFT JOIN survey_final sf
        ON sf.account_id = s.account_id
        AND sf.snapshot_date = s.snapshot_date
    LEFT JOIN case_features kf
        ON kf.account_id = s.account_id
        AND kf.snapshot_date = s.snapshot_date
    LEFT JOIN account_static a
        ON a.account_id = s.account_id
),

latest_snapshot_per_account AS (
    SELECT
        account_id,
        max(data_snapshot_date) AS data_snapshot_date
    FROM feature_frame
    WHERE data_snapshot_date <= current_date
    GROUP BY account_id
)

SELECT
    f.account_id,
    f.data_snapshot_date,
    CAST(current_date AS date) AS inference_run_date,

    f.calls_30d,
    f.calls_90d,
    f.calls_180d,
    f.calls_trend_30d,
    f.avg_handle_time_30d,
    f.avg_handle_time_90d,
    f.avg_eval_score_30d,
    f.avg_eval_score_90d,
    f.low_eval_count_90d,
    f.days_since_last_call,

    f.surveys_30d,
    f.surveys_90d,
    f.surveys_180d,
    f.surveys_trend_30d,
    f.avg_quality_score_30d,
    f.avg_quality_score_90d,
    f.fcr_eligible_30d,
    f.fcr_positive_30d,
    f.fcr_rate_30d,
    f.fcr_eligible_90d,
    f.fcr_positive_90d,
    f.fcr_rate_90d,
    f.days_since_last_survey,

    f.cases_30d,
    f.cases_90d,
    f.cases_180d,
    f.cases_trend_30d,
    f.days_since_last_case,

    f.total_enrollment,
    f.site_count,
    f.account_complexity_score

FROM feature_frame f
INNER JOIN latest_snapshot_per_account l
    ON l.account_id = f.account_id
    AND l.data_snapshot_date = f.data_snapshot_date;
