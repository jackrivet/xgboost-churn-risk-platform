CREATE OR REPLACE VIEW churn_training_view_example AS
WITH
asof AS (
    SELECT
        CAST(
            date_add('day', -1, date_add('month', 1, date_trunc('month', m)))
            AS date
        ) AS as_of_date
    FROM UNNEST(
        sequence(
            date_trunc('month', DATE '2024-01-01'),
            date_trunc('month', date_add('month', -1, current_date)),
            INTERVAL '1' MONTH
        )
    ) t(m)
),

filtered_accounts AS (
    SELECT
        a.account_id,
        a.account_type,
        a.account_status,
        TRY_CAST(a.customer_start_date AS date) AS active_start_date,
        TRY_CAST(a.customer_end_date AS date) AS active_end_date,
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

at_risk AS (
    SELECT
        a.account_id,
        s.as_of_date,
        fa.active_start_date,
        fa.active_end_date,
        fa.total_enrollment,
        fa.site_count,
        CASE
            WHEN fa.site_count IS NULL AND fa.total_enrollment IS NULL THEN NULL
            ELSE
                COALESCE(fa.site_count, 0)
                + CASE WHEN COALESCE(fa.total_enrollment, 0) > 3000 THEN 1 ELSE 0 END
                + CASE WHEN COALESCE(fa.total_enrollment, 0) > 10000 THEN 1 ELSE 0 END
        END AS account_complexity_score
    FROM accounts a
    CROSS JOIN asof s
    LEFT JOIN filtered_accounts fa
        ON fa.account_id = a.account_id
    WHERE
        fa.active_start_date IS NOT NULL
        AND fa.active_start_date <= s.as_of_date
        AND (
            fa.active_end_date IS NULL
            OR fa.active_end_date > s.as_of_date
        )
),

labels AS (
    SELECT
        account_id,
        as_of_date,
        CAST(
            CASE
                WHEN active_end_date IS NOT NULL
                     AND active_end_date > as_of_date
                     AND active_end_date <= date_add('day', 365, as_of_date)
                THEN 1
                ELSE 0
            END AS integer
        ) AS label_churn_365d
    FROM at_risk
),

call_base AS (
    SELECT
        c.account_id,
        CAST(c.call_end_time AS timestamp) AS call_end_time,
        CAST(c.handle_time AS double) AS handle_time,
        CAST(c.quality_score_pct AS double) AS quality_score_pct
    FROM analytics.support_calls c
),

call_features AS (
    SELECT
        r.account_id,
        r.as_of_date,

        COUNT_IF(
            cb.call_end_time >= date_add('day', -30, CAST(r.as_of_date AS timestamp))
            AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
        ) AS calls_30d,

        COUNT_IF(
            cb.call_end_time >= date_add('day', -90, CAST(r.as_of_date AS timestamp))
            AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
        ) AS calls_90d,

        COUNT_IF(
            cb.call_end_time >= date_add('day', -180, CAST(r.as_of_date AS timestamp))
            AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
        ) AS calls_180d,

        COUNT_IF(
            cb.call_end_time >= date_add('day', -30, CAST(r.as_of_date AS timestamp))
            AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
        )
        -
        COUNT_IF(
            cb.call_end_time >= date_add('day', -60, CAST(r.as_of_date AS timestamp))
            AND cb.call_end_time < date_add('day', -30, CAST(r.as_of_date AS timestamp))
        ) AS calls_trend_30d,

        AVG(
            CASE
                WHEN cb.call_end_time >= date_add('day', -30, CAST(r.as_of_date AS timestamp))
                     AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
                THEN cb.handle_time
            END
        ) AS avg_handle_time_30d,

        AVG(
            CASE
                WHEN cb.call_end_time >= date_add('day', -90, CAST(r.as_of_date AS timestamp))
                     AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
                THEN cb.handle_time
            END
        ) AS avg_handle_time_90d,

        AVG(
            CASE
                WHEN cb.call_end_time >= date_add('day', -30, CAST(r.as_of_date AS timestamp))
                     AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
                THEN cb.quality_score_pct
            END
        ) AS avg_eval_score_30d,

        AVG(
            CASE
                WHEN cb.call_end_time >= date_add('day', -90, CAST(r.as_of_date AS timestamp))
                     AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
                THEN cb.quality_score_pct
            END
        ) AS avg_eval_score_90d,

        COUNT_IF(
            cb.call_end_time >= date_add('day', -90, CAST(r.as_of_date AS timestamp))
            AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
            AND cb.quality_score_pct < 80
        ) AS low_eval_count_90d,

        date_diff('day', MAX(CAST(cb.call_end_time AS date)), r.as_of_date) AS days_since_last_call

    FROM at_risk r
    LEFT JOIN call_base cb
        ON cb.account_id = r.account_id
        AND cb.call_end_time < CAST(r.as_of_date AS timestamp)
        AND cb.call_end_time >= date_add('day', -180, CAST(r.as_of_date AS timestamp))
    GROUP BY
        r.account_id,
        r.as_of_date
),

survey_features AS (
    SELECT
        r.account_id,
        r.as_of_date,

        COUNT_IF(
            s.survey_date >= date_add('day', -30, r.as_of_date)
            AND s.survey_date < r.as_of_date
        ) AS surveys_30d,

        COUNT_IF(
            s.survey_date >= date_add('day', -90, r.as_of_date)
            AND s.survey_date < r.as_of_date
        ) AS surveys_90d,

        COUNT_IF(
            s.survey_date >= date_add('day', -180, r.as_of_date)
            AND s.survey_date < r.as_of_date
        ) AS surveys_180d,

        COUNT_IF(
            s.survey_date >= date_add('day', -30, r.as_of_date)
            AND s.survey_date < r.as_of_date
        )
        -
        COUNT_IF(
            s.survey_date >= date_add('day', -60, r.as_of_date)
            AND s.survey_date < date_add('day', -30, r.as_of_date)
        ) AS surveys_trend_30d,

        AVG(
            CASE
                WHEN s.survey_date >= date_add('day', -30, r.as_of_date)
                     AND s.survey_date < r.as_of_date
                THEN CAST(s.survey_score AS double)
            END
        ) AS avg_quality_score_30d,

        AVG(
            CASE
                WHEN s.survey_date >= date_add('day', -90, r.as_of_date)
                     AND s.survey_date < r.as_of_date
                THEN CAST(s.survey_score AS double)
            END
        ) AS avg_quality_score_90d,

        SUM(
            CASE
                WHEN s.survey_date >= date_add('day', -30, r.as_of_date)
                     AND s.survey_date < r.as_of_date
                THEN CAST(s.fcr_eligible AS double)
                ELSE 0
            END
        ) AS fcr_eligible_30d,

        SUM(
            CASE
                WHEN s.survey_date >= date_add('day', -30, r.as_of_date)
                     AND s.survey_date < r.as_of_date
                THEN CAST(s.fcr_positive AS double)
                ELSE 0
            END
        ) AS fcr_positive_30d,

        SUM(
            CASE
                WHEN s.survey_date >= date_add('day', -90, r.as_of_date)
                     AND s.survey_date < r.as_of_date
                THEN CAST(s.fcr_eligible AS double)
                ELSE 0
            END
        ) AS fcr_eligible_90d,

        SUM(
            CASE
                WHEN s.survey_date >= date_add('day', -90, r.as_of_date)
                     AND s.survey_date < r.as_of_date
                THEN CAST(s.fcr_positive AS double)
                ELSE 0
            END
        ) AS fcr_positive_90d,

        date_diff('day', MAX(CAST(s.survey_date AS date)), r.as_of_date) AS days_since_last_survey

    FROM at_risk r
    LEFT JOIN analytics.customer_surveys s
        ON s.account_id = r.account_id
        AND s.survey_date < r.as_of_date
        AND s.survey_date >= date_add('day', -180, r.as_of_date)
    GROUP BY
        r.account_id,
        r.as_of_date
),

survey_final AS (
    SELECT
        account_id,
        as_of_date,
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
        days_since_last_survey
    FROM survey_features
),

case_features AS (
    SELECT
        r.account_id,
        r.as_of_date,

        COUNT_IF(
            CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date)
                BETWEEN date_add('day', -30, r.as_of_date)
                AND date_add('day', -1, r.as_of_date)
        ) AS cases_30d,

        COUNT_IF(
            CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date)
                BETWEEN date_add('day', -90, r.as_of_date)
                AND date_add('day', -1, r.as_of_date)
        ) AS cases_90d,

        COUNT_IF(
            CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date)
                BETWEEN date_add('day', -180, r.as_of_date)
                AND date_add('day', -1, r.as_of_date)
        ) AS cases_180d,

        COUNT_IF(
            CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date)
                BETWEEN date_add('day', -30, r.as_of_date)
                AND date_add('day', -1, r.as_of_date)
        )
        -
        COUNT_IF(
            CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date)
                BETWEEN date_add('day', -60, r.as_of_date)
                AND date_add('day', -31, r.as_of_date)
        ) AS cases_trend_30d,

        date_diff(
            'day',
            MAX(CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date)),
            r.as_of_date
        ) AS days_since_last_case

    FROM at_risk r
    LEFT JOIN analytics.support_cases c
        ON c.account_id = r.account_id
        AND CAST(TRY(from_iso8601_timestamp(CAST(c.created_at AS varchar))) AS date) < r.as_of_date
    GROUP BY
        r.account_id,
        r.as_of_date
)

SELECT
    l.account_id,
    l.as_of_date,

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

    ccf.cases_30d,
    ccf.cases_90d,
    ccf.cases_180d,
    ccf.cases_trend_30d,
    ccf.days_since_last_case,

    r.total_enrollment,
    r.site_count,
    r.account_complexity_score,

    l.label_churn_365d

FROM labels l
INNER JOIN at_risk r
    ON r.account_id = l.account_id
    AND r.as_of_date = l.as_of_date
LEFT JOIN call_features cf
    ON cf.account_id = l.account_id
    AND cf.as_of_date = l.as_of_date
LEFT JOIN survey_final sf
    ON sf.account_id = l.account_id
    AND sf.as_of_date = l.as_of_date
LEFT JOIN case_features ccf
    ON ccf.account_id = l.account_id
    AND ccf.as_of_date = l.as_of_date;
