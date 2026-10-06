"""Construct the standard sample-accounting records used by task builders.

Callers calculate every count on their existing sample and in their existing
order. These helpers only name the output fields; they do not apply filters,
handle missing values, or change how simultaneous and sequential drops count.
"""


def construction_step(task_id, step, label, rows_before, rows_after, rows_dropped):
    return {
        "task_id": task_id,
        "step": step,
        "step_label": label,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_dropped,
    }


def filter_count(task_id, step, order, filter_id, condition, fail_count):
    return {
        "task_id": task_id,
        "step": step,
        "detail_order": order,
        "filter_id": filter_id,
        "pass_condition": condition,
        "fail_count": fail_count,
    }
