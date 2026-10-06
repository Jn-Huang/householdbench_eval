"""Shared production functions for HouseholdBench.

I/O and publication live in io; table contracts and typed reads in table_schema;
prompt text and record checks in prompt_rendering; answer domains and validation
in responses. Sampling, macro context, evaluation, and XGBoost retain their own
modules. Source-specific transformations stay in cps, michigan, psid, and
psid_events.
"""
