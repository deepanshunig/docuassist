# Recorded local evaluations

dev.json and holdout_v2.json measure the released prompt, identified by their matching prompt_sha256. holdout_v2 was authored after the prompt was frozen, has separate question groups, and was not used for implementation tuning. All collections contain only the five public fictional manuals.

final.json is a historical evaluation of a retired prompt with factual examples. Its scores do not describe the released implementation. The retirement reason and manual quality findings are documented in ../../docs/progress.md.

Retrieval coverage and answer-status agreement are not factual answer accuracy. Read generated answers and their passages before interpreting the scores. Rerunning a split replaces its report and measures the current implementation.
