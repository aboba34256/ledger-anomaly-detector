# Evaluation results

Averaged over 5 random synthetic ledgers of 10,000 entries (298 seeded irregularities each, 3% rate).

## Recall by irregularity type

| Irregularity | Seeded (avg) | Caught by intended rule | Caught by any rule | Recall |
|---|---|---|---|---|
| benford_vendor | 45 | 45 | 45 | 100% |
| duplicate | 76 | 76 | 76 | 100% |
| off_hours | 38 | 38 | 38 | 100% |
| outlier | 25 | 22 | 22 | 86% |
| round_number | 38 | 38 | 38 | 100% |
| self_approved | 38 | 38 | 38 | 100% |
| threshold | 38 | 38 | 38 | 100% |

## Overall

| Alert level | Recall | Precision | False-positive rate | Entries flagged (avg) |
|---|---|---|---|---|
| Any flag | 98.9% | 92.1% | 0.26% | 320 |
| Medium risk or higher | 59.2% | 87.5% | 0.26% | 202 |
