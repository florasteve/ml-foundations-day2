# Model Card v0 — Day 2 Logistic Regression

**Use case:** Educational binary classification demos (synthetic data; tiny HR churn sample).  
**Data:** Synthetic Gaussian blobs (balanced); toy SQLite HR table (`data/hr.db`).  
**Features:** Standardized numeric; one-hot encoded categorical (`dept`).  
**Model:** Logistic Regression (custom NumPy and scikit-learn parity).  
**Metrics:** Accuracy on held-out split; visualize decision boundary.  
**Limitations:** Tiny, non-representative dataset; not production-ready.  
**Ethics:** HR scenarios can encode bias; never deploy without fairness assessment and privacy review.
