# Model Card v0 — Day 2 Logistic Regression

**Use case:** Educational binary classification demos (synthetic blobs; tiny HR churn sample).  
**Data:** Synthetic blobs (balanced); toy SQLite HR table (`data/hr.db`).  
**Features:** Standardized numeric; one-hot encoded `dept`.  
**Model:** Logistic Regression (custom NumPy and scikit-learn parity).  
**Metrics:** Accuracy on held-out split; decision boundary plot.  
**Limitations:** Tiny, non-representative data; not production-ready.  
**Ethics:** HR use can encode bias; do not deploy without fairness/privacy review.
