# Credit Risk Modeling for Bati Bank

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements

The Basel II Accord emphasizes three pillars for banking supervision, with Pillar 1 focusing on minimum capital requirements calculated based on credit risk. This directly impacts our model needs:

- **Interpretability**: Regulators require models whose decisions can be explained to ensure fair lending practices and compliance with anti-discrimination laws.
  
- **Documentation**: Comprehensive model documentation is needed to demonstrate the soundness of our risk measurement approach during audits.
  
- **Risk Sensitivity**: The model must properly differentiate risk levels to calculate appropriate capital reserves (8% of risk-weighted assets under Basel II).

### 2. Proxy Variable Necessity and Risks

Since we lack direct default data from the eCommerce platform:

**Why a proxy is necessary:**
- Without historical loan performance data, we cannot observe actual defaults
- Customer transaction behavior (RFM patterns) can serve as a reasonable analog for creditworthiness
- This approach follows alternative data credit scoring methodologies recognized under Basel II

**Potential business risks:**
- **Misclassification risk**: Good customers labeled as high-risk may be denied credit, losing potential revenue
- **False negatives**: Bad customers labeled as low-risk could lead to higher default rates
- **Regulatory scrutiny**: Proxy-based models may face additional validation requirements
- **Model drift risk**: eCommerce behavior patterns may change over time differently than credit behavior

### 3. Model Complexity Trade-offs

| Factor               | Simple Model (Logistic Regression + WoE)       | Complex Model (Gradient Boosting)         |
|----------------------|-----------------------------------------------|------------------------------------------|
| **Interpretability** | High - Clear feature weights                 | Low - Harder to explain predictions      |
| **Performance**      | Potentially lower predictive accuracy        | Typically higher accuracy                |
| **Compliance**       | Easier to validate with regulators           | May require additional documentation     |
| **Implementation**   | Straightforward implementation               | More complex hyperparameter tuning       |
| **Maintenance**      | Easier to monitor and update                 | Requires more sophisticated monitoring   |

**Recommended approach:** Given regulatory requirements, we recommend starting with an interpretable model and only increasing complexity if the business case justifies the additional compliance burden.