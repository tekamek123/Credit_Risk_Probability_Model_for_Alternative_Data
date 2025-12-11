# Credit Risk Probability Model for Alternative Data

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

## Project Overview

This project develops a Credit Scoring Model for Bati Bank's buy-now-pay-later service partnership with an eCommerce platform. The model leverages alternative data (transactional behavioral patterns) to assess credit risk and predict default probability for customers who may not have traditional credit histories.

## Business Need

Bati Bank, a leading financial service provider with over 10 years of experience, is partnering with an eCommerce company to enable a buy-now-pay-later service. This requires a credit scoring system that can:

1. **Define a proxy variable** to categorize users as high risk (bad) or low risk (good)
2. **Select observable features** that are good predictors of default
3. **Develop a model** that assigns risk probability for new customers
4. **Convert risk probability** to a credit score
5. **Predict optimal loan amount and duration** based on risk assessment

## Credit Scoring Business Understanding

### Introduction

Credit scoring is the process of assigning a quantitative measure to a potential borrower as an estimate of how likely default will happen in the future. Traditionally, creditors build credit scoring models using statistical techniques to analyze various information of previous borrowers in relation to their loan performance. The result is either a score that represents the creditworthiness of an applicant or a prediction of whether an applicant will default in the future.

The definition of default in the context of credit scoring must comply with the **Basel II Capital Accord**, which prescribes the minimum amount of regulatory capital an institution must hold to provide a safety cushion against unexpected losses. Under the Advanced Internal Ratings-Based (A-IRB) approach, financial institutions can build risk models for three key risk parameters:

- **Probability of Default (PD)**: The likelihood that a loan will not be repaid and will fall into default
- **Loss Given Default (LGD)**: The estimated economic loss, expressed as a percentage of exposure, if an obligor goes into default
- **Exposure at Default (EAD)**: A measure of the monetary exposure should an obligor go into default

### Key Questions and Answers

#### 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord fundamentally transforms how financial institutions approach credit risk management by requiring them to develop and validate their own internal risk models. This regulatory framework has profound implications for model development:

**Regulatory Requirements:**
- **Model Validation**: Basel II mandates that institutions must validate their internal models through rigorous backtesting, stress testing, and ongoing monitoring. Regulators must be able to understand and verify the model's methodology, assumptions, and performance.
- **Capital Adequacy**: The accord requires institutions to hold regulatory capital based on their risk estimates (PD, LGD, EAD). Inaccurate or poorly documented models can lead to insufficient capital reserves, exposing the institution to regulatory penalties and financial instability.
- **Model Governance**: Basel II emphasizes the need for comprehensive model documentation, including data sources, feature selection rationale, model assumptions, and performance metrics. This documentation must be accessible to both internal stakeholders and external regulators.

**Impact on Model Interpretability:**
- **Regulatory Scrutiny**: Regulators need to understand how the model works to approve its use for capital calculation. Black-box models that cannot be explained may face regulatory rejection, regardless of their predictive performance.
- **Audit Trail**: Financial institutions must maintain a clear audit trail showing how decisions were made. Interpretable models allow auditors and regulators to trace from input features to final risk estimates.
- **Risk Management**: Senior management and risk committees need to understand model outputs to make informed strategic decisions. Interpretable models facilitate better risk governance and strategic planning.
- **Model Maintenance**: Over time, models must be recalibrated and updated. Interpretable models make it easier to identify when and why model performance degrades, enabling proactive maintenance.

**Documentation Requirements:**
- **Model Development Process**: Clear documentation of data preprocessing, feature engineering, and model selection criteria
- **Assumptions and Limitations**: Explicit documentation of model assumptions, data quality issues, and known limitations
- **Performance Metrics**: Comprehensive reporting of model performance across different segments and time periods
- **Change Management**: Documentation of model updates, version control, and impact assessments

In summary, Basel II's emphasis on risk measurement creates a regulatory imperative for interpretable and well-documented models. The inability to explain model decisions can result in regulatory non-compliance, financial penalties, and operational restrictions.

#### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

**Why Proxy Variables are Necessary:**

In this project, we have transactional data from an eCommerce platform but no direct loan default history. Traditional credit scoring relies on historical loan performance data (defaults, late payments, etc.), which is unavailable in this context. Therefore, we must create a proxy variable that approximates credit risk using available behavioral data.

**Approach: RFM-Based Proxy Variable**

The key innovation lies in transforming behavioral data into a predictive risk signal using **Recency, Frequency, and Monetary (RFM) patterns**:

- **Recency (R)**: How recently a customer made a transaction - indicates engagement and active relationship
- **Frequency (F)**: How often a customer transacts - indicates loyalty and consistent behavior
- **Monetary (M)**: How much a customer spends - indicates value and financial capacity

By analyzing these patterns in conjunction with other observable behaviors (fraud indicators, transaction patterns, product categories), we can engineer a proxy that categorizes users as:
- **High Risk (Bad)**: Customers with patterns suggesting potential default (e.g., irregular transactions, fraud history, declining engagement)
- **Low Risk (Good)**: Customers with patterns suggesting reliable repayment behavior (e.g., consistent transactions, no fraud, stable spending)

**Potential Business Risks of Proxy-Based Predictions:**

1. **Proxy Mismatch Risk**: The proxy variable may not accurately reflect actual loan default behavior. Behavioral patterns in eCommerce transactions may differ from loan repayment behavior. For example:
   - A customer who frequently returns items might be categorized as high risk, but may actually be a responsible borrower
   - A customer with irregular eCommerce activity might have stable income and be a good credit risk

2. **Model Drift Risk**: The relationship between the proxy and actual default may change over time due to:
   - Changes in customer behavior patterns
   - Economic conditions affecting both eCommerce and credit behavior differently
   - Evolution of the eCommerce platform's user base

3. **Selection Bias Risk**: The proxy may be based on a subset of behaviors that don't generalize to the broader population of potential borrowers. This could lead to:
   - Over-rejection of creditworthy customers (Type I error)
   - Over-acceptance of risky customers (Type II error)

4. **Regulatory Risk**: Regulators may question the validity of proxy-based models, especially if:
   - The proxy cannot be clearly linked to actual default risk
   - The model cannot demonstrate predictive power on actual loan performance
   - The proxy introduces discriminatory biases

5. **Business Model Risk**: Incorrect risk assessments can lead to:
   - **Credit Losses**: Approving loans to high-risk customers who default
   - **Opportunity Costs**: Rejecting creditworthy customers, losing potential revenue
   - **Reputation Damage**: Poor credit decisions affecting customer trust and brand reputation

6. **Data Quality Risk**: The proxy depends on the quality and completeness of transactional data:
   - Missing or incomplete transaction histories
   - Data quality issues affecting RFM calculations
   - Changes in data collection methods over time

**Mitigation Strategies:**

- **Validation Framework**: Establish a validation framework to monitor the proxy's predictive power as actual loan performance data becomes available
- **Conservative Approach**: Initially adopt a conservative risk assessment approach, gradually refining as real-world performance data accumulates
- **Regular Recalibration**: Continuously monitor and recalibrate the proxy based on emerging loan performance data
- **Segmentation**: Develop different proxies for different customer segments to improve accuracy
- **Documentation**: Thoroughly document the proxy definition, rationale, and limitations for regulatory and internal review

#### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

**Simple, Interpretable Models: Logistic Regression with Weight of Evidence (WoE)**

**Advantages:**
- **Regulatory Compliance**: Highly interpretable models are preferred by regulators. Each coefficient can be explained, and the relationship between features and outcomes is transparent
- **Auditability**: Easy to audit and validate. Regulators and internal auditors can understand exactly how each feature contributes to the risk score
- **Stability**: Less prone to overfitting and more stable across different data samples. Predictions are more consistent and reliable
- **Business Understanding**: Business stakeholders can understand and trust the model. This facilitates adoption and strategic decision-making
- **Debugging**: When model performance degrades, it's easier to identify which features are causing issues
- **Compliance with Fair Lending**: Easier to demonstrate that the model doesn't discriminate against protected classes, as each feature's contribution is explicit
- **Basel II Alignment**: Meets regulatory requirements for model interpretability and documentation

**Disadvantages:**
- **Limited Predictive Power**: May not capture complex non-linear relationships and feature interactions present in the data
- **Feature Engineering Dependency**: Requires extensive feature engineering (like WoE transformation) to capture non-linear relationships
- **Performance Ceiling**: May have lower predictive accuracy compared to more complex models, potentially leading to higher credit losses or opportunity costs

**Complex, High-Performance Models: Gradient Boosting (XGBoost, LightGBM)**

**Advantages:**
- **Superior Predictive Performance**: Can capture complex non-linear relationships, feature interactions, and patterns that simpler models miss
- **Automatic Feature Engineering**: Can learn complex feature interactions without explicit engineering
- **Better Risk Discrimination**: Higher AUC and better separation between good and bad customers can lead to:
  - More accurate risk-based pricing
  - Better portfolio risk management
  - Reduced credit losses through better risk identification

**Disadvantages:**
- **Regulatory Challenges**: Black-box nature makes it difficult to explain predictions to regulators, potentially leading to:
  - Regulatory rejection or restrictions
  - Increased scrutiny and longer approval processes
  - Requirements for additional validation and monitoring
- **Interpretability Issues**: Difficult to explain why a specific customer received a particular score, which can lead to:
  - Challenges in customer communication and dispute resolution
  - Difficulty in identifying and fixing model biases
  - Reduced trust from business stakeholders
- **Overfitting Risk**: More prone to overfitting, especially with limited data, requiring careful validation
- **Maintenance Complexity**: More difficult to maintain, debug, and update. When performance degrades, identifying root causes is challenging
- **Fair Lending Compliance**: Harder to demonstrate that the model doesn't discriminate, as feature contributions are not explicit

**Hybrid Approach: Best of Both Worlds**

Many financial institutions adopt a **hybrid approach** that balances interpretability and performance:

1. **Two-Stage Modeling**:
   - Use gradient boosting for feature selection and interaction discovery
   - Retrain a logistic regression model using the most important features and interactions identified
   - This provides better performance than pure logistic regression while maintaining interpretability

2. **Ensemble Methods**:
   - Combine predictions from both simple and complex models
   - Use simple models for regulatory reporting and complex models for operational decisions
   - Weight the models based on their performance and interpretability needs

3. **Post-Hoc Interpretability**:
   - Use SHAP (SHapley Additive exPlanations) or LIME to explain gradient boosting predictions
   - While not as interpretable as logistic regression, this provides some explainability
   - However, regulators may still prefer inherently interpretable models

4. **Segmented Approach**:
   - Use simple models for low-risk segments where interpretability is critical
   - Use complex models for high-risk segments where predictive power is more important
   - This balances regulatory needs with business performance

**Recommendation for This Project:**

Given the regulatory context and Basel II requirements, we recommend:

1. **Primary Model**: Start with **Logistic Regression with WoE transformation** to ensure regulatory compliance and interpretability
2. **Benchmark Model**: Develop a **Gradient Boosting model** as a benchmark to understand the performance ceiling
3. **Validation**: Compare both models' performance and document the trade-offs
4. **Iterative Improvement**: As the model gains regulatory acceptance and actual loan performance data becomes available, consider hybrid approaches that maintain interpretability while improving performance

This approach ensures regulatory compliance while providing a path for performance optimization as the model matures and gains acceptance.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Fields

- **TransactionId**: Unique transaction identifier on the platform
- **BatchId**: Unique number assigned to a batch of transactions for processing
- **AccountId**: Unique number identifying the customer on the platform
- **SubscriptionId**: Unique number identifying the customer subscription
- **CustomerId**: Unique identifier attached to Account
- **CurrencyCode**: Country currency
- **CountryCode**: Numerical geographical code of the country
- **ProviderId**: Source provider of the Item bought
- **ProductId**: Item name being bought
- **ProductCategory**: ProductIds are organized into these broader product categories
- **ChannelId**: Identifies if the customer used web, Android, IOS, pay later, or checkout
- **Amount**: Value of the transaction. Positive for debits from the customer account and negative for credits into the customer account
- **Value**: Absolute value of the amount
- **TransactionStartTime**: Transaction start time
- **PricingStrategy**: Category of Xente's pricing structure for merchants
- **FraudResult**: Fraud status of transaction 1 = yes, 0 = No

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

(To be updated as the project progresses)

## References

- [Basel II Capital Accord](https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html)
- [Credit Scoring through Predictive Modelling for Basel II](https://www.linkedin.com/pulse/credit-scoring-through-predictive-modelling-basel-ii-using-odeneye/)
- [Alternative Credit Scoring Approaches](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)

## License

(To be determined)

