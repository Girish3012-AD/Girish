# Customer Churn KPIs & Metrics Definition

## 1. Churn Rate

**Definition**: Percentage of customers who churned in a given period

**Formula**:
```
Churn Rate = (Number of Churned Customers / Total Customers) × 100
```

**Target**: < 20% monthly
**Alert Threshold**: > 25%

---

## 2. Churn Risk Buckets

| Risk Level | Churn Probability | Action Priority |
|------------|-------------------|-----------------|
| **Low** | < 30% | Monitor |
| **Medium** | 30-50% | Engage |
| **High** | 50-70% | Retain |
| **Very High** | >= 70% | Immediate |

**Usage**: Segment customers for targeted retention campaigns

---

## 3. Revenue at Risk

**Definition**: Total MRR (Monthly Recurring Revenue) from high-risk customers

**Formula**:
```
Revenue at Risk = SUM(monthly_price) WHERE churn_probability >= 0.7
```

**Business Impact**: Prioritize retention efforts by revenue impact

---

## 4. Retention Action Success Rate

**Definition**: % of at-risk customers who were successfully retained after intervention

**Formula**:
```
Success Rate = (Retained after Action / Total Actions) × 100
```

**Tracking**: Compare churn rate before/after retention campaigns

**Target**: > 40% retention of high-risk customers

---

## 5. Customer Lifetime Value (CLV) at Risk

**Definition**: Estimated total revenue loss from predicted churners

**Formula**:
```
CLV at Risk = SUM(monthly_price × avg_customer_lifetime_months) 
WHERE churn_prediction = 1
```

**Assumption**: Avg customer lifetime = 18 months

---

## 6. Model Performance Metrics

### Accuracy
Percentage of correct predictions
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision
Of predicted churners, how many actually churned
```
Precision = TP / (TP + FP)
```

### Recall
Of actual churners, how many were predicted
```
Recall = TP / (TP + FN)
```

### ROC-AUC
Area under the ROC curve (discrimination ability)

**Target**: ROC-AUC > 0.80

---

## 7. Segment-wise Churn Rate

**Definition**: Churn rate broken down by customer segment

**Usage**: Identify which segments have highest risk

**Dimensions**:
- Plan type (Basic/Standard/Premium)
- Device type (Android/iOS/Web)
- Acquisition channel
- Location

---

## 8. Engagement Score

**Definition**: Composite score of customer activity

**Components**:
- Sessions (30d)
- Avg session duration
- Crashes (inverse)
- Support tickets (inverse)

**Formula**:
```
Engagement Score = (sessions × avg_minutes) / (1 + crashes + support_tickets)
```

**Correlation**: Higher engagement = Lower churn probability

---

## 9. Payment Health Score

**Definition**: Indicator of payment reliability

**Formula**:
```
Payment Health = successful_amount / (successful_amount + failed_payments_count × avg_price)
```

**Range**: 0-1 (higher is better)

---

## 10. Time to Churn

**Definition**: Days between prediction and actual churn

**Usage**: Measure lead time for retention actions

**Target**: Predict >= 30 days in advance for effective intervention
