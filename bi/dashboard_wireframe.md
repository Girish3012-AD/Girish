# Churn Analytics Dashboard - Tableau Wireframe

## Dashboard Overview

**Purpose**: Executive dashboard for customer churn monitoring and retention strategy

**Target Users**: Business stakeholders, retention team, customer success managers

**Refresh**: Daily

---

## Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│                   CHURN ANALYTICS DASHBOARD                 │
│                      [Date Filter] [Plan Type Filter]       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Churn    │  │ Revenue  │  │ High Risk│  │ Avg Churn│    │
│  │ Rate     │  │ at Risk  │  │ Count    │  │ Prob     │    │
│  │ 22%      │  │ ₹4.5M    │  │  850     │  │  34%     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Churn Probability Distribution    │   Risk Category Pie    │
│  [Histogram Chart]                  │   [Pie Chart]          │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Churn Rate by Segment              │  Top 10 Actions        │
│  [Bar Chart]                        │  [Horizontal Bars]     │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│              Top 50 High-Risk Customers Table                │
│  [Customer ID | Risk | Plan | MRR | Action | Segment]       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Components Specification

### 1. **KPI Cards** (Top Row)

#### Card 1: Churn Rate
- **Metric**: `SUM(churn_prediction) / COUNT(customer_id) × 100`
- **Format**: Percentage, 1 decimal
- **Color**: Red if > 25%, Yellow if 20-25%, Green if < 20%

#### Card 2: Revenue at Risk
- **Metric**: `SUM(monthly_price) WHERE churn_probability >= 0.7`
- **Format**: Currency (₹)
- **Color**: Red gradient

#### Card 3: High Risk Count
- **Metric**: `COUNT(customer_id) WHERE risk_category = 'Very High'`
- **Format**: Integer with comma separator
- **Color**: Orange

#### Card 4: Avg Churn Probability
- **Metric**: `AVG(churn_probability) × 100`
- **Format**: Percentage
- **Color**: Blue

---

### 2. **Churn Probability Distribution** (Left, Middle)

**Chart Type**: Histogram

**X-axis**: `churn_probability` (bins of 0.1)
**Y-axis**: Count of customers

**Color**: Gradient from green (low) to red (high)

**Purpose**: Show distribution of risk across customer base

---

### 3. **Risk Category Breakdown** (Right, Middle)

**Chart Type**: Pie Chart

**Dimension**: `risk_category`
**Measure**: Count of customers

**Colors**:
- Low: Green (#2ecc71)
- Medium: Yellow (#f1c40f)
- High: Orange (#e67e22)
- Very High: Red (#e74c3c)

**Labels**: Show percentage and count

---

### 4. **Churn Rate by Segment** (Left, Bottom)

**Chart Type**: Vertical Bar Chart

**Dimensions**: Create calculated fields for:
- Plan Type
- Device Type
- Acquisition Channel
- Location (Top 10)

**Measure**: Churn Rate %

**Sort**: Descending by churn rate

**Color**: Red gradient (darker = higher churn)

---

### 5. **Top 10 Retention Actions** (Right, Bottom)

**Chart Type**: Horizontal Bar Chart

**Dimension**: `recommended_action`
**Measure**: Count of customers

**Filter**: Top 10 by count

**Color**: Blue gradient

**Purpose**: Show which actions are most recommended

---

### 6. **High-Risk Customer Table** (Bottom)

**Table Columns**:
1. Customer ID
2. Churn Probability (%, sorted desc)
3. Risk Category (with color pill)
4. Plan Type
5. Monthly Price (₹)
6. Recommended Action
7. Segment ID

**Filters**: `churn_probability >= 0.5`

**Sort**: By churn_probability DESC

**Limit**: Top 50

**Conditional Formatting**:
- Churn probability: Color scale red gradient
- Risk category: Pills with category colors

---

## Filters Panel (Top Right)

### Global Filters:
1. **Date Range**: Calendar selector (default: last 30 days)
2. **Plan Type**: Multi-select (Basic/Standard/Premium)
3. **Device Type**: Multi-select
4. **Risk Category**: Multi-select
5. **Location**: Multi-select (top cities)

**Apply to**: All worksheets

---

## Data Source

**File**: `bi/tableau_ready_dataset.csv`

**Key Columns**:
- `customer_id` (Dimension, String)
- `churn_probability` (Measure, Float)
- `churn_prediction` (Dimension, Integer)
- `risk_category` (Dimension, String)
- `recommended_action` (Dimension, String)
- `plan_type` (Dimension, String)
- `monthly_price` (Measure, Float)
- `segment_id` (Dimension, Integer)
- Behavior metrics (Measures)

---

## Tableau Implementation Steps

### Step 1: Import Data
```
Data → New Data Source → Text File
Select: bi/tableau_ready_dataset.csv
```

### Step 2: Create Calculated Fields

```
// Churn Rate
SUM([Churn Prediction]) / COUNT([Customer Id]) * 100

// Revenue at Risk
SUM(IF [Churn Probability] >= 0.7 THEN [Monthly Price] ELSE 0 END)

// High Risk Flag
IF [Risk Category] = "Very High" THEN 1 ELSE 0 END
```

### Step 3: Build Worksheets
1. Create KPI cards using `TEXT` marks
2. Create histogram using `BINS` for churn_probability
3. Create pie chart with risk_category
4. Create bar charts for segments
5. Create table view with formatting

### Step 4: Assemble Dashboard
- Drag worksheets onto dashboard canvas
- Add filters
- Configure interactivity
- Apply formatting/colors

### Step 5: Publish
- Tableau Public or Tableau Server
- Schedule refresh if connected to live data

---

## Alternative: Power BI

**Dataset Import**: Same CSV file

**Key Visuals**:
- Card visuals for KPIs
- Column chart for segments
- Pie chart for risk breakdown
- Table visual for customer list

**DAX Measures**:
```dax
Churn Rate = DIVIDE(SUM(churn_prediction), COUNT(customer_id)) * 100
Revenue at Risk = CALCULATE(SUM(monthly_price), churn_probability >= 0.7)
```

---

## Export & Sharing

**Static Export**: PDF snapshot for executive reports

**Interactive**: Tableau Public link or Power BI workspace

**Embedding**: iframe in internal portal
