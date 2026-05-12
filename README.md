# Misbehavior Detection in Vehicular Ad-Hoc Networks (VANETs)

## Overview
This project implements a misbehavior detection system for Vehicular Ad-Hoc Networks (VANETs), focusing on identifying malicious or incorrect messages in vehicle-to-vehicle (V2V) communication. The system evaluates whether transmitted data is physically plausible and consistent, addressing the limitation that authentication alone does not guarantee message truthfulness.

The project compares two approaches:
- A **Random Forest (AI-based) baseline**
- A **lightweight rule-based detector**

## Objectives
- Detect VANET misbehavior such as:
  - Position spoofing
  - False event reporting
  - DoS/DDoS-like activity
- Evaluate detection performance using:
  - Precision, Recall, F1-score
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
- Analyze tradeoffs between detection accuracy and reliability

## Dataset
This project uses a **VeReMi-derived dataset**, a benchmark dataset for evaluating misbehavior detection in VANETs.

⚠️ Note:  
The dataset is not included in this repository due to GitHub size limitations.  
You can obtain it from:
https://veremi-dataset.github.io/

## Project Structure

- vanetmisbehavior/
- |
-  |---src/
- | |-----load_data.py
- | |-----prepare_data.py
- | |-----features.py
- | |-----train_baseline.py
- | |-----rule_baseline.py
- | |-----tune_rule_baseline.py
- | |-----improved_rule.py
- | |-----demo_view.py
- |
-  |---results/
- | |-----final_comparison.txt
- | |-----final_rule_baseline_metrics.txt
- | |-----improved_rule_baseline_summary.csv
- |
-  |---.gitignore
-  |____README.md


## Methodology

### Feature Extraction
The system extracts features representing physical and contextual behavior:
- Speed magnitude
- Acceleration magnitude
- Position magnitude
- Heading
- Noise differences (position, speed, acceleration, heading)
- Combined noise feature

### Detection Approaches

#### 1. Random Forest (AI Baseline)
- Learns patterns from labeled data
- Achieves higher detection recall
- Produces higher false positives

#### 2. Rule-Based Detector
- Uses statistical thresholds derived from benign data
- Computes a rule score based on violated constraints
- Applies a fusion threshold for classification

## Results

| Model                | Accuracy | Recall | FPR   | Notes |
|---------------------|----------|--------|-------|------|
| Random Forest       | ~0.52    | ~0.52  | 0.4659| High detection, many false positives |
| Rule-Based (0.88)   | ~0.50    | ~0.40  | 0.3978| Balanced tradeoff |

### Key Insight
There is a tradeoff between:
- **Detection performance (recall)**
- **Reliability (false positive rate)**

Rule-based detection reduces false alarms but misses subtle attacks, while AI detects more attacks but introduces more false positives.

## Demo

Run the following:

```bash
python3 src/train_baseline.py
python3 src/rule_baseline.py
python3 src/demo_view.py