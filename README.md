# Proactive Threat Intelligence & Behavior Profiling Engine

## Overview

This repository contains a scalable, multi-stage data analytics pipeline designed to transform massive volumes of network traffic into actionable cybersecurity intelligence. Developed as part of a research project on proactive vulnerability discovery, the engine automates the identification and profiling of malicious behavior patterns using distributed computing.

The system addresses the challenge of analyzing terabyte-scale internet scan data—a significant hurdle for traditional Security Operations Centers (SOCs)—by leveraging **Apache Spark** to profile threats at scale.

## Core Methodology

The pipeline implements an end-to-end workflow designed for production-grade network data analysis:

* **Scalable Ingestion**: Uses HDFS and Spark to process multi-million record datasets (validated on the CIC-Collection with 9.1M+ records).

* **Dynamic Feature Engineering**: Employs decision-tree-based importance analysis to isolate the most predictive features, such as TCP window sizes and packet length statistics, tailored to specific traffic compositions.

* **Behavioral Profiling**: Implements unsupervised **K-Means clustering** ($k=3$) to segment traffic into distinct behavioral archetypes, generating "fingerprints" for diverse attack variants.

* **Mitigation Recommendation**: Utilizes **Alternating Least Squares (ALS)** collaborative filtering to map discovered threat profiles to verified mitigation strategies, automating the security response workflow.

* **Robustness Validation**: Integrates **Monte Carlo simulations** to stress-test cluster stability against data perturbations, ensuring identified signatures are reliable and reproducible.


## Technical Architecture

* **Distributed Computing**: Apache Spark (PySpark), Hadoop Distributed File System (HDFS).

* **Machine Learning (MLlib)**: K-Means (Clustering), ALS (Recommendation), Decision Trees (Feature Importance), StandardScaler, VectorAssembler.

* **Automation & Reporting**: Automated generation of publication-ready PDF intelligence reports using ReportLab.


## Deployment & Usage

### Execution Modes

The engine provides two execution paths:

1. **Demo Mode**: Executes a stratified sample for rapid pipeline validation.
2. **Full Mode**: Processes the entire dataset for comprehensive production analysis.

```bash
# Example command for full-scale production analysis
python threat_profiling_pipeline.py --mode full --file network_traffic.parquet

```

## Strategic Impact

This pipeline solves critical "last-mile" challenges in modern cybersecurity operations:

* **Granular Intelligence**: Demonstrates that profiling on specific attack labels yields significantly higher stability and consistency than broad, generic labels.

* **Automated Decision Support**: Provides data-driven recommendations, enabling SOC analysts to apply consistent countermeasures to similar attack behaviors.

* **Validated Robustness**: Ensures that security policies are based on stable behavioral profiles that hold up under noise, as validated by Monte Carlo stress testing.

## Research Context

This tool was developed as part of a formal study into analyzing threat profiles in global network traffic. For detailed methodology and performance analysis across datasets like CTU-13 and CIC-Collection, please refer to the project documentation.
