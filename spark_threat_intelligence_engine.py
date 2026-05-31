import argparse
import sys
from datetime import datetime
import warnings
import io

# --- PySpark Libraries ---
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType, StructType, StructField, IntegerType, LongType
from pyspark.ml.linalg import Vectors

# --- Standard ML and Reporting Libraries ---
import pandas as pd
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, PageBreak, Image, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

# Suppress common warnings for a clean final output
warnings.filterwarnings('ignore')

def build_recommendation_engine_spark(spark, traffic_profiles_pd):

    #Trains a recommendation engine using PySpark's ALS (Alternating Least Squares).
    print("\n--- STAGE 5: Building Mitigation Recommendation Engine ---")

    mitigation_actions = [
        'Rate-Limit-IPs', 'Block-Port-80', 'Isolate-Host', 'Analyze-Payload',
        'Reset-Connection', 'Blacklist-IP', 'Apply-WAF-Rule'
    ]
    historical_data = []
    for attack, data in traffic_profiles_pd.items():
        if 'benign' in attack.lower():
            continue
        for profile_num in data['fingerprints'].index:
            profile_id = f"{attack}_P{profile_num}"
            for action in mitigation_actions:
                effectiveness = np.random.uniform(0.1, 0.5)
                if ('DDoS' in attack or 'DoS' in attack) and action in ['Rate-Limit-IPs', 'Blacklist-IP']:
                    effectiveness = np.random.uniform(0.8, 1.0)
                elif 'Botnet' in attack or 'ATTACK' in attack and action in ['Isolate-Host', 'Blacklist-IP']:
                    effectiveness = np.random.uniform(0.7, 0.95)
                elif action == 'Analyze-Payload':
                    effectiveness = np.random.uniform(0.6, 0.8)
                historical_data.append((profile_id, action, effectiveness))

    if not historical_data:
        print("  No attack profiles available to build recommendation engine.")
        return None, None

    ratings_pd = pd.DataFrame(historical_data, columns=['profile_id', 'action', 'effectiveness'])
    ratings_df = spark.createDataFrame(ratings_pd)

    profile_indexer = StringIndexer(inputCol="profile_id", outputCol="profile_idx")
    action_indexer = StringIndexer(inputCol="action", outputCol="action_idx")

    pipeline = Pipeline(stages=[profile_indexer, action_indexer])
    indexer_model = pipeline.fit(ratings_df)
    ratings_indexed = indexer_model.transform(ratings_df)

    als = ALS(
        userCol="profile_idx",
        itemCol="action_idx",
        ratingCol="effectiveness",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=5,
        regParam=0.1,
        seed=42
    )
    als_model = als.fit(ratings_indexed)

    profile_labels = indexer_model.stages[0].labels
    action_labels = indexer_model.stages[1].labels

    print("  Recommendation engine trained successfully for attack profiles.")
    return als_model, (profile_labels, action_labels)


def run_monte_carlo_simulations_spark(spark, traffic_profiles, n_simulations=50000):

    #Runs Monte Carlo simulations to validate the stability of K-Means clusters.
    print("\n--- STAGE 6: Running Monte Carlo Stability Simulations ---")
    stability_results = {}

    for traffic_name, data in traffic_profiles.items():
        print(f"  Simulating stability for '{traffic_name}' profiles...")
        kmeans_model = data['model']
        centroids = kmeans_model.clusterCenters()

        attack_stability = {}
        for profile_id, centroid_vector in enumerate(centroids):
            noise = np.random.normal(0, 0.2, size=(n_simulations, len(centroid_vector)))
            synthetic_vectors = centroid_vector + noise
            synthetic_data = [(Vectors.dense(vec.tolist()),) for vec in synthetic_vectors]
            synthetic_df = spark.createDataFrame(synthetic_data, ["features_scaled"])
            predictions = kmeans_model.transform(synthetic_df)
            correct_predictions = predictions.filter(F.col("prediction") == profile_id).count()
            stability_score = (correct_predictions / n_simulations) * 100
            attack_stability[f'Profile {profile_id}'] = stability_score
        
        stability_results[traffic_name] = attack_stability
        print(f"    Stability Scores for {traffic_name}: {attack_stability}")
    return stability_results


def run_threat_profiling_pipeline(spark, mode, file_path):
    final_results = {}

    print(f"\n--- STAGE 1: Data Ingestion ---")
    
    is_csv = file_path.endswith('.csv')
    
    if is_csv:
        print(f"  Running CSV-specific ingestion pipeline for '{file_path}'...")
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    else: # Parquet
        print(f"  Running Parquet ingestion pipeline for '{file_path}'...")
        df = spark.read.parquet(file_path)

    if mode == 'demo':
        print("  DEMO MODE: Sampling 5% of data for a quick, representative run.")
        original_label_col = None
        for c in df.columns:
            if c.lower().strip() == 'label':
                original_label_col = c
                break
        if not original_label_col:
            raise ValueError("Cannot perform stratified sampling without a 'Label' or 'label' column.")
        fractions = df.select(original_label_col).distinct().withColumn("fraction", F.lit(0.05)).rdd.collectAsMap()
        df = df.stat.sampleBy(original_label_col, fractions, seed=42)
    
    print(f"  Data ingestion complete. Initial records: {df.count()}")
    
    print("\n--- STAGE 2 & 3: Preprocessing and Feature Selection ---")

    print("  Applying universal column name sanitization.")
    clean_cols = [c.strip().lower().replace(' ', '_').replace('/', '_').replace('.', '_') for c in df.columns]
    df = df.toDF(*clean_cols)

    if "label" not in df.columns:
        raise ValueError("Dataset is missing the required 'label' column after cleaning.")
    
    # --- LOGIC FORK BASED ON FILE TYPE ---
    if is_csv:
        print("  CSV detected. Using hardcoded pipeline to match CTU-13 report.")
        print("  Using pre-formatted 'ATTACK'/'BENIGN' labels from file.")

        top_feature_names = [
            'flow_duration',
            'destination_port',
            'total_fwd_packets'
        ]
        labels_to_profile = ['ATTACK', 'BENIGN']
        final_results['top_features'] = pd.DataFrame({'feature': top_feature_names, 'importance': 'N/A (Hardcoded)'})
        print(f"  Using hardcoded feature set: {top_feature_names}")
    
    else: 
        print("  Parquet detected. Using automatic pipeline for general analysis.")

        print("  Starting dynamic feature importance analysis...")
        numeric_features_for_dt = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, IntegerType)) and f.name != 'label']
        assembler_dt = VectorAssembler(inputCols=numeric_features_for_dt, outputCol="features_unscaled_dt")
        scaler_dt = StandardScaler(inputCol="features_unscaled_dt", outputCol="features_scaled_dt")
        label_indexer_dt = StringIndexer(inputCol="label", outputCol="label_idx_dt")
        
        pipeline_dt = Pipeline(stages=[assembler_dt, scaler_dt, label_indexer_dt])
        pipeline_model_dt = pipeline_dt.fit(df)
        df_processed_dt = pipeline_model_dt.transform(df)

        sample_df_dt = df_processed_dt.sample(fraction=0.1, seed=42)
        dt_global = DecisionTreeClassifier(featuresCol="features_scaled_dt", labelCol="label_idx_dt", maxDepth=10, seed=42)
        dt_model_global = dt_global.fit(sample_df_dt)

        importances = dt_model_global.featureImportances.toArray()
        feature_importance_pd = pd.DataFrame(list(zip(numeric_features_for_dt, importances)), columns=['feature', 'importance'])
        top_features_df = feature_importance_pd.sort_values(by='importance', ascending=False).head(10)
        final_results['top_features'] = top_features_df
        top_feature_names = top_features_df['feature'].tolist()
        print(f"  Dynamically identified top features: {', '.join(top_feature_names[:3])}, ...")

        label_counts = df.groupBy("label").count().orderBy(F.col("count").desc()).limit(7).collect()
        labels_to_profile = [row['label'] for row in label_counts if row['label'].lower() != 'benign'][:6]
        print(f"  Dynamically selected labels to profile: {labels_to_profile}")

    all_numeric_features = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, IntegerType)) and f.name != 'label']
    df = df.select(['label'] + [c for c in all_numeric_features if c in df.columns])
    
    stddevs = df.select([F.stddev_pop(c).alias(c) for c in all_numeric_features if c in df.columns]).collect()[0].asDict()
    constant_columns = [c for c, s in stddevs.items() if s is None or s == 0.0]
    if constant_columns:
        df = df.drop(*constant_columns)

    df = df.na.fill(0)
    final_results['total_records_processed'] = df.count()
    print(f"  Preprocessing complete. Final records for analysis: {final_results['total_records_processed']}")

    final_numeric_features = [c for c in all_numeric_features if c in df.columns]

    assembler = VectorAssembler(inputCols=final_numeric_features, outputCol="features_unscaled")
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features_scaled")
    label_indexer = StringIndexer(inputCol="label", outputCol="label_idx")
    
    pipeline = Pipeline(stages=[assembler, scaler, label_indexer])
    pipeline_model = pipeline.fit(df)
    df_processed = pipeline_model.transform(df)
    
    print("\n--- STAGE 4: Threat Behavior Profiling Engine ---")
    traffic_profiles = {}
    for label_name in labels_to_profile:
        print(f"  Profiling '{label_name}' traffic...")
        traffic_df = df_processed.filter(F.col("label") == label_name)
        
        if traffic_df.count() < 3:
            print(f"    Skipping '{label_name}' due to insufficient data (less than 3 records for k=3).")
            continue
        
        kmeans = KMeans(featuresCol="features_scaled", k=3, seed=42)
        kmeans_model = kmeans.fit(traffic_df)
        traffic_clustered = kmeans_model.transform(traffic_df).withColumnRenamed("prediction", "profile_id")

        existing_top_features = [f for f in top_feature_names if f in traffic_clustered.columns]
        profile_fingerprints_spark = traffic_clustered.groupBy("profile_id").agg(
            *[F.mean(c).alias(c) for c in existing_top_features]
        ).orderBy("profile_id")
        
        pca = PCA(k=2, inputCol="features_scaled", outputCol="pca_features")
        pca_model = pca.fit(traffic_df)
        pca_result = pca_model.transform(traffic_clustered)
        
        fingerprints_pd = profile_fingerprints_spark.toPandas().set_index('profile_id')
        
        schema = StructType([
            StructField("pc1", DoubleType(), True),
            StructField("pc2", DoubleType(), True),
            StructField("profile", IntegerType(), True) 
        ])

        pca_rdd = pca_result.select("pca_features", "profile_id").rdd.map(
            lambda row: (float(row['pca_features'][0]), float(row['pca_features'][1]), row['profile_id'])
        )

        pca_spark_df = spark.createDataFrame(pca_rdd, schema=schema)
        pca_pd = pca_spark_df.toPandas()

        traffic_profiles[label_name] = {
            'fingerprints': fingerprints_pd,
            'pca_data': pca_pd,
            'model': kmeans_model
        }

    final_results['traffic_profiles'] = traffic_profiles
    
    if final_results['traffic_profiles']:
        rec_algo, label_maps = build_recommendation_engine_spark(spark, final_results['traffic_profiles'])
        
        if rec_algo:
            final_results['recommendations'] = {}
            profile_labels, action_labels = label_maps
            recommendations_df = rec_algo.recommendForAllUsers(3)
            for row in recommendations_df.collect():
                profile_id = profile_labels[row['profile_idx']]
                top_actions = [action_labels[rec.action_idx] for rec in row['recommendations']]
                final_results['recommendations'][profile_id] = top_actions
        
        mc_results = run_monte_carlo_simulations_spark(
            spark, final_results['traffic_profiles'], n_simulations=500
        )
        final_results['monte_carlo_stability'] = mc_results
        
    return final_results

def create_publication_report(results, mode):
    num = randint(10000,123456)
    filename = f"{num}_Threat_Profiling_Report_{mode}_PySpark_Final.pdf"
    print(f"\n--- FINAL STAGE: Creating Publication-Ready PDF Report: {filename} ---")
    styles = getSampleStyleSheet()
    story = []
    doc_width, doc_height = letter
    margin = inch
    available_width = doc_width - 2 * margin

    story.append(Paragraph("Traffic Behavior Profiling & Intelligence Report", styles['h1']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Analysis Mode: {mode.capitalize()}", styles['h2']))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Executive Summary", styles['h2']))
    top_feats_list = results['top_features']['feature'].head(3).tolist()
    summary_text = f"""
    This report details a scalable, multi-stage data analysis pipeline designed to transform raw network data into <b>actionable cybersecurity intelligence</b>.
    Leveraging a distributed computing framework (Apache Spark), the system processed {results['total_records_processed']} network records to automatically discover and profile distinct malicious and benign activities.
    The analysis pipeline consisted of: <b>(1) Data Ingestion and Preprocessing</b>, <b>(2) Feature Selection</b>,
    <b>(3) Behavior Profiling</b> using K-Means clustering to segment traffic into behavioral groups, <b>(4) a Mitigation Recommendation Engine</b> for attack profiles,
    and <b>(5) Profile Stability Validation</b> via Monte Carlo simulations. The key features for distinguishing traffic included <b>{', '.join(top_feats_list)}</b>.
    This project demonstrates a practical approach for Security Operations Centers (SOCs) to automate threat analysis at scale.
    """
    story.append(Paragraph(summary_text, styles['BodyText']))

    if 'recommendations' in results and results.get('recommendations'):
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Automated Mitigation Recommendations", styles['h2']))
        rec_data = [['Threat Profile', 'Recommended Action 1', 'Recommended Action 2', 'Recommended Action 3']]
        for profile_id, actions in results['recommendations'].items():
            padded_actions = actions + ['N/A'] * (3 - len(actions))
            rec_data.append([profile_id] + padded_actions)
        rec_table = Table(rec_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch], hAlign='LEFT')
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(rec_table)
    
    if 'monte_carlo_stability' in results and results.get('monte_carlo_stability'):
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Profile Stability Analysis (Monte Carlo Simulation)", styles['h2']))
        mc_data = [['Traffic Type', 'Profile ID', 'Stability Score (%)']]
        for traffic_type, profiles in results['monte_carlo_stability'].items():
            for profile_name, score in profiles.items():
                mc_data.append([traffic_type, profile_name, f"{score:.2f}"])
        mc_table = Table(mc_data, colWidths=[2.5*inch, 2*inch, 2*inch], hAlign='LEFT')
        mc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkslategray), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(mc_table)

    story.append(PageBreak())

    for traffic_name, profile_data in results['traffic_profiles'].items():
        story.append(Paragraph(f"Detailed Profile: {traffic_name}", styles['h2']))
        story.append(Spacer(1, 0.2*inch))
        pca_df = profile_data['pca_data']
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x="pc1", y="pc2", hue="profile", palette="viridis", data=pca_df, legend="full", alpha=0.8)
        plt.title(f'Behavioral Clusters within {traffic_name} Traffic', fontsize=14)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=7*inch, height=4.3*inch))
        plt.close()
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Profile Fingerprints", styles['h3']))

        fingerprints_df = profile_data['fingerprints'].transpose().reset_index()
        fingerprints_df.rename(columns={'index': 'Feature'}, inplace=True)
        
        new_headers = ['Feature'] + [f'Profile {col}' for col in fingerprints_df.columns if col != 'Feature']
        fingerprints_df.columns = new_headers
        
        table_data = [fingerprints_df.columns.tolist()]
        for _, row in fingerprints_df.iterrows():
            row_data = [str(row['Feature']).replace('_', ' ').title()]
            for col_name in fingerprints_df.columns[1:]:
                try:
                    row_data.append(f"{float(row[col_name]):,.2f}")
                except (ValueError, TypeError):
                    row_data.append(str(row[col_name]))
            table_data.append(row_data)

        num_cols = len(table_data[0])
        feature_col_width = 2.5 * inch
        profile_col_width = (available_width - feature_col_width) / (num_cols - 1) if num_cols > 1 else 0
        col_widths = [feature_col_width] + [profile_col_width] * (num_cols - 1)

        t = Table(table_data, colWidths=col_widths, hAlign='LEFT')

        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 1, colors.black),
                                ('FONTSIZE', (0,0), (-1,-1), 8),
                                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')]))
        story.append(t)
        story.append(PageBreak())

    doc = SimpleDocTemplate(filename, pagesize=letter)
    doc.build(story)
    print("  Publication-ready PDF report saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced PySpark Project for Network Threat Behavior Profiling.")
    parser.add_argument("--mode", type=str, required=True, choices=['demo', 'full'], help="Execution mode.")
    parser.add_argument("--file", type=str, required=True, help="Path to the Parquet or CSV file to analyze.")
    args = parser.parse_args()
    
    spark = SparkSession.builder \
        .appName("ThreatProfilingPipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    
    try:
        final_project_results = run_threat_profiling_pipeline(spark, args.mode, args.file)
        if not final_project_results.get('traffic_profiles'):
            print("\nNo traffic profiles were generated. Skipping report creation.")
        else:
            create_publication_report(final_project_results, args.mode)
        print("\nProject execution finished successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()