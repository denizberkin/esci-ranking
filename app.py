import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image

from inference import test
from utils.save import load_model, load_df
from utils.variables import MODEL_SAVE_FOLDER, ROOT_FOLDER
from pipeline import preprocess_pipeline

import sys
sys.path.append('.')


st.set_page_config(
    page_title="ESCI Product Ranking Model",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ESCI Product Ranking Model")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Model Testing", "Plot Visualization"])


def get_available_models():
    if not os.path.exists(MODEL_SAVE_FOLDER):
        return []
    return [f for f in os.listdir(MODEL_SAVE_FOLDER) if f.endswith('.pkl')]


def get_available_test_files():
    if not os.path.exists(ROOT_FOLDER):
        return []
    return [f for f in os.listdir(ROOT_FOLDER) if f.startswith('test') and f.endswith('.parquet')]


def get_available_plots():
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        return []
    return [f for f in os.listdir(plot_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def run_inference(model_path, test_file, limit):
    try:
        model = load_model(model_path)
        
        test_df = load_df([test_file])
        
        if limit > 0:
            test_df = test_df.head(limit)
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("preprocessing test data...")
        test_df, feature_columns = preprocess_pipeline(test_df, is_test=True)
        progress_bar.progress(50)
        
        # Run inference
        progress_text.text("running inference...")
        results = test(
            model=model,
            test_df=test_df,
            feature_columns=feature_columns,
            plot_feature_importance=False
        )
        progress_bar.progress(100)
        progress_text.text("inference complete!")
        
        return results, test_df
    except Exception as e:
        st.error(f"Error running inference: {str(e)}")
        return None, None
    
    
def display_query_results(results, test_df, num_queries=5, num_products=5):
    if not results or "results_by_query" not in results:
        st.warning("No results to display")
        return
    
    query_ids = list(results["results_by_query"].keys())
    
    if len(query_ids) == 0:
        st.warning("No queries found in results")
        return
    
    if "metrics" in results:
        metrics = results["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("NDCG", f"{metrics['avg_ndcg']:.4f}")
        col2.metric("Kendall Tau", f"{metrics['avg_kendall_tau']:.4f}")
        col3.metric("Weighted Tau", f"{metrics['avg_weighted_tau']:.4f}")
    
    sample_size = min(num_queries, len(query_ids))
    sample_query_ids = query_ids[:sample_size]
    
    for qid in sample_query_ids:
        query_data = results["results_by_query"][qid]
        query_row = test_df[test_df["query_id"] == qid].iloc[0]
        query_text = query_row["query"]
        
        st.subheader(f"Query: '{query_text}' (ID: {qid})")
        
        products = []
        for i, (pred, ex_id) in enumerate(zip(query_data["predictions"], query_data["example_ids"])):
            try:
                product_row = test_df[test_df["example_id"] == ex_id].iloc[0]
                
                product = {
                    "Rank": i + 1,
                    "Score": float(pred),
                    "Title": product_row["product_title"],
                    "Example ID": ex_id
                }
                
                if "product_brand" in product_row:
                    product["Brand"] = product_row["product_brand"]
                
                if "true_labels" in query_data:
                    product["True Label"] = query_data["true_labels"][i]
                    
                products.append(product)
            except Exception as e:
                st.warning(f"Error processing product: {str(e)}")
                continue
        
        products = sorted(products, key=lambda x: x["Score"], reverse=True)
        
        # Display top N products
        display_products = products[:num_products]
        product_df = pd.DataFrame(display_products)
        st.dataframe(product_df)
        st.markdown("---")
        
        
if page == "Model Testing":
    st.header("Model Testing")
    
    # Get available models and test files
    available_models = get_available_models()
    available_test_files = get_available_test_files()
    
    if not available_models:
        st.warning("No models found in the models directory.")
    elif not available_test_files:
        st.warning("No test files found in the data directory.")
    else:
        # Model selection
        selected_model = st.selectbox("Select a model", available_models)
        model_path = os.path.join(MODEL_SAVE_FOLDER, selected_model)
        
        # Test file selection
        selected_test_file = st.selectbox("Select a test file", available_test_files)
        
        # Data limit option
        data_limit = st.number_input("Limit number of test samples (0 for all)", value=100, min_value=0, step=100)
        
        # Run inference button
        if st.button("Run Inference"):
            with st.spinner("Running inference..."):
                results, test_df = run_inference(model_path, selected_test_file, data_limit)
                
            if results is not None:
                st.success("Inference completed successfully!")
                # Display query results
                display_query_results(results, test_df)
                
                # Add a button to download results
                if "results_by_query" in results:
                    results_file = os.path.join(MODEL_SAVE_FOLDER, "predictions", "test_predictions.csv")
                    if os.path.exists(results_file):
                        with open(results_file, "rb") as file:
                            btn = st.download_button(
                                label="Download Predictions CSV",
                                data=file,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )

# Plot Visualization Page
if page == "Plot Visualization":
    st.header("Plot Visualization")
    
    tab1, tab2 = st.tabs(["Pre-generated Plots", "Create New Plots"])
    
    with tab1:
        # Get available plots
        available_plots = get_available_plots()
        
        if not available_plots:
            st.warning("No plots found in the plots directory.")
        else:
            # Plot selection
            selected_plot = st.selectbox("Select a plot to view", available_plots)
            
            # Display the selected plot
            try:
                plot_path = os.path.join("plots", selected_plot)
                image = Image.open(plot_path)
                st.image(image, caption=selected_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading plot: {str(e)}")
    
    with tab2:
        st.subheader("Generate New Plots")
        
        plot_type = st.selectbox(
            "Select a plot type", 
            ["Feature Importance", "Feature Correlation", "Feature Distribution"]
        )
        
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            st.warning("No models found in the models directory.")
        else:
            # Model selection for feature importance
            if plot_type == "Feature Importance":
                selected_model = st.selectbox("Select a model for feature importance", available_models)
                if st.button("Generate Feature Importance Plot"):
                    with st.spinner("Generating plot..."):
                        try:
                            model_path = os.path.join(MODEL_SAVE_FOLDER, selected_model)
                            model = load_model(model_path)
                            
                            # Get feature importance
                            feature_importance = model.feature_importance()
                            feature_names = model.feature_name()
                            
                            # Create DataFrame for plotting
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': feature_importance
                            })
                            
                            # Sort by importance
                            importance_df = importance_df.sort_values('Importance', ascending=False)
                            
                            # Limit to top 20 features for readability
                            top_features = importance_df.head(20)
                            
                            # Create plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                            ax.set_title('Feature Importance')
                            ax.set_xlabel('Importance')
                            ax.set_ylabel('Features')
                            
                            # Display plot
                            st.pyplot(fig)
                            
                            # Save plot option
                            save_path = os.path.join("plots", f"feature_importance_{time.strftime('%Y%m%d_%H%M%S')}.png")
                            plt.savefig(save_path, bbox_inches='tight', dpi=300)
                            st.success(f"Plot saved to {save_path}")
                        except Exception as e:
                            st.error(f"Error generating plot: {str(e)}")
            
            # Feature correlation requires data
            elif plot_type == "Feature Correlation":
                # Get available test files
                available_test_files = get_available_test_files()
                
                if not available_test_files:
                    st.warning("No test files found in the data directory.")
                else:
                    # Test file selection
                    selected_test_file = st.selectbox("Select a data file", available_test_files)
                    
                    # Number of samples
                    num_samples = st.slider("Number of samples to use", 100, 10000, 1000, 100)
                    
                    if st.button("Generate Correlation Plot"):
                        with st.spinner("Generating plot..."):
                            try:
                                # Load data
                                df = load_df([selected_test_file])
                                df = df.head(num_samples)
                                
                                # Process data to get features
                                df, feature_columns = preprocess_pipeline(df, is_test=True)
                                
                                # Calculate correlation matrix
                                if "labels" in df.columns:
                                    feature_columns.append("labels")
                                
                                corr_df = df[feature_columns].corr()
                                
                                # Create plot
                                fig, ax = plt.subplots(figsize=(12, 10))
                                mask = np.triu(np.ones_like(corr_df, dtype=bool))
                                sns.heatmap(corr_df, mask=mask, cmap="RdBu_r", vmax=1, vmin=-1, center=0,
                                            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
                                ax.set_title('Feature Correlation Matrix')
                                
                                # Display plot
                                st.pyplot(fig)
                                
                                # Save plot option
                                save_path = os.path.join("plots", f"correlation_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png")
                                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                                st.success(f"Plot saved to {save_path}")
                            except Exception as e:
                                st.error(f"Error generating plot: {str(e)}")
            
            # Feature distribution requires data
            elif plot_type == "Feature Distribution":
                # Get available test files
                available_test_files = get_available_test_files()
                
                if not available_test_files:
                    st.warning("No test files found in the data directory.")
                else:
                    # Test file selection
                    selected_test_file = st.selectbox("Select a data file", available_test_files)
                    
                    # Number of samples
                    num_samples = st.slider("Number of samples to use", 100, 10000, 1000, 100)
                    
                    if st.button("Load Features"):
                        with st.spinner("Loading features..."):
                            try:
                                # Load data
                                df = load_df([selected_test_file])
                                df = df.head(num_samples)
                                
                                # Process data to get features
                                df, feature_columns = preprocess_pipeline(df, is_test=True)
                                
                                # Let user select a feature to visualize
                                st.session_state.feature_df = df
                                st.session_state.feature_columns = feature_columns
                                st.success(f"Loaded {len(feature_columns)} features from {num_samples} samples")
                            except Exception as e:
                                st.error(f"Error loading features: {str(e)}")
                    
                    if 'feature_df' in st.session_state and 'feature_columns' in st.session_state:
                        selected_feature = st.selectbox("Select a feature to visualize", st.session_state.feature_columns)
                        
                        if st.button("Generate Distribution Plot"):
                            with st.spinner("Generating plot..."):
                                try:
                                    # Create plot
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.histplot(st.session_state.feature_df[selected_feature].dropna(), kde=True, ax=ax)
                                    ax.set_title(f'Distribution of {selected_feature}')
                                    ax.set_xlabel(selected_feature)
                                    ax.set_ylabel('Frequency')
                                    
                                    # Display plot
                                    st.pyplot(fig)
                                    
                                    # Show statistics
                                    stats_df = pd.DataFrame({
                                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count'],
                                        'Value': [
                                            st.session_state.feature_df[selected_feature].mean(),
                                            st.session_state.feature_df[selected_feature].median(),
                                            st.session_state.feature_df[selected_feature].std(),
                                            st.session_state.feature_df[selected_feature].min(),
                                            st.session_state.feature_df[selected_feature].max(),
                                            st.session_state.feature_df[selected_feature].count()
                                        ]
                                    })
                                    st.dataframe(stats_df)
                                    
                                    # Save plot option
                                    save_path = os.path.join("plots", f"{selected_feature}_distribution_{time.strftime('%Y%m%d_%H%M%S')}.png")
                                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                                    st.success(f"Plot saved to {save_path}")
                                except Exception as e:
                                    st.error(f"Error generating plot: {str(e)}")

st.markdown("---")