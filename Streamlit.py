import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
class_model = load_pickle("F:\\Guvi Projects\\Clickstream - Customer Conversion\\Pickles\\best_model_class.pkl")
reg_model = load_pickle("F:\\Guvi Projects\\Clickstream - Customer Conversion\\Pickles\\best_model_reg.pkl")
clust_model = load_pickle("F:\\Guvi Projects\\Clickstream - Customer Conversion\\Pickles\\best_model_clust.pkl")

try:
    preprocessor = load_pickle("F:\\Guvi Projects\\Clickstream - Customer Conversion\\Pickles\\preprocessed_data.pkl")
except Exception as e:
    st.warning(f"Could not load preprocessor: {e}")
    preprocessor = None

def debug_model_features(model, df):

    st.write("Model Debugging Information")
    
    if isinstance(model, dict):
        st.write("Model is a dictionary with the following keys:")
        st.write(list(model.keys()))
        
        for key, value in model.items():
            if hasattr(value, 'feature_names_in_'):
                st.write(f"Found model with feature information in key '{key}'")
                st.write(f"This model expects {len(value.feature_names_in_)} features:")
                st.write(value.feature_names_in_)
                
                common_features = [col for col in value.feature_names_in_ if col in df.columns]
                st.write(f"Found {len(common_features)}/{len(value.feature_names_in_)} expected features in data")
                
                if len(common_features) < len(value.feature_names_in_):
                    missing = [f for f in value.feature_names_in_ if f not in df.columns]
                    st.write("Missing features:")
                    st.write(missing)
                return
            elif hasattr(value, 'n_features_in_'):
                st.write(f"Model in key '{key}' expects {value.n_features_in_} features but doesn't provide feature names")
                return

        st.write("No sklearn model with feature information found in the dictionary")
        st.write("Available dataframe features:")
        st.write(df.columns.tolist())
        
    elif hasattr(model, 'feature_names_in_'):
        st.write(f"Model expects {len(model.feature_names_in_)} features:")
        st.write(model.feature_names_in_)
        
        common_features = [col for col in model.feature_names_in_ if col in df.columns]
        st.write(f"Found {len(common_features)}/{len(model.feature_names_in_)} expected features in data")
        
        if len(common_features) < len(model.feature_names_in_):
            missing = [f for f in model.feature_names_in_ if f not in df.columns]
            st.write("Missing features:")
            st.write(missing)
    
    elif hasattr(model, 'n_features_in_'):
        st.write(f"Model expects {model.n_features_in_} features but doesn't provide feature names")
        st.write("Available dataframe features:")
        st.write(df.columns.tolist())
    
    else:
        st.write("Model doesn't provide feature information")
        st.write("Available dataframe features:")
        st.write(df.columns.tolist())

st.sidebar.title("Pages")

page = st.sidebar.radio("Go to", ["Upload Data", "Manual Entry", "Classification", "Regression", "Clustering"])

if "df_features" not in st.session_state:
    st.session_state.df_features = None

if page == "Upload Data":

    st.title("Upload CSV File")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
        
        if st.button('Compute New Features'):
            df_features = data.copy()
            total_clicks = df_features.groupby('session_id')['order'].count()
            avg_price = df_features.groupby('session_id')['price'].mean()
            unique_products = df_features.groupby('session_id')['page2_clothing_model'].nunique()
            browsing_depth = df_features.groupby('session_id')['page'].max()
            df_features['total_clicks'] = df_features['session_id'].map(total_clicks)
            df_features['avg_price'] = df_features['session_id'].map(avg_price)
            df_features['unique_products'] = df_features['session_id'].map(unique_products)
            df_features['browsing_depth'] = df_features['session_id'].map(browsing_depth)
            df_features['date'] = pd.to_datetime(df_features[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
            df_features['weekday'] = df_features['date'].dt.dayofweek
            df_features['weekend'] = (df_features['weekday'] >= 5).astype(int)
            median_price = df_features['price'].median()
            df_features['high_price_preference'] = (df_features['price'] > median_price).astype(int)
            df_features.drop(columns=['date'], inplace = True)

            st.session_state.df_features = df_features

            st.write("Feature Engineered Data")
            st.write(df_features.head())

        if st.button("Save Data"):
            if st.session_state.df_features is not None:
                st.write("Data saved successfully for further analysis.")
            else:
                st.warning("Please compute features before saving.")
        

elif page == "Manual Entry":

    st.title("Enter Data Manually for Analysis")

    country_mappings = {
        1: 'Australia', 2: 'Austria', 3: 'Belgium', 4: 'British Virgin Islands', 5: 'Cayman Islands',
        6: 'Christmas Island', 7: 'Croatia', 8: 'Cyprus', 9: 'Czech Republic', 10: 'Denmark',
        11: 'Estonia', 12: 'unidentified', 13: 'Faroe Islands', 14: 'Finland', 15: 'France',
        16: 'Germany', 17: 'Greece', 18: 'Hungary', 19: 'Iceland', 20: 'India', 21: 'Ireland',
        22: 'Italy', 23: 'Latvia', 24: 'Lithuania', 25: 'Luxembourg', 26: 'Mexico', 27: 'Netherlands',
        28: 'Norway', 29: 'Poland', 30: 'Portugal', 31: 'Romania', 32: 'Russia', 33: 'San Marino',
        34: 'Slovakia', 35: 'Slovenia', 36: 'Spain', 37: 'Sweden', 38: 'Switzerland', 39: 'Ukraine',
        40: 'United Arab Emirates', 41: 'United Kingdom', 42: 'USA', 43: 'biz (.biz)', 44: 'com (.com)',
        45: 'int (.int)', 46: 'net (.net)', 47: 'org (*.org)'
    }

    category_mappings = {
        1: 'Trousers', 2: 'Skirts', 3: 'Blouses', 4: 'Sales'
    }

    color_mappings = {
        1: 'beige',2: 'black',3: 'blue',4: 'brown',5: 'burgundy',6: 'gray',7: 'green',8: 'navy blue',9: 'many colors',
        10: 'olive',11: 'pink',12: 'red',13: 'violet',14: 'white'
    }

    location_mappings = {
    1:'top left',2:'top in the middle',3:'top right',4:'bottom left',5:'bottom in the middle',6:'bottom right'
    }
    model_mappings = {
        1:'Face', 2:'Profile'
    }

    page_mappings = {
        1: "Home", 2: "Category", 3: "Product", 4: "Cart", 5: "Checkout"
    }

    year = st.selectbox("Select Year", [2008])
    month = st.slider("Enter Month", 1, 12, 1)
    day = st.slider("Enter Day", 1, 31, 1)
    session_id = st.number_input("Select Session ID", min_value = 1000, max_value = 99999)
    order = st.slider("Enter Click Sequence", 1, 200, 1)
    country = st.selectbox("Select Country", list(country_mappings.values()))
    category = st.selectbox("Select Category", list(category_mappings.values()))
    model_number = st.text_area("Enter Model Number")
    color = st.selectbox("Select Colour", list(color_mappings.values()))
    location = st.selectbox("Select Photo Location", list(location_mappings.values()))
    model = st.selectbox("Select Model Photo", list(model_mappings.values()))
    price = st.slider("Enter Price ($)", 10, 200, 10)
    page = st.selectbox("Select Page", list(page_mappings.values()))

    country = list(country_mappings.keys())[list(country_mappings.values()).index(country)]
    category = list(category_mappings.keys())[list(category_mappings.values()).index(category)]
    color = list(color_mappings.keys())[list(color_mappings.values()).index(color)]
    location = list(location_mappings.keys())[list(location_mappings.values()).index(location)]
    model = list(model_mappings.keys())[list(model_mappings.values()).index(model)]
    page = list(page_mappings.keys())[list(page_mappings.values()).index(page)]

    user_input = pd.DataFrame([{
        "year": year, "month": month, "day": day, "session_id": session_id, "order": order, "country": country, "page1_main_category": category,
        "page2_clothing_model":model_number , "colour": color, "location": location, "model_photography": model, "price": price, "page": page
    }])

    if st.button("View Entered Data"):
        st.write("Entered Data")
        st.write(user_input)

    if st.button("Compute New Features"):
        df_features = user_input.copy()
        total_clicks = df_features.groupby('session_id')['order'].count()
        avg_price = df_features.groupby('session_id')['price'].mean()
        unique_products = df_features.groupby('session_id')['page2_clothing_model'].nunique()
        browsing_depth = df_features.groupby('session_id')['page'].max()
        df_features['total_clicks'] = df_features['session_id'].map(total_clicks)
        df_features['avg_price'] = df_features['session_id'].map(avg_price)
        df_features['unique_products'] = df_features['session_id'].map(unique_products)
        df_features['browsing_depth'] = df_features['session_id'].map(browsing_depth)
        df_features['date'] = pd.to_datetime(df_features[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
        df_features['weekday'] = df_features['date'].dt.dayofweek
        df_features['weekend'] = (df_features['weekday'] >= 5).astype(int)
        median_price = 100
        df_features['high_price_preference'] = (df_features['price'] > median_price).astype(int)
        df_features.drop(columns=['date'], inplace = True)
        st.session_state.df_features = df_features
        st.write("Feature Engineered Data")
        st.write(df_features)
    
    if st.button("Save Data"):
        if st.session_state.df_features is not None:
            st.write("Data saved successfully for further analysis.")
        else:
            st.warning("Please compute features before saving.")

elif page == "Classification":

    st.title("Customer Purchase Prediction")
    st.write("Predict if a customer will complete a purchase.")

    if st.session_state.df_features is not None:
        st.write("### Feature Engineered Data Used for Classification")
        st.write(st.session_state.df_features.head())
        
        debug_model_features(class_model, st.session_state.df_features)
        
        try:
            if hasattr(class_model, 'feature_names_in_'):
                common_features = [col for col in class_model.feature_names_in_ 
                                  if col in st.session_state.df_features.columns]
                
                if len(common_features) == len(class_model.feature_names_in_):
                    st.write("Using exact features expected by model...")
                    prediction_data = st.session_state.df_features[common_features]
                    predictions = class_model.predict(prediction_data)
                    st.session_state.df_features["Prediction"] = predictions
                    
                    st.write("Classification Results")
                    display_cols = ['page1_main_category', 'colour', 'location', 
                                  'model_photography', 'page', 'high_price_preference', 'Prediction']
                    st.write(st.session_state.df_features[display_cols])

                else:
                    st.warning(f"Only found {len(common_features)} of {len(class_model.feature_names_in_)} required features !!!")
                    st.write("Try using an alternative approach...")
                    raise ValueError("Insufficient features !!!")
            else:
                raise ValueError("Model doesn't provide feature information")
                
        except Exception as e:
            st.write(f"Direct prediction failed: {str(e)}")

            st.write("Trying alternative approach with basic features...")
            
            basic_features = ['total_clicks', 'avg_price', 'unique_products', 
                            'browsing_depth', 'page', 'price', 'high_price_preference', 
                            'weekday', 'weekend']
            
            available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
            
            if len(available_features) > 0:
                st.write(f"Using {len(available_features)} basic features for prediction")
                
                df_for_prediction = st.session_state.df_features[available_features]

                scaler = StandardScaler()
                numeric_features = df_for_prediction.select_dtypes(include=['float64', 'int64']).columns
                df_for_prediction[numeric_features] = scaler.fit_transform(df_for_prediction[numeric_features])

                try:
                    predictions = class_model.predict(df_for_prediction)
                    st.session_state.df_features["Prediction"] = predictions
                    
                    st.write("Classification Results (using basic features)")

                    display_cols = ['page1_main_category', 'colour', 'location', 
                                  'model_photography', 'page', 'high_price_preference', 'Prediction']
                    st.write(st.session_state.df_features[display_cols])

                except Exception as e:
                    st.error(f"Alternative prediction failed: {str(e)}")
                    st.write("The model and current data features are incompatible.")
                    st.write("Consider retraining your model with these features or using a compatible dataset.")
            else:
                st.error("Cannot find compatible features for this model")
    else:
        st.warning("Please upload or manually enter data before running classification.")

elif page == "Regression":

    st.title("Customer Spending Prediction")
    st.write("Predict how much a customer will spend.")

    if st.session_state.df_features is not None:
        st.write("Feature Engineered Data Used for Regression")
        st.write(st.session_state.df_features.head())

        debug_model_features(reg_model, st.session_state.df_features)
        
        try:
            if hasattr(reg_model, 'feature_names_in_'):
                common_features = [col for col in reg_model.feature_names_in_ 
                                  if col in st.session_state.df_features.columns]
                
                if len(common_features) == len(reg_model.feature_names_in_):
                    st.write("Using exact features expected by model")
                    prediction_data = st.session_state.df_features[common_features]
                    predictions = reg_model.predict(prediction_data)
                    st.session_state.df_features["Predicted_Spending"] = predictions
                    
                    st.write("Regression Results")
                    display_cols = ['page1_main_category', 'colour', 'location', 
                                  'model_photography', 'page', 'high_price_preference', 'Predicted_Spending']
                    st.write(st.session_state.df_features[display_cols])

                    st.write("Visualization of Predictions")
                    if 'total_clicks' in st.session_state.df_features.columns:
                        fig = px.scatter(st.session_state.df_features, 
                                        x='total_clicks', 
                                        y='Predicted_Spending',
                                        color='high_price_preference' if 'high_price_preference' in st.session_state.df_features.columns else None,
                                        hover_data=['page1_main_category'] if 'page1_main_category' in st.session_state.df_features.columns else None)
                        st.plotly_chart(fig)
                else:
                    st.warning(f"Only found {len(common_features)} of {len(reg_model.feature_names_in_)} required features")
                    st.write("Try using an alternative approach...")
                    raise ValueError("Insufficient features")
            else:
                raise ValueError("Model doesn't provide feature information")
                
        except Exception as e:
            st.write(f"Direct prediction failed: {str(e)}")
            
            st.write("Trying alternative approach with basic features...")
            
            basic_features = ['total_clicks', 'avg_price', 'unique_products', 
                            'browsing_depth', 'page', 'price', 'high_price_preference', 
                            'weekday', 'weekend']
            
            available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
            
            if len(available_features) > 0:
                st.write(f"Using {len(available_features)} basic features for prediction")
                
                df_for_prediction = st.session_state.df_features[available_features]
                
                scaler = StandardScaler()
                numeric_features = df_for_prediction.select_dtypes(include=['float64', 'int64']).columns
                df_for_prediction[numeric_features] = scaler.fit_transform(df_for_prediction[numeric_features])

                try:
                    predictions = reg_model.predict(df_for_prediction)
                    st.session_state.df_features["Predicted_Spending"] = predictions
                    
                    st.write("Regression Results (using basic features)")
                    display_cols = ['page1_main_category', 'colour', 'location', 
                                  'model_photography', 'page', 'high_price_preference', 'Predicted_Spending']
                    st.write(st.session_state.df_features[display_cols])
                    
                    st.write("Visualization of Predictions")
                    fig = px.histogram(st.session_state.df_features, x='Predicted_Spending', 
                                      nbins=20, title='Distribution of Predicted Spending')
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Alternative prediction failed: {str(e)}")
                    st.write("The model and current data features are incompatible.")
                    st.write("Consider retraining your model with these features or using a compatible dataset.")
            else:
                st.error("Cannot find compatible features for this model")
    else:
        st.warning("Please upload or manually enter data before running regression analysis.")

elif page == "Clustering":

    st.title("Customer Segmentation")
    st.write("Group customers into segments based on their behavior.")

    if st.session_state.df_features is not None:
        st.write("Feature Engineered Data Used for Clustering")
        st.write(st.session_state.df_features.head())
        
        debug_model_features(clust_model, st.session_state.df_features)
        
        try:

            if isinstance(clust_model, dict):

                if 'kmeans' in clust_model and hasattr(clust_model['kmeans'], 'predict'):
                    actual_model = clust_model['kmeans']
                    st.write("Using KMeans model from dictionary")

                else:
                    model_found = False
                    for key, value in clust_model.items():
                        if hasattr(value, 'predict') or hasattr(value, 'fit_predict'):
                            actual_model = value
                            model_found = True
                            st.write(f"Using model from dictionary key: {key}")
                            break
                    
                    if not model_found:
                        st.write("No usable model found in dictionary. Creating a new KMeans model.")
                        from sklearn.cluster import KMeans
                        
                        basic_features = ['total_clicks', 'avg_price', 'unique_products', 
                                        'browsing_depth', 'price', 'high_price_preference']
                        available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
                        
                        if len(available_features) > 0:

                            df_for_clustering = st.session_state.df_features[available_features]
                            
                            scaler = StandardScaler()
                            numeric_features = df_for_clustering.select_dtypes(include=['float64', 'int64']).columns
                            df_for_clustering[numeric_features] = scaler.fit_transform(df_for_clustering[numeric_features])
                            
                            actual_model = KMeans(n_clusters=4, random_state=42)
                            actual_model.fit(df_for_clustering)
                            
                            common_features = available_features
                            clustering_data = df_for_clustering
                            st.write("Using newly created KMeans model with basic features")
                            
                            clusters = actual_model.predict(clustering_data)
                            st.session_state.df_features["Cluster"] = clusters
                            
                            st.write("On-the-fly Clustering Results")
                            display_cols = ['page1_main_category', 'colour', 'location', 
                                          'model_photography', 'page', 'high_price_preference', 'Cluster']
                            st.write(st.session_state.df_features[display_cols])
                            
                            st.write("Cluster Centers (standardized features)")
                            cluster_centers_df = pd.DataFrame(
                                actual_model.cluster_centers_, 
                                columns=available_features
                            )
                            st.write(cluster_centers_df)
                            
                            st.write("Cluster Visualization")
                            cluster_counts = st.session_state.df_features["Cluster"].value_counts().reset_index()
                            cluster_counts.columns = ["Cluster", "Count"]
                            fig = px.pie(cluster_counts, names="Cluster", values="Count", 
                                       title="Customer Segment Distribution")
                            st.plotly_chart(fig)
                            
                            st.write("Cluster Analysis")
                            numeric_cols = st.session_state.df_features.select_dtypes(include=['float64', 'int64']).columns
                            cluster_profiles = st.session_state.df_features.groupby('Cluster')[numeric_cols].mean()
                            st.write(cluster_profiles)
                            
                            continue_regular_flow = False
                        else:
                            st.error("Cannot find any basic features for clustering")
                            continue_regular_flow = False
                    else:
                        continue_regular_flow = True
                        
            else:
                actual_model = clust_model
                continue_regular_flow = True
            
            if hasattr(actual_model, 'feature_names_in_'):
                common_features = [col for col in actual_model.feature_names_in_ 
                                  if col in st.session_state.df_features.columns]
                
                if len(common_features) == len(actual_model.feature_names_in_):
                    st.write("Using exact features expected by model")
                    clustering_data = st.session_state.df_features[common_features]
                    clusters = actual_model.predict(clustering_data)
                    st.session_state.df_features["Cluster"] = clusters
                    
                    st.write("Clustering Results")
                    display_cols = ['page1_main_category', 'colour', 'location', 
                                  'model_photography', 'page', 'high_price_preference', 'Cluster']
                    st.write(st.session_state.df_features[display_cols])
                    
                    st.write("Cluster Analysis")
                    cluster_counts = st.session_state.df_features["Cluster"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    
                    fig1 = px.bar(cluster_counts, x="Cluster", y="Count", 
                                 title="Number of Customers in Each Cluster")
                    st.plotly_chart(fig1)
                    
                    if 'total_clicks' in st.session_state.df_features.columns and 'avg_price' in st.session_state.df_features.columns:
                        fig2 = px.scatter(st.session_state.df_features, 
                                        x='total_clicks', 
                                        y='avg_price',
                                        color='Cluster',
                                        hover_data=['page1_main_category'] if 'page1_main_category' in st.session_state.df_features.columns else None,
                                        title="Cluster Distribution by Total Clicks and Average Price")
                        st.plotly_chart(fig2)
                    
                    st.write("Cluster Characteristics")
                    numeric_cols = st.session_state.df_features.select_dtypes(include=['float64', 'int64']).columns
                    cluster_profiles = st.session_state.df_features.groupby('Cluster')[numeric_cols].mean()
                    st.write(cluster_profiles)
                    
                else:
                    st.warning(f"Only found {len(common_features)} of {len(actual_model.feature_names_in_)} required features")
                    st.write("Try using an alternative approach...")
                    raise ValueError("Insufficient features")
            else:
                raise ValueError("Model doesn't provide feature information")
                
        except Exception as e:
            st.write(f"Direct clustering failed: {str(e)}")
            
            st.write("Trying alternative approach with basic features...")

            basic_features = ['total_clicks', 'avg_price', 'unique_products', 
                            'browsing_depth', 'price', 'high_price_preference']
            
            available_features = [col for col in basic_features if col in st.session_state.df_features.columns]
            
            if len(available_features) > 0:
                st.write(f"Using {len(available_features)} basic features for clustering")
                
                df_for_clustering = st.session_state.df_features[available_features]
                
                scaler = StandardScaler()
                numeric_features = df_for_clustering.select_dtypes(include=['float64', 'int64']).columns
                df_for_clustering[numeric_features] = scaler.fit_transform(df_for_clustering[numeric_features])
                
                try:

                    if isinstance(clust_model, dict):
                        model_found = False
                        if 'kmeans' in clust_model and hasattr(clust_model['kmeans'], 'predict'):
                            clusters = clust_model['kmeans'].predict(df_for_clustering)
                            model_found = True
                            st.write("Using existing KMeans model from dictionary")
                        else:

                            for key, value in clust_model.items():
                                if hasattr(value, 'predict'):
                                    clusters = value.predict(df_for_clustering)
                                    model_found = True
                                    st.write(f"Using model from key '{key}' for prediction")
                                    break

                        if not model_found:
                            st.write("No usable clustering model found. Creating a new KMeans model on the fly.")
                            from sklearn.cluster import KMeans

                            temp_model = KMeans(n_clusters=4, random_state=42)
                            temp_model.fit(df_for_clustering)
                            clusters = temp_model.labels_

                            st.write("On-the-fly KMeans Clustering Results")
                            st.write("Cluster centers (standardized features):")
                            cluster_centers_df = pd.DataFrame(
                                temp_model.cluster_centers_, 
                                columns=df_for_clustering.columns
                            )
                            st.write(cluster_centers_df)
                    else:
                        if hasattr(clust_model, 'predict'):
                            clusters = clust_model.predict(df_for_clustering)
                        else:
                            st.write("Model doesn't have predict method. Creating a new KMeans model.")
                            from sklearn.cluster import KMeans
                            temp_model = KMeans(n_clusters=4, random_state=42)
                            temp_model.fit(df_for_clustering)
                            clusters = temp_model.labels_
                    st.session_state.df_features["Cluster"] = clusters
                    
                    st.write("Clustering Results (using basic features)")
                    display_cols = ['page1_main_category', 'colour', 'location', 
                                  'model_photography', 'page', 'high_price_preference', 'Cluster']
                    st.write(st.session_state.df_features[display_cols])
                    
                    cluster_counts = st.session_state.df_features["Cluster"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    fig = px.pie(cluster_counts, names="Cluster", values="Count", 
                               title="Customer Segment Distribution")
                    st.plotly_chart(fig)
                    
                    st.write("Basic Cluster Characteristics")
                    cluster_means = st.session_state.df_features.groupby('Cluster')[available_features].mean()
                    st.write(cluster_means)
                    
                except Exception as e:
                    st.error(f"Alternative clustering failed: {str(e)}")
                    st.write("The model and current data features are incompatible.")
                    st.write("Consider retraining your model with these features or using a compatible dataset.")
            else:
                st.error("Cannot find compatible features for this model")
    else:
        st.warning("Please upload or manually enter data before running clustering analysis.")

if page in ["Classification", "Regression", "Clustering"]:

    if st.button("Save Predictions"):
        if "df_features" in st.session_state and st.session_state.df_features is not None:
            file_path = "predictions.csv"
            st.session_state.df_features.to_csv(file_path, index=False)
            
            st.download_button(
                label="Download Predictions",
                data=st.session_state.df_features.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )

            st.success("Predictions saved successfully!")
        else:
            st.warning("No predictions available to save.")