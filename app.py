
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Salary Prediction App (India)",
    page_icon="🇮🇳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #FF9933; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #424242; text-align: center; margin-bottom: 2rem; }
    .prediction-box { background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin: 20px 0; }
    .info-box { background-color: #FFF9C4; padding: 15px; border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🇮🇳 Indian Salary Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict expected salary in INR (₹)</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dataset Info", "📈 Analytics", "🎯 Predict Salary"])

# Load and prepare data (REALISTIC INDIAN CONTEXT)
@st.cache_resource
def load_and_prepare_data():
    np.random.seed(42)
    n_samples = 1000

    ages = np.random.randint(21, 55, n_samples)
    experience = np.random.randint(0, 30, n_samples)

    degrees = ['Bachelor', 'Master', 'PhD', 'Diploma']
    roles = ['Software Engineer', 'Data Scientist', 'Product Manager',
             'Marketing Manager', 'Financial Analyst', 'HR Manager',
             'Designer', 'Sales Executive']

    degree_values = np.random.choice(degrees, n_samples)
    role_values = np.random.choice(roles, n_samples)

    # --- REALISTIC INDIAN SALARY LOGIC (IN INR) ---

    # Base Salary for Freshers (in INR)
    base_salary = 350000

    # Experience Factor: ~1 Lakh per year of experience
    experience_factor = experience * 100000

    # Degree Factor: Masters/PhD get a boost
    degree_factor = np.where(degree_values == 'PhD', 200000,
                    np.where(degree_values == 'Master', 150000,
                    np.where(degree_values == 'Bachelor', 50000, 20000)))

    # Role Factor: Tech roles pay more in India
    role_factor = np.where(role_values == 'Software Engineer', 150000,
                  np.where(role_values == 'Data Scientist', 180000,
                  np.where(role_values == 'Product Manager', 120000,
                  np.where(role_values == 'Financial Analyst', 80000, 50000))))

    # Age Factor: Small increment for maturity
    age_factor = (ages - 21) * 5000

    # Noise: Market variation
    noise = np.random.normal(0, 80000, n_samples)

    # Calculate Total Salary
    salary = base_salary + experience_factor + degree_factor + role_factor + age_factor + noise

    # Ensure minimum salary is realistic (e.g., 2.5 Lakhs)
    salary = np.maximum(salary, 250000)

    df = pd.DataFrame({
        'Age': ages,
        'Degree': degree_values,
        'Role': role_values,
        'Experience': experience,
        'Salary': salary.astype(int)
    })

    return df

# Train model
@st.cache_resource
def train_model(df):
    le_degree = LabelEncoder()
    le_role = LabelEncoder()

    df['Degree_Encoded'] = le_degree.fit_transform(df['Degree'])
    df['Role_Encoded'] = le_role.fit_transform(df['Role'])

    features = ['Age', 'Degree_Encoded', 'Role_Encoded', 'Experience']
    X = df[features]
    y = df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, le_degree, le_role, r2, X_test, y_test, y_pred

# Load data and train
df = load_and_prepare_data()
model, scaler, le_degree, le_role, r2_score_val, X_test, y_test, y_pred = train_model(df)

# HOME PAGE
if page == "🏠 Home":
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="Dataset Size", value=f"{len(df)} records")
    with col2: st.metric(label="Model Accuracy (R²)", value=f"{r2_score_val:.2%}")
    with col3: st.metric(label="Currency", value="INR (₹)")

    st.markdown("---")
    st.header("How It Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📝 Input Features
        1. **Age** - Your current age
        2. **Degree** - Educational qualification
        3. **Role** - Job position
        4. **Experience** - Years of work experience
        """)
    with col2:
        st.markdown("""
        ### 🎯 Output
        Predicts **expected annual salary in ₹** based on Indian market trends.
        """)
    st.info("💡 **Tip**: Navigate to 'Predict Salary' to get your personalized prediction!")

# DATASET INFO PAGE
elif page == "📊 Dataset Info":
    st.markdown("---")
    st.header("Dataset Overview")
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ANALYTICS PAGE
elif page == "📈 Analytics":
    st.markdown("---")
    st.header("Data Analytics & Insights")

    st.subheader("Salary Distribution (INR)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(df['Salary'], bins=30, color='#FF9933', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Annual Salary (₹)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Salaries in India')
    st.pyplot(fig1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Experience vs Salary")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(df['Experience'], df['Salary'], alpha=0.5, color='#4CAF50')
        ax2.set_xlabel('Experience (years)')
        ax2.set_ylabel('Salary (₹)')
        st.pyplot(fig2)
    with col2:
        st.subheader("Average Salary by Role")
        role_salary = df.groupby('Role')['Salary'].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(role_salary.index, role_salary.values, color='#2196F3')
        ax3.set_xlabel('Average Salary (₹)')
        st.pyplot(fig3)

# PREDICTION PAGE
elif page == "🎯 Predict Salary":
    st.markdown("---")
    st.header("Get Your Salary Prediction")
    st.info("Fill in your details below to get a personalized salary prediction in ₹!")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=24, step=1)
            experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1, step=1)
        with col2:
            degree = st.selectbox("Educational Degree", options=['Bachelor', 'Master', 'PhD', 'Diploma'])
            role = st.selectbox("Job Role", options=['Software Engineer', 'Data Scientist', 'Product Manager',
                                       'Marketing Manager', 'Financial Analyst', 'HR Manager',
                                       'Designer', 'Sales Executive'])

        submitted = st.form_submit_button("🔮 Predict Salary", type="primary")

    if submitted:
        degree_encoded = le_degree.transform([degree])[0]
        role_encoded = le_role.transform([role])[0]

        input_data = np.array([[age, degree_encoded, role_encoded, experience]])
        input_scaled = scaler.transform(input_data)
        predicted_salary = model.predict(input_scaled)[0]
        predicted_salary = max(predicted_salary, 250000) # Min 2.5 Lakhs

        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #2E7D32; margin-top: 0;">💰 Predicted Annual Salary</h2>
            <h1 style="color: #1B5E20; margin: 10px 0;">₹ {predicted_salary:,.0f}</h1>
            <p style="font-size: 1.1rem;">
                <strong>Profile:</strong> {age} yrs | {degree} | {role} | {experience} yrs Exp
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insights
        avg_role = df[df['Role'] == role]['Salary'].mean()
        avg_exp = df[df['Experience'] == experience]['Salary'].mean()

        col1, col2 = st.columns(2)
        with col1: st.metric(f"Avg for {role}", f"₹ {avg_role:,.0f}")
        with col2: st.metric(f"Avg for {experience} Yrs Exp", f"₹ {avg_exp:,.0f}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #757575;'>© 2026 Indian Salary Prediction System</div>", unsafe_allow_html=True)

print("✅ app.py updated with Realistic Indian Data!")
