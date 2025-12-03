import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. LOAD THE SAVED FILES ---
@st.cache_resource
def load_data():
    """Load model and encoders with enhanced error handling"""
    try:
        with open('salary_model.pickle', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        try:
            with open('salary_model.pickle', 'rb') as file:
                model = pickle.load(file)
        except FileNotFoundError:
            st.error("❌ Model file not found. Please upload 'salary_model.pickle' or 'salary_model.pkl'.")
            return None, None

    try:
        with open('encoders.pickle', 'rb') as file:
            encoders = pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Encoders file not found. Please run the preprocessing step to generate 'encoders.pkl'.")
        return None, None

    return model, encoders

model, encoders = load_data()

if model and encoders:
    le_job = encoders["le_job"]
    le_exp = encoders["le_exp"]
    le_ind = encoders["le_ind"]
    le_city = encoders["le_city"]

    # --- 2. PAGE CONFIGURATION & ENHANCED STYLING ---
    st.set_page_config(
        page_title="Sri Lanka Salary Predictor 2025",
        page_icon="🇱🇰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced CSS styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 18px;
            font-weight: 600;
            border-radius: 12px;
            padding: 12px 28px;
            width: 100%;
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .salary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        
        .salary-amount {
            font-size: 48px;
            font-weight: 700;
            color: white;
            margin: 10px 0;
        }
        
        .salary-label {
            font-size: 20px;
            color: rgba(255,255,255,0.9);
            font-weight: 500;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 18px;
            margin-bottom: 30px;
        }
        
        .info-box {
            background: rgb(0,0,0);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
            margin: 15px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR SETTINGS ---
    st.sidebar.image("https://flagcdn.com/w320/lk.png", width=100)
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown("### 💱 Currency Settings")
    currency_rate = st.sidebar.number_input(
        "Exchange Rate (1 USD = ? LKR)", 
        value=300.0, 
        step=10.0,
        help="Current exchange rate for USD to LKR"
    )
    
    is_annual = st.sidebar.checkbox(
        "Model predicts annual salary", 
        value=True,
        help="Check if model output is annual, will convert to monthly"
    )
    
    st.sidebar.markdown("### 📊 Display Options")
    show_comparison = st.sidebar.checkbox("Show city comparison", value=True)
    num_cities = st.sidebar.slider("Cities to compare", 3, 8, 5)
    show_insights = st.sidebar.checkbox("Show market insights", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        "This AI-powered tool predicts salaries based on job title, "
        "experience, industry, and location in Sri Lanka. "
        "Data is constantly updated to reflect market trends."
    )

    # --- MAIN PAGE ---
    st.markdown("<h1> Sri Lanka Salary Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Get accurate salary estimates based on real market data</p>", unsafe_allow_html=True)

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["💼 Salary Prediction", "📈 Market Insights", "ℹ️ Guide"])

    with tab1:
        # Input Form with better layout
        with st.form("salary_form"):
            st.markdown("### Enter Your Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                job = st.selectbox(
                    "🎯 Job Title",
                    le_job.classes_,
                    help="Select your job role"
                )
                experience = st.selectbox(
                    "📊 Experience Level",
                    le_exp.classes_,
                    help="Your years of experience"
                )
                
            with col2:
                industry = st.selectbox(
                    "🏢 Industry",
                    le_ind.classes_,
                    help="The industry you work in"
                )
                city = st.selectbox(
                    "📍 Location",
                    le_city.classes_,
                    help="City where you work"
                )

            st.markdown("---")
            submitted = st.form_submit_button("🚀 Calculate Salary", use_container_width=True)

        # --- 3. PREDICTION LOGIC ---
        if submitted:
            with st.spinner("🔍 Analyzing market trends..."):
                time.sleep(0.8)
                
                # Prepare Inputs
                job_num = le_job.transform([job])[0]
                exp_num = le_exp.transform([experience])[0]
                ind_num = le_ind.transform([industry])[0]
                city_num = le_city.transform([city])[0]

                # Predict
                raw_prediction = model.predict([[job_num, exp_num, ind_num, city_num]])[0]
                
                # Adjustments
                adjusted_salary = raw_prediction
                if is_annual:
                    adjusted_salary = adjusted_salary / 12
                final_salary = adjusted_salary * currency_rate
                annual_salary = final_salary * 12
                
                # --- ENHANCED RESULT DISPLAY ---
                st.markdown("### 💰 Your Estimated Salary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Monthly Salary</h4>
                        <h2 style="color: #333; margin: 5px 0;">LKR {final_salary:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Annual Salary</h4>
                        <h2 style="color: #333; margin: 5px 0;">LKR {annual_salary:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    daily_rate = final_salary / 22  # Assuming 22 working days
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Daily Rate</h4>
                        <h2 style="color: #333; margin: 5px 0;">LKR {daily_rate:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # Job Details Summary
                st.markdown(f"""
                <div class="info-box">
                    <strong>Position:</strong> {experience} {job}<br>
                    <strong>Industry:</strong> {industry}<br>
                    <strong>Location:</strong> {city}
                </div>
                """, unsafe_allow_html=True)

                # --- COMPARISON CHART ---
                if show_comparison:
                    st.markdown("---")
                    st.markdown("### 📊 Location Comparison")
                    st.write(f"Compare salaries across different cities for **{job}** position")

                    # Get cities for comparison
                    all_cities = list(le_city.classes_)
                    comparison_cities = list(np.random.choice(
                        [c for c in all_cities if c != city], 
                        min(num_cities - 1, len(all_cities) - 1), 
                        replace=False
                    ))
                    comparison_cities.append(city)
                    comparison_cities.sort()

                    comparison_data = []
                    
                    for c in comparison_cities:
                        c_num = le_city.transform([c])[0]
                        pred = model.predict([[job_num, exp_num, ind_num, c_num]])[0]
                        
                        if is_annual: 
                            pred = pred / 12
                        pred = pred * currency_rate
                        
                        comparison_data.append({
                            "City": c, 
                            "Monthly Salary": pred,
                            "Selected": "Your City" if c == city else "Other"
                        })

                    chart_df = pd.DataFrame(comparison_data)
                    
                    # Create interactive Plotly chart
                    fig = px.bar(
                        chart_df, 
                        x="City", 
                        y="Monthly Salary",
                        color="Selected",
                        color_discrete_map={"Your City": "#667eea", "Other": "#b8c5e6"},
                        title=f"Salary Comparison for {job} Across Cities"
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        height=400
                    )
                    
                    fig.update_traces(
                        texttemplate='LKR %{y:,.0f}',
                        textposition='outside'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                # --- MARKET INSIGHTS ---
                if show_insights:
                    st.markdown("---")
                    st.markdown("### 💡 Market Insights")
                    
                    # Calculate some insights
                    avg_salary = chart_df["Monthly Salary"].mean()
                    salary_diff = ((final_salary - avg_salary) / avg_salary) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "vs. Average Salary",
                            f"LKR {final_salary:,.0f}",
                            f"{salary_diff:+.1f}%"
                        )
                    
                    with col2:
                        ranking = (chart_df["Monthly Salary"] >= final_salary).sum()
                        st.metric(
                            "City Ranking",
                            f"#{ranking} of {len(comparison_cities)}",
                            "Higher is better"
                        )

                # --- DOWNLOAD REPORT ---
                st.markdown("---")
                report_text = f"""
╔═══════════════════════════════════════════════════════════╗
║     SRI LANKA SALARY PREDICTION REPORT - 2025            ║
╚═══════════════════════════════════════════════════════════╝

📋 POSITION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Job Title:           {job}
Experience Level:    {experience}
Industry:            {industry}
Location:            {city}

💰 SALARY BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monthly Salary:      LKR {final_salary:,.2f}
Annual Salary:       LKR {annual_salary:,.2f}
Daily Rate:          LKR {daily_rate:,.2f}

📊 MARKET COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average (Sample):    LKR {avg_salary:,.2f}
Difference:          {salary_diff:+.1f}%
City Ranking:        #{ranking} of {len(comparison_cities)}

📍 CITY COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
                for _, row in chart_df.iterrows():
                    report_text += f"{row['City']:15} LKR {row['Monthly Salary']:>12,.2f}\n"

                report_text += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Powered by Sri Lanka Jobs AI Predictor
Exchange Rate: 1 USD = {currency_rate} LKR

⚠️  DISCLAIMER: This is an estimate based on market data and 
should be used as a reference only. Actual salaries may vary
based on company, skills, and negotiation.
                """
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Detailed Report",
                        data=report_text,
                        file_name=f"Salary_Report_{job.replace(' ', '_')}_{city}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

    with tab2:
        st.markdown("### 📈 Market Insights & Trends")
        st.info("💡 **Coming Soon:** Industry trends, salary growth projections, and demand analysis")
        
        # Placeholder for future analytics
        st.markdown("""
        This section will include:
        - 📊 Industry-wise salary trends
        - 📈 Year-over-year growth rates
        - 🔥 High-demand job roles
        - 🌍 Regional market analysis
        - 💼 Skills that command premium salaries
        """)

    with tab3:
        st.markdown("### ℹ️ How to Use This Tool")
        
        st.markdown("""
        #### 📝 Steps:
        1. **Select your job details** in the Salary Prediction tab
        2. **Click "Calculate Salary"** to get your estimate
        3. **Review the results** including city comparisons
        4. **Download the report** for your records
        
        #### 🎯 Tips for Accurate Results:
        - Choose the most accurate job title that matches your role
        - Select the experience level that best represents your background
        - Ensure the city matches your work location
        
        #### ⚙️ Settings:
        - Adjust the **exchange rate** in the sidebar if needed
        - Toggle **city comparison** to see location-based differences
        - Customize the number of cities to compare
        
        #### 📊 Understanding Results:
        - **Monthly Salary**: Your estimated monthly compensation
        - **Annual Salary**: Total yearly compensation
        - **Daily Rate**: Useful for freelance or contract work
        - **City Comparison**: How your location compares to others
        
        #### ⚠️ Important Notes:
        - These are estimates based on market data
        - Actual salaries depend on company, skills, and negotiation
        - Use this as a reference point for salary discussions
        """)

else:
    st.error("⚠️ Application cannot start. Please ensure both 'salary_model.pickle' and 'encoders.pkl' are present.")
    st.info("📁 Required files: `salary_model.pickle` (or `.pkl`) and `encoders.pkl`")