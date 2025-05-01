import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta
from wordcloud import WordCloud
import plotly.express as px
from connect import MongoDbConnector
from sklearn.preprocessing import LabelEncoder


object = MongoDbConnector()

st.title("THE GREATEST PROJECT OF DATA SCIENCE TOOLS 🔥")

# Sidebar navigation
st.sidebar.title("Job Dashboard")
# Sidebar content
page = st.sidebar.radio("Navigate", ["Software Jobs", "Data Science Jobs", "Cybersecurity Jobs", "All Jobs"])
 
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #0B1F3A;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
json_files = object.get('Software')
json_files_sci = object.get('Data')
json_files_cyb  = object.get('Cyber')

# Software Jobs Page
if page == "Software Jobs":
    st.title("💻 Upload Software Job Data Files")
    if json_files:
      dfs = [pd.json_normalize(item) if isinstance(item, dict) else pd.read_json(item) for item in json_files]
      st.session_state.software_df = pd.concat(dfs, ignore_index=True)
     
    if 'software_df' in st.session_state:
        st.success("your files uploaded success")
        combined_df = st.session_state.software_df

        st.write("so, this is all jobs in software folders -- >",len(combined_df))
        st.write("")
        st.write("And this sample of it")
        st.dataframe(combined_df.head())
        st.write("")
        st.write("but In our project we search for (full stack ,frontend , backend ,devops)")
        st.markdown("### So,")


        combined_df['title'] =combined_df['title'].str.lower()

        front_end = combined_df[combined_df['title'].str.contains(r'front|Ui Ux', case=False ,na=False)]
        st.write("this is the jobs in front end -- >",len(front_end))

        full_stack = combined_df[combined_df['title'].str.contains(r'stack|software engineer', case=False , na=False)]
        st.write("this is the jobs in full stack -- >",len(full_stack))

        devops = combined_df[combined_df['title'].str.contains(r'devops|infrastructure engineer', case=False, na=False)]
        st.write("this is the jobs in devops -- >",len(devops))

        backend = combined_df[combined_df['title'].str.contains(r'back|software engineer',case=False , na=False)]
        st.write("this is the jobs in backend -- >",len(backend))
        
        soft_jobs = pd.concat([front_end, backend, devops,full_stack], ignore_index=True)
        st.session_state.softjobs=soft_jobs

        st.write("And combin all this data to be in one data frame with ",len(soft_jobs)," rows which mean number of jops")
        st.markdown("### this our data now of our jops")
        st.dataframe(soft_jobs.head(10))

        print(soft_jobs.isnull().sum())
        def convert_posted_date(text):
            try:
                text = text.strip().replace('Posted', '').replace('ago', '').strip()

                if 'day' in text:
                    days = int(text.replace('days', '').replace('day', '').strip())
                    posted_date = datetime.today() - timedelta(days=days)
                elif 'month' in text:
                    months = int(text.replace('months', '').replace('month', '').strip())
                    posted_date = datetime.today() - timedelta(days=months * 30)
                else:
                    posted_date = pd.NaT
                return posted_date.date()

            except Exception as e:
                print(f"Error: {e}")
                return pd.NaT
        st.markdown("### We make a clean for the posted date columnto be like:")
        
        soft_jobs['posted_date'] = soft_jobs['posted_date'].apply(convert_posted_date)
        st.dataframe(soft_jobs['posted_date'].head())

        job_counts = soft_jobs['title'].value_counts()

        st.markdown("# Visualize data")
        st.write("so, we make a visualize for jobs and the number of postig date")
        st.markdown("### For all data")
        plt.figure(figsize=(30,10))  
        bars = plt.bar(job_counts.index, job_counts.values, color='skyblue', width=0.5)
        plt.title('Distribution of Software Job Titles', fontsize=22)
        plt.xlabel('Job Title', fontsize=18)
        plt.ylabel('Number of Postings', fontsize=18)
        plt.xticks(rotation=75, ha='right', fontsize=12)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)


        st.markdown("### For the top 20 jobs")
        print('Top 20 Software Job Titles')
        top_jobs = job_counts.head(20)  

        plt.figure(figsize=(18,8))
        bars = plt.bar(top_jobs.index, top_jobs.values, color='skyblue', width=0.5)  # عرض أعمدة أعرض شوية

        plt.title('Top 20 Software Job Titles', fontsize=18)
        plt.xlabel('Job Title', fontsize=14)
        plt.ylabel('Number of Postings', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)

        # كتابة الأرقام فوق الأعمدة
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("### Some analysis on postings date over time by days")

        date_counts = soft_jobs['posted_date'].value_counts().sort_index()
        plt.figure(figsize=(16,8))
        plt.plot(date_counts.index, date_counts.values, marker='o', linestyle='-', color='teal', linewidth=2, markersize=6)
        plt.title('Number of Job Postings Over Time', fontsize=18)
        plt.xlabel('Posted Date', fontsize=14)
        plt.ylabel('Number of Postings', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=12)

        # اكتب الأرقام فوق كل نقطة
        for x, y in zip(date_counts.index, date_counts.values):
            plt.text(x, y+0.5, str(y), ha='center', va='bottom', fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

       # 1. Split the skills string into a list
        soft_jobs['skills'] = soft_jobs['skills'].str.split(',')

        # 2. Flatten all the skills into one big list
        all_skills = soft_jobs['skills'].explode()

        # 3. Remove extra spaces from each skill (like ' Python' -> 'Python')
        all_skills = all_skills.str.strip()

        # -----------------------------
        # Count top skills
        # -----------------------------
        skill_counts = all_skills.value_counts()

        # -----------------------------
        # Show the top 20 most frequent skills
        # -----------------------------
        top_skills = skill_counts.head(20)

        # Normalize the counts for coloring
        norm = plt.Normalize(top_skills.min(), top_skills.max())
        colors = plt.cm.viridis(norm(top_skills))  # You can change 'viridis' to another colormap

        # -----------------------------
        # Plot Visualization
        # -----------------------------
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot on the given ax
        bars = ax.barh(top_skills.index, top_skills.values, color=colors)

        # Add numbers on bars
        for bar in bars:
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    int(bar.get_width()),
                    va='center')

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # required for colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Number of Job Listings Requiring Skill')

        # Labels and style
        ax.set_xlabel('Number of Job Listings Requiring Skill')
        ax.set_title('Top 20 Most Required Skills')
        ax.invert_yaxis()
        ax.grid(axis='x')

        st.pyplot(plt)

        combined_df = st.session_state.software_df
        combined_df['title'] =combined_df['title'].str.lower()

        front_end = combined_df[combined_df['title'].str.contains(r'front|Ui Ux', case=False ,na=False)]

        full_stack = combined_df[combined_df['title'].str.contains(r'stack|software engineer', case=False , na=False)]

        devops = combined_df[combined_df['title'].str.contains(r'devops|infrastructure engineer', case=False, na=False)]


        backend = combined_df[combined_df['title'].str.contains(r'back|software engineer',case=False , na=False)]
        
        soft_jobs = pd.concat([front_end, backend, devops,full_stack], ignore_index=True)
       # Define the real software skills list
        available_software_skills = [
            'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'node.js', 'react', 'angular',
            'git', 'docker', 'kubernetes', 'aws', 'azure', 'machine learning', 'data structures', 'algorithms',
            'devops', 'spring', 'hibernate', 'mongodb', 'postgresql', 'redis', 'typescript', 'vue.js'
        ]

        # Initialize a list to store all skills
        collected_skills = []

        # Iterate through each job posting and process the skills
        for skill_set in soft_jobs['skills']:
            if isinstance(skill_set, str):  # Ensure the skills are a string, not NaN
                # Split the skills string and filter by the available software skills
                filtered_skills = [skill.strip().lower() for skill in skill_set.split(',')]
                filtered_skills = [skill for skill in filtered_skills if skill in available_software_skills]
                collected_skills.extend(filtered_skills)

        # -----------------------------
        # Count top skills
        # -----------------------------
        from collections import Counter

        # Count the frequency of each skill
        skill_frequency = Counter(collected_skills)

        # Get the top 20 most common skills
        top_skills = skill_frequency.most_common(20)

        # -----------------------------
        # Plot Visualization
        # -----------------------------

        # Normalize the counts for coloring
        norm = plt.Normalize(min(skill_frequency.values()), max(skill_frequency.values()))
        colors = plt.cm.viridis(norm([count for skill, count in top_skills]))  # Use viridis colormap

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the horizontal bar chart with the color map
        bars = ax.barh([skill for skill, _ in top_skills], [count for _, count in top_skills], color=colors)

        # Add numbers on the bars
        for bar in bars:
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    int(bar.get_width()),
                    va='center')

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # required for colorbar
        cbar = fig.colorbar(sm, ax=ax)  # Attach colorbar to the specific ax
        cbar.set_label('Number of Job Listings Requiring Skill')

        # Labels and style
        ax.set_xlabel('Number of Jobs')
        ax.set_ylabel('Skill')
        ax.set_title('Top 20 Software Engineering Skills')
        ax.invert_yaxis()  # To show the highest skill on top
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Plot the visualization in Streamlit
        st.pyplot(plt)

# Data Science Jobs Page
elif page == "Data Science Jobs":
    st.title("📊 Upload Data Science Job Data Files")
    if json_files_sci:
        dfs_sci = [pd.json_normalize(item) if isinstance(item, dict) else pd.read_json(item) for item in json_files_sci]
        st.session_state.data_sci_df = pd.concat(dfs_sci, ignore_index=True)

    if 'data_sci_df' in st.session_state:
        st.success("your files uploaded success")
        combined_df_sci = st.session_state.data_sci_df

        combined_df_sci['title'] = combined_df_sci['title'].str.lower()

        st.markdown("sample of the all folders jobs of data science")
        st.dataframe(combined_df_sci.head())

        st.write("so, this is all jobs in data science folders -- >",len(combined_df_sci))
        st.write("there is a problem that data science has so many fileds so, i need some help.")
        st.write("")
        st.markdown("### So,")
        st.write("I use this this visualize to help me find the the most words have mentioned in title job ")

        # Sample text data (you can replace this with your data)
        text = ' '.join(combined_df_sci['title'].astype(str))
        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Turn off axis numbers and ticks
        st.pyplot(plt)



        st.markdown("### We search for specific jobs we need in this fields")
        st.write("For data analsis we search for : analyst , data entry , analytics , data analysis")
        st.write("For data science we search for : data scient , machine learning , business analyst , financial analyst , ai , lead")
        st.write("For engineering we search for : support engineer , data engineer, data architect , senior engineer")
        st.write("")
        # List of keywords/phrases to search for
        keywords = [
             'analyst','data entry','analytics','data analysis','data scient','support engineer',
             'data engineer','data architect','senior engineer','machine learning','business analyst',
             'financial analyst','ai','lead'
            ]  

        # Convert the list of keywords into a regular expression pattern (case-insensitive)
        pattern = '|'.join(keywords)

        # Filter the DataFrame for titles that contain any of the keywords (case-insensitive)

        sci_jobs = combined_df_sci[combined_df_sci['title'].str.contains(pattern, case=False, na=False)]

        st.session_state.scijobs=sci_jobs
        st.write("Now data have ",len(sci_jobs))

        st.markdown("### Sample of the data")
        st.dataframe(sci_jobs.head(10))

        st.markdown("# visualize data")
        st.markdown("### There the most 20 skills needed in data science fileds")



        # 1. Split the skills string into a list
        sci_jobs['skills'] = sci_jobs['skills'].str.split(',')

        # 2. Flatten all the skills into one big list
        all_skills = sci_jobs['skills'].explode()

        # 3. Remove extra spaces from each skill (like ' Python' -> 'Python')
        all_skills = all_skills.str.strip()

        # 4. Count the frequency of each skill
        skill_counts = all_skills.value_counts()
 
        # 5. Show the top 20 most frequent skills
        st.write("Top 20 skills in data science",skill_counts.head(20))

        # Get top 20 skills
        top_skills = skill_counts.head(20)

        # Normalize the counts for coloring
        norm = plt.Normalize(top_skills.min(), top_skills.max())
        colors = plt.cm.viridis(norm(top_skills))  # You can change 'viridis' to another colormap

        # Create figure and axis manually
        fig, ax = plt.subplots(figsize=(12,6))

        # Now plot on the given ax
        bars = ax.barh(top_skills.index, top_skills.values, color=colors)

        # Add numbers on bars
        for bar in bars:
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height()/2,
                    int(bar.get_width()),
                    va='center')

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # required for colorbar
        cbar = fig.colorbar(sm, ax=ax)  # <--- attach to the correct ax
        cbar.set_label('Number of Job Listings Requiring Skill')

        # Labels and style
        ax.set_xlabel('Number of Job Listings Requiring Skill')
        ax.set_title('Top 20 Most Required Skills')
        ax.invert_yaxis()
        ax.grid(axis='x')

        st.pyplot(plt)
        #-----------------------------------------------------------------------------------------------------------------------------------
        combined_df_sci = st.session_state.data_sci_df

        combined_df_sci['title'] = combined_df_sci['title'].str.lower()
        # Sample dataframe creation (replace with your actual data loading)
        keywords = [
             'analyst','data entry','analytics','data analysis','data scient','support engineer',
             'data engineer','data architect','senior engineer','machine learning','business analyst',
             'financial analyst','ai','lead'
            ]  

        # Convert the list of keywords into a regular expression pattern (case-insensitive)
        pattern = '|'.join(keywords)

        # Filter the DataFrame for titles that contain any of the keywords (case-insensitive)

        sci_jobs = combined_df_sci[combined_df_sci['title'].str.contains(pattern, case=False, na=False)]
        # 1. Split the skills string into a list
        sci_jobs['skills'] = sci_jobs['skills'].str.split(',')

        # 2. Flatten all the skills into one big list
        all_ds_skills = sci_jobs['skills'].explode()

        # 3. Remove extra spaces from each skill
        all_ds_skills = all_ds_skills.astype(str).str.strip()

        # 4. Define a list of real data science skills (expandable)
        ds_real_skills = {
            'Python', 'R', 'SQL', 'Julia', 'Scala',
            'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Scikit-learn',
            'Statsmodels', 'XGBoost', 'LightGBM', 'CatBoost', 'NLTK',
            'SpaCy', 'Gensim', 'OpenCV', 'Keras', 'TensorFlow', 'PyTorch',
            'Jupyter', 'Google Colab', 'Power BI', 'Tableau', 'Excel',
            'Apache Spark', 'Hadoop', 'Airflow', 'MLflow', 'DVC', 'KubeFlow',
            'AWS', 'Azure', 'GCP', 'Google BigQuery', 'SageMaker', 'Databricks',
            'PostgreSQL', 'MongoDB', 'MySQL', 'Snowflake', 'ClickHouse',
            'NoSQL', 'Presto', 'Hive', 'Redshift',
            'Statistics', 'Probability', 'Machine Learning', 'Deep Learning',
            'Reinforcement Learning', 'Natural Language Processing', 'Computer Vision',
            'Data Visualization', 'Data Cleaning', 'EDA', 'Feature Engineering',
            'Big Data', 'Data Mining', 'Data Engineering', 'A/B Testing',
            'Time Series Analysis', 'Bayesian Inference', 'Linear Regression',
            'Classification', 'Clustering', 'Recommendation Systems',
            'Docker', 'Kubernetes', 'Flask', 'FastAPI', 'Streamlit', 'Gradio',
            'Git', 'CI/CD',
            'Business Intelligence', 'ETL', 'Data Pipelines', 'Dash', 'Plotly'
        }

        # 5. Normalize all skills to lowercase
        all_ds_skills = all_ds_skills.str.lower()

        # 6. Normalize real skills to lowercase and map to preserve original casing
        ds_real_skills_lower = {s.lower(): s for s in ds_real_skills}

        # 7. Filter only matching skills and map them back to their original casing
        ds_filtered_skills = all_ds_skills[all_ds_skills.isin(ds_real_skills_lower.keys())]
        ds_filtered_skills = ds_filtered_skills.map(ds_real_skills_lower)

        # 8. Count the frequency of each skill
        ds_skill_counts = ds_filtered_skills.value_counts()

        # 9. Get top 30 skills
        top_ds_skills = ds_skill_counts.head(30)
        print(top_ds_skills)

        # 10. Normalize the counts for coloring
        ds_norm = plt.Normalize(top_ds_skills.min(), top_ds_skills.max())
        ds_colors = plt.cm.viridis(ds_norm(top_ds_skills))

        # 11. Create figure and axis manually
        fig_ds, ax_ds = plt.subplots(figsize=(12, 6))

        # 12. Plot on the given ax
        ds_bars = ax_ds.barh(top_ds_skills.index, top_ds_skills.values, color=ds_colors)

        # 13. Add numbers on bars
        for bar in ds_bars:
            ax_ds.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    int(bar.get_width()),
                    va='center')

        # 14. Create colorbar
        ds_sm = plt.cm.ScalarMappable(cmap='viridis', norm=ds_norm)
        ds_sm.set_array([])
        ds_cbar = fig_ds.colorbar(ds_sm, ax=ax_ds)
        ds_cbar.set_label('Number of Job Listings Requiring Skill')

        # 15. Labels and style
        ax_ds.set_xlabel('Number of Job Listings Requiring Skill')
        ax_ds.set_title('Real Data Science Skills')
        ax_ds.invert_yaxis()
        ax_ds.grid(axis='x')

        plt.tight_layout()
        st.pyplot(plt)









        
# Cybersecurity Jobs Page
elif page == "Cybersecurity Jobs":
    st.title("🛡️ Upload Cybersecurity Job Data Files")
    
    if json_files_cyb:
        dfs_cyb = [pd.json_normalize(item) if isinstance(item, dict) else pd.read_json(item) for item in json_files_cyb]
        st.session_state.cyber_df = pd.concat(dfs_cyb, ignore_index=True)

    if 'cyber_df' in st.session_state:
        st.success("your files uploaded success")
        cyb_jops = st.session_state.cyber_df
        cyb_jops['title'] = cyb_jops['title'].str.lower()
        
        st.session_state.cybjobs=cyb_jops

        st.markdown("sample of the all folders jobs of data science")
        st.dataframe(cyb_jops.head())
        st.write("This is all jobs in data science folders -- >",len(cyb_jops))
        st.write("there is a problem that cyber security fileds a few.")
        st.write("so,")
        st.write("I use all the data and i didn' make any search for specific filed.")
        st.markdown("### but ,")
        st.write("IF i want to search for the most fields visualize to help me find the the most words have mentioned in cyber security")
        st.write("")
        st.write("this visualize help to find the the most words have mentioned in cyber security")
        # Sample text data (you can replace this with your data)
        text = ' '.join(cyb_jops['title'].astype(str))

        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Turn off axis numbers and ticks
        st.pyplot(plt)

        st.markdown("# visulaize data")
        st.write("We have ensure that all data selected have it location not a nan value using this:")
        st.write("combined_df_cyb[['location']].dropna()")
        st.write("")
        st.markdown("### Location of the most cyber security jobs")
        df_plot = cyb_jops[['location']].dropna()
        # Count all locations
        location_counts = df_plot['location'].value_counts()

        # Use all locations
        labels = location_counts.index
        sizes = location_counts.values

        # Plot pie chart
        fig, ax = plt.subplots(figsize=(6,6))

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,  # Label locations directly
            autopct='%1.1f%%',  # Show % inside
            startangle=140,
            textprops=dict(color="black", fontsize=14),
            radius=0.7
        )

        # No legend
        # Beautify
        ax.set_title('Job Distribution by Location', fontsize=12,loc = 'left')
        plt.tight_layout()
        st.pyplot(plt)


        st.markdown("### Names of companies and thiere location")
        df_plot = cyb_jops[['location', 'company']].dropna()
        # Group the data and create a pivot table
        pivot_table = df_plot.groupby(['company', 'location']).size().unstack(fill_value=0)
        # Sort companies if you want (optional)
        pivot_table = pivot_table.sort_index()
        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(14,8))
        # Draw the stacked bar chart
        pivot_table.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.8)
        # Beautify the chart
        ax.set_title('Stacked Bar Chart: Locations across Companies', fontsize=20)
        ax.set_xlabel('Company', fontsize=16)
        ax.set_ylabel('Number of Job Listings', fontsize=16)
        ax.legend(title='Location', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=90)  # Rotate company names for better readability
        plt.tight_layout()
        st.pyplot(plt)


        st.markdown("### And this visulazie for the job title , comapny and it's location")
        # Focus only on needed columns
        df_plot = cyb_jops[['title', 'company', 'location']].dropna()

        # Encode text into numbers
        label_encoder_title = LabelEncoder()
        label_encoder_company = LabelEncoder()
        label_encoder_location = LabelEncoder()

        df_plot['title_encoded'] = label_encoder_title.fit_transform(df_plot['title'])
        df_plot['company_encoded'] = label_encoder_company.fit_transform(df_plot['company'])
        df_plot['location_encoded'] = label_encoder_location.fit_transform(df_plot['location'])

        # Create an interactive 3D scatter plot
        fig = px.scatter_3d(
            df_plot,
            x='title_encoded',
            y='company_encoded',
            z='location_encoded',
            color='location',  # Different colors by location
            hover_data=['title', 'company', 'location'],
            title='Beautiful 3D Visualization of Job Titles, Companies, and Locations',
            height=800,
            width=1000
        )

        # Improve the look
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            scene=dict(
                xaxis_title='Job Title',
                yaxis_title='Company',
                zaxis_title='Location',
                bgcolor='rgba(0,0,0,0)',  # Transparent background
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            template='plotly_dark'  # Dark beautiful theme
        )
        st.plotly_chart(fig, use_container_width=True)
        # Assume combined_df_cyb is already loaded

        # Focus only on title and company
        df_plot = cyb_jops[['title', 'company']].dropna()

        # Group the data and create a pivot table
        pivot_table = df_plot.groupby(['company', 'title']).size().unstack(fill_value=0)

        # Reset index to make the 'company' a column
        pivot_table = pivot_table.reset_index()

        # Melt the DataFrame to long format for Plotly
        df_long = pivot_table.melt(id_vars=['company'], var_name='title', value_name='count')

        # Create the interactive stacked bar chart
        fig = px.bar(df_long,
                    x='company',
                    y='count',
                    color='title',
                    title='Interactive Stacked Bar Chart: Titles across Companies',
                    labels={'company': 'Company', 'count': 'Number of Job Listings'},
                    hover_data={'company': True, 'title': True, 'count': True})  # Hover data

        # Update layout for better aesthetics and bigger chart
        fig.update_layout(
            barmode='stack',
            xaxis_tickangle=45,
            width=1200,  # Increase the width
            height=800,  # Increase the height
        )

        # Show the interactive chart
        st.plotly_chart(fig, use_container_width=True)
elif page == "All Jobs":

    # Check that all needed DataFrames exist in session_state
    if all(key in st.session_state for key in ['cybjobs', 'scijobs', 'softjobs']):
        cyb = st.session_state.cybjobs
        sci = st.session_state.scijobs
        soft = st.session_state.softjobs

        # Combine the dataframes
        combined_data = pd.concat([cyb, sci, soft], ignore_index=True)

        # Count the top 20 most frequent titles
        top_titles = combined_data['title'].value_counts().head(20)

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        ax = top_titles.plot(kind='bar', color='skyblue')

        # Add numbers on top of each bar
        for i, v in enumerate(top_titles):
            ax.text(i, v + 0.2, str(v), ha='center', fontsize=12)

        st.markdown("# this the most 20 jobs in all fileds")

        plt.title('Top 20 Most Frequent Job Titles', fontsize=16)
        plt.xlabel('Job Title', fontsize=12)
        plt.ylabel('Number of Job Listings', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

        # Define the function inside the block
        def count_job_types(df, dataset_name):
            hybrid = 0
            onsite = 0
            remote = 0

            columns_to_search = ['description', 'title']
            if 'full_text' in df.columns:
                columns_to_search.append('full_text')

            for idx, row in df[columns_to_search].dropna(how='all').iterrows():
                combined_text = ' '.join([str(row[col]).lower() for col in columns_to_search if pd.notnull(row[col])])

                if 'hybrid' in combined_text:
                    hybrid += 1
                elif 'on site' in combined_text or 'onsite' in combined_text:
                    onsite += 1
                elif 'remote' in combined_text or 'work from home' in combined_text:
                    remote += 1
            return pd.Series({'Hybrid': hybrid, 'On Site': onsite, 'Remote': remote}, name=dataset_name)

        # Call the function safely
        sci_counts = count_job_types(sci, 'filtered_df')
        cyb_counts = count_job_types(cyb, 'combined_df_cyb')
        soft_counts = count_job_types(soft, 'soft_jobs')

        # Combine all results
        all_counts = sci_counts + cyb_counts + soft_counts

        # Plot Pie Chart
        st.write("")
        st.write("")
        plt.figure(figsize=(4, 4))
        plt.pie(all_counts, labels=all_counts.index, autopct='%1.1f%%', startangle=100, colors=['#66b3ff','#99ff99','#ffcc99'])
        plt.title('Job Types Distribution (Hybrid, On Site, Remote)', fontsize=7)
        plt.axis('equal')
        st.pyplot(plt)
        
        st.markdown("# This show us the percent of each job ")

        job_counts = {
            'Cyber Jobs': len(cyb['title']),
            'Science Jobs': len(sci['title']),
            'Software Jobs': len(soft['title'])
        }
        # Pie chart
        labels = list(job_counts.keys())
        sizes = list(job_counts.values())

        plt.figure(figsize=(7, 7))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,colors =['#FF4136', '#2ECC40', '#007BFF'])
        plt.title('Job Distribution')
        plt.axis('equal')  # Makes the pie chart a circle
        st.pyplot(plt)
    else:
        st.warning("Please upload job data for all categories (Software, Data Science, and Cybersecurity) first.")


