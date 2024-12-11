import streamlit as st
import pandas as pd
import pandasql as pds
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import io

warnings.filterwarnings('ignore')

def convert_df(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

# Function to download plots as PNG
def download_plot_as_png(fig, filename="plot.png"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

def add_spacing():
    st.write("\n")
    st.divider()

# Dashboard setup
st.title("SBDC Client Milestone Analysis")

# File upload
st.sidebar.header("Upload Datasets")
hours_file = st.sidebar.file_uploader("Upload Consulting Hours Dataset", type=["xlsx", "csv"])
capital_file = st.sidebar.file_uploader("Upload Capital Milestones Dataset", type=["xlsx", "csv"])
established_file = st.sidebar.file_uploader("Upload Business Established Dataset", type=["xlsx", "csv"])

# Processing if all files are uploaded
if hours_file and capital_file and established_file:
    # Load datasets
    hours = pd.read_excel(hours_file) if hours_file.name.endswith('.xlsx') else pd.read_csv(hours_file)
    capital = pd.read_excel(capital_file) if capital_file.name.endswith('.xlsx') else pd.read_csv(capital_file)
    established = pd.read_excel(established_file) if established_file.name.endswith('.xlsx') else pd.read_csv(established_file)

    # Renaming columns for consistency
    hours = hours.rename(columns={
        'Client ID': 'Client ID',
        'Session Date': 'Session Date',
        'Total Hours': 'Total Counseling Hours'
    })
    capital = capital.rename(columns={
        'Client ID': 'Client ID',
        'Funding Type': 'Funding Type',
        'Reporting Date': 'Capital Reporting Date'
    })
    established = established.rename(columns={
        'Client ID': 'Client ID',
        'Milestone Date': 'Business Established Date'
    })

    # Drop null Client ID and process dates
    hours = hours.dropna(subset=['Client ID'])
    capital = capital.dropna(subset=['Client ID'])
    established = established.dropna(subset=['Client ID'])
    established['Business Established Date'] = pd.to_datetime(established['Business Established Date'], errors='coerce')
    hours['Session Date'] = pd.to_datetime(hours['Session Date'], errors='coerce')

    # Tabs for analysis
    tab1, tab2 = st.tabs(["Capital Analysis", "Established Analysis"])

    with tab1:
        st.header("Capital Analysis")

        # SQL query to join hours and capital datasets
        query_capital = """
        SELECT 
            hours."Client ID" AS "Client ID",
            hours."Session Date" AS "Session Date",
            hours."Total Counseling Hours" AS "Session Hours",
            capital."Funding Type" AS "Funding Type",
            capital."Capital Reporting Date" AS "Capital Reporting Date"
        FROM hours
        LEFT JOIN capital
        ON hours."Client ID" = capital."Client ID"
        ORDER BY capital."Capital Reporting Date"
        """
        bus_capital = pds.sqldf(query_capital, locals())

        # SQL query to summarize sessions and hours before funding
        query_funding = """
        SELECT 
            "Client ID",
            "Funding Type",
            "Capital Reporting Date",
            COUNT("Session Date") AS "Total Sessions",
            ROUND(SUM("Session Hours")) AS "Total Hours"
        FROM bus_capital
        WHERE "Session Date" <= "Capital Reporting Date"
        GROUP BY "Client ID", "Funding Type", "Capital Reporting Date"
        ORDER BY "Capital Reporting Date"
        """
        client_funding_summary = pds.sqldf(query_funding, locals())
        client_funding_summary['Capital Reporting Date'] = pd.to_datetime(client_funding_summary['Capital Reporting Date']).dt.date
        client_funding_summary['Capital Reporting Date'] = pd.to_datetime(client_funding_summary['Capital Reporting Date'])

        # Display full dataset
        st.subheader("Full Dataset")
        st.dataframe(client_funding_summary)
        st.download_button(
            "Download Dataset", 
            data=convert_df(client_funding_summary), 
            file_name="Full_Capital_Analysis.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        add_spacing()

        # Filters
        st.subheader("Data Filters")
        min_date = pd.to_datetime(client_funding_summary['Capital Reporting Date']).min().date()
        max_date = pd.to_datetime(client_funding_summary['Capital Reporting Date']).max().date()

        # Initialize session state for filters if not already set
        if "cap_start_date" not in st.session_state:
            st.session_state["cap_start_date"] = min_date
            st.session_state["cap_end_date"] = max_date
            st.session_state["cap_funding_types"] = ["All"]

        # User input for filters
        start_date = st.date_input("Start Date", value=st.session_state["cap_start_date"], min_value=min_date, max_value=max_date, key="cap_start")
        end_date = st.date_input("End Date", value=st.session_state["cap_end_date"], min_value=min_date, max_value=max_date, key="cap_end")
        funding_types = client_funding_summary['Funding Type'].dropna().unique().tolist()
        funding_types.insert(0, "All")
        selected_funding_types = st.multiselect("Select Funding Type(s)", options=funding_types, default=st.session_state["cap_funding_types"])

        # Generate Results Button
        if st.button("Generate Results", key="generate_cap"):
            # Save filters to session state
            st.session_state["cap_start_date"] = start_date
            st.session_state["cap_end_date"] = end_date
            st.session_state["cap_funding_types"] = selected_funding_types

            # Filter the data and save to session state
            filtered_summary = client_funding_summary[
                (pd.to_datetime(client_funding_summary['Capital Reporting Date']).dt.date >= start_date) &
                (pd.to_datetime(client_funding_summary['Capital Reporting Date']).dt.date <= end_date)
            ]
            if "All" not in selected_funding_types:
                filtered_summary = filtered_summary[filtered_summary['Funding Type'].isin(selected_funding_types)]

            st.session_state["cap_filtered_summary"] = filtered_summary

        # Display results only when available in session state
        if "cap_filtered_summary" in st.session_state:
            filtered_summary = st.session_state["cap_filtered_summary"]

            avg_sessions_by_time = filtered_summary.groupby('Capital Reporting Date')['Total Sessions'].mean()
            avg_hours_by_time = filtered_summary.groupby('Capital Reporting Date')['Total Hours'].mean()
            milestone_count = len(filtered_summary)
            overall_avg_sessions = filtered_summary['Total Sessions'].mean()
            overall_avg_hours = filtered_summary['Total Hours'].mean()

            add_spacing()

            st.write(f"**Number of Milestones in Selected Range:** {milestone_count}")
            st.write(f"**Overall Average Sessions:** {overall_avg_sessions:.2f}")
            st.write(f"**Overall Average Hours:** {overall_avg_hours:.2f}")

            add_spacing()

            # Plot: Average Sessions Over Time
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            ax1.plot(avg_sessions_by_time.index, avg_sessions_by_time, label='Average Sessions', color='blue')
            ax1.set_title("Average Sessions Over Time", pad=20)
            ax1.set_xlabel("Capital Reporting Date", labelpad=20)
            ax1.set_ylabel("Average Sessions", labelpad=20)
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            st.download_button("Download Sessions Plot", data=download_plot_as_png(fig1), file_name="sessions_plot.png")

            add_spacing()

            # Plot: Average Hours Over Time
            fig2, ax2 = plt.subplots(figsize=(15, 8))
            ax2.plot(avg_hours_by_time.index, avg_hours_by_time, label='Average Hours', color='green')
            ax2.set_title("Average Hours Over Time", pad=20)
            ax2.set_xlabel("Capital Reporting Date", labelpad=20)
            ax2.set_ylabel("Average Hours", labelpad=20)
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            st.pyplot(fig2)

            st.download_button("Download Hours Plot", data=download_plot_as_png(fig2), file_name="hours_plot.png")


    # Established Analysis Tab
    with tab2:
        st.header("Established Analysis")

        # SQL Query for Business Established Analysis
        query_established = """
        SELECT 
            hours."Client ID" AS "Client ID",
            established."Business Established Date" AS "Business Established Date",
            COUNT(hours."Session Date") AS "Total Sessions",
            ROUND(SUM(hours."Total Counseling Hours")) AS "Total Hours"
        FROM hours
        LEFT JOIN established
        ON hours."Client ID" = established."Client ID"
        WHERE hours."Session Date" <= established."Business Established Date"
        GROUP BY hours."Client ID", "Business Established Date"
        ORDER BY "Business Established Date"
        """
        client_milestone_summary = pds.sqldf(query_established, locals())
        client_milestone_summary['Business Established Date'] = pd.to_datetime(client_milestone_summary['Business Established Date']).dt.date
        client_milestone_summary['Business Established Date'] = pd.to_datetime(client_milestone_summary['Business Established Date'])

        st.subheader("Full Dataset")
        st.dataframe(client_milestone_summary)
        st.download_button(
            "Download Dataset", 
            data=convert_df(client_milestone_summary), 
            file_name="Full_Established_Analysis.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        add_spacing()

        # Filters
        st.subheader("Data Filters")
        min_date = pd.to_datetime(client_milestone_summary['Business Established Date']).min().date()
        max_date = pd.to_datetime(client_milestone_summary['Business Established Date']).max().date()

        # Initialize session state for filters if not already set
        if "est_start_date" not in st.session_state:
            st.session_state["est_start_date"] = min_date
            st.session_state["est_end_date"] = max_date

        # User input for filters
        start_date = st.date_input("Start Date", value=st.session_state["est_start_date"], min_value=min_date, max_value=max_date, key="est_start")
        end_date = st.date_input("End Date", value=st.session_state["est_end_date"], min_value=min_date, max_value=max_date, key="est_end")

        # Generate Results Button
        if st.button("Generate Results", key="generate_est"):
            # Save filters to session state
            st.session_state["est_start_date"] = start_date
            st.session_state["est_end_date"] = end_date

            # Filter the data and save to session state
            filtered_summary = client_milestone_summary[
                (pd.to_datetime(client_milestone_summary['Business Established Date']).dt.date >= start_date) &
                (pd.to_datetime(client_milestone_summary['Business Established Date']).dt.date <= end_date)
            ]

            st.session_state["est_filtered_summary"] = filtered_summary

        # Display results only when available in session state
        if "est_filtered_summary" in st.session_state:
            filtered_summary = st.session_state["est_filtered_summary"]

            avg_sessions_by_time = filtered_summary.groupby('Business Established Date')['Total Sessions'].mean()
            avg_hours_by_time = filtered_summary.groupby('Business Established Date')['Total Hours'].mean()
            milestone_count = len(filtered_summary)
            overall_avg_sessions = filtered_summary['Total Sessions'].mean()
            overall_avg_hours = filtered_summary['Total Hours'].mean()

            add_spacing()

            st.write(f"**Number of Milestones in Selected Range:** {milestone_count}")
            st.write(f"**Overall Average Sessions:** {overall_avg_sessions:.2f}")
            st.write(f"**Overall Average Hours:** {overall_avg_hours:.2f}")

            add_spacing()

            # Plot: Average Sessions Over Time
            fig3, ax3 = plt.subplots(figsize=(15, 8))
            ax3.plot(avg_sessions_by_time.index, avg_sessions_by_time, label='Average Sessions', color='blue')
            ax3.set_title("Average Sessions Over Time", pad=20)
            ax3.set_xlabel("Business Established Date", labelpad=20)
            ax3.set_ylabel("Average Sessions", labelpad=20)
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            st.download_button("Download Sessions Plot", data=download_plot_as_png(fig3), file_name="established_sessions_plot.png")

            add_spacing()

            # Plot: Average Hours Over Time
            fig4, ax4 = plt.subplots(figsize=(15, 8))
            ax4.plot(avg_hours_by_time.index, avg_hours_by_time, label='Average Hours', color='green')
            ax4.set_title("Average Hours Over Time", pad=20)
            ax4.set_xlabel("Business Established Date", labelpad=20)
            ax4.set_ylabel("Average Hours", labelpad=20)
            ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            st.pyplot(fig4)

            st.download_button("Download Hours Plot", data=download_plot_as_png(fig4), file_name="established_hours_plot.png")

else:
    st.sidebar.write("Please upload all three datasets to proceed.")