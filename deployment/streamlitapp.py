import pickle
import pandas as pd
import streamlit as st


model_path = "../models/model.pkl"
with open(model_path, "rb") as f:
    model, vectorizer = pickle.load(f)


def main():

    df = {}

    st.title("Attrition Prediction API")
    businesstravel = st.selectbox(
        "Travel Frequency", ("Travel_Frequently", "Travel_Rarely", "Non-Travel")
    )
    department = st.selectbox(
        "Department ", ("Research & Development", "Sales", "Human Resources")
    )
    educationfield = st.selectbox(
        "Field of study ",
        (
            "Medical",
            "Other",
            "Marketing",
            "Life Sciences",
            "Technical Degree",
            "Human Resources",
        ),
    )
    gender = st.selectbox("Gender", ("Male", "Female"))
    jobrole = st.selectbox(
        "Job Role ",
        (
            "Laboratory Technician",
            "Sales Representative",
            "Sales Executive",
            "Healthcare Representative",
            "Manager",
            "Manufacturing Director",
            "Research Scientist",
            "Human Resources",
            "Research Director",
        ),
    )
    maritalstatus = st.selectbox("Marital Status ", ("Married", "Divorced", "Single"))
    overtime = st.selectbox("Over Time Worker ", ("Yes", "No"))
    newage = st.selectbox("Age Bracket ", ("31 - 42", "43 - 60", "18 - 30"))
    masterylevel = st.selectbox("Mastery Level", ("intermediate", "master", "entry"))
    loyaltylevel = st.selectbox("Loyalty Level ", ("loyal", "fairly", "very-loyal"))
    dueforprom = st.selectbox("Promotion Due", ("overdue", "due"))
    age = st.number_input("Age")
    dailyrate = st.number_input("Dailyrate")
    distancefromhome = st.number_input("Distance from home ")
    education = st.number_input("Education")
    environmentsatisfaction = st.number_input("Environment Satisfaction")
    hourlyrate = st.number_input("Hourly Rate")
    jobinvolvement = st.number_input("Job involvement")
    joblevel = st.number_input("Job Level ")
    jobsatisfaction = st.number_input("Job Satisfaction ")
    monthlyincome = st.number_input("Monthly Income")
    monthlyrate = st.number_input("Monthly Rate ")
    numcompaniesworked = st.number_input("Number of Companies Worked ")
    percentsalaryhike = st.number_input("Percentage Salary Hike ")
    performancerating = st.number_input("Performance Rating ")
    relationshipsatisfaction = st.number_input("Relationship Satisfaction")
    stockoptionlevel = st.number_input("Stock Option Level ")
    totalworkingyears = st.number_input("Total Working years ")
    trainingtimeslastyear = st.number_input("Training Times Last Year ")
    worklifebalance = st.number_input("Work Life Balance ")
    yearsatcompany = st.number_input("Years at the Company ")
    yearsincurrentrole = st.number_input("Years in Current Role ")
    yearssincelastpromotion = st.number_input("Years Since Last Promotion")
    yearswithcurrmanager = st.number_input("Years with Surrent Manager ")

    keys = [
        "businesstravel",
        "department",
        "educationfield",
        "gender",
        "jobrole",
        "maritalstatus",
        "overtime",
        "newage",
        "masterylevel",
        "loyaltylevel",
        "dueforprom",
        "age",
        "dailyrate",
        "distancefromhome",
        "education",
        "environmentsatisfaction",
        "hourlyrate",
        "jobinvolvement",
        "joblevel",
        "jobsatisfaction",
        "monthlyincome",
        "monthlyrate",
        "numcompaniesworked",
        "percentsalaryhike",
        "performancerating",
        "relationshipsatisfaction",
        "stockoptionlevel",
        "totalworkingyears",
        "trainingtimeslastyear",
        "worklifebalance",
        "yearsatcompany",
        "yearsincurrentrole",
        "yearssincelastpromotion",
        "yearswithcurrmanager",
    ]

    values = [
        businesstravel,
        department,
        educationfield,
        gender,
        jobrole,
        maritalstatus,
        overtime,
        newage,
        masterylevel,
        loyaltylevel,
        dueforprom,
        age,
        dailyrate,
        distancefromhome,
        education,
        environmentsatisfaction,
        hourlyrate,
        jobinvolvement,
        joblevel,
        jobsatisfaction,
        monthlyincome,
        monthlyrate,
        numcompaniesworked,
        percentsalaryhike,
        performancerating,
        relationshipsatisfaction,
        stockoptionlevel,
        totalworkingyears,
        trainingtimeslastyear,
        worklifebalance,
        yearsatcompany,
        yearsincurrentrole,
        yearssincelastpromotion,
        yearswithcurrmanager,
    ]

    for i, v in zip(keys, values):
        df[i] = v

    if st.button("Predict"):
        x_values = vectorizer.transform(df)
        prediction = model.predict(x_values).round(2)
        if prediction < 0.5:
            output = f"The worker has a low attrition probability of {prediction}"
        else:
            output = f"The worker has a high atttrition probability of {prediction}"

        st.success(output)


if __name__ == "__main__":
    main()
