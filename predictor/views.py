from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle
import os
from django.conf import settings

# Load model
model_path = os.path.join(settings.BASE_DIR, 'predictor/ml_model/loan_model.pkl')
model = pickle.load(open(model_path, 'rb'))

def predict(request):

    if request.method == 'POST':

        Gender = int(request.POST['Gender'])
        Married = int(request.POST['Married'])
        Dependents = int(request.POST['Dependents'])
        Education = int(request.POST['Education'])
        Self_Employed = int(request.POST['Self_Employed'])
        Credit_History = float(request.POST['Credit_History'])
        Property_Area = int(request.POST['Property_Area'])

        LoanAmount = float(request.POST['LoanAmount'])
        Loan_Term = float(request.POST['Loan_Term'])
        Total_Income = float(request.POST['Total_Income'])

        LoanAmount_Log = np.log(LoanAmount)
        Loan_Amount_TermLog = np.log(Loan_Term)
        Total_Income_Log = np.log(Total_Income)

        features = np.array([[Gender, Married, Dependents, Education,
                              Self_Employed, Credit_History,
                              Property_Area, LoanAmount_Log,
                              Loan_Amount_TermLog, Total_Income_Log]])

        prediction = model.predict(features)

        result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"

        return render(request, 'index.html', {'result': result})

    return render(request, 'index.html')
