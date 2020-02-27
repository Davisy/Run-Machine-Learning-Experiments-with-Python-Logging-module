from imports import * 

def preprocessing(data):

	# replace with numerical values
	data['Dependents'].replace('3+', 3,inplace=True)
	data['Loan_Status'].replace('N', 0,inplace=True)
	data['Loan_Status'].replace('Y', 1,inplace=True)

	# handle missing data 
	data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
	data['Married'].fillna(data['Married'].mode()[0], inplace=True)
	data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
	data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
	data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
	data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
	data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)

	# drop ID column
	data = data.drop('Loan_ID',axis=1)

	#scale the data
	data["ApplicantIncome"] = MinMaxScaler().fit_transform(data["ApplicantIncome"].values.reshape(-1,1))
	data["LoanAmount"] = MinMaxScaler().fit_transform(data["LoanAmount"].values.reshape(-1,1))
	data["CoapplicantIncome"] = MinMaxScaler().fit_transform(data["CoapplicantIncome"].values.reshape(-1,1))
	data["Loan_Amount_Term"] = MinMaxScaler().fit_transform(data["Loan_Amount_Term"].values.reshape(-1,1))

	#change categorical features to numerical
	data = pd.get_dummies(data)

	return data 