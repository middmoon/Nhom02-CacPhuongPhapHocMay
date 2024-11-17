

from flask import Flask, render_template, request
import numpy as np
import pickle
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from form, using the updated feature names
        marital_status = int(request.form.get('marital_status'))
        application_mode = int(request.form.get('application_mode'))
        application_order = int(request.form.get('application_order'))
        course = int(request.form.get('course'))
        daytime_evening_attendance = int(request.form.get('daytime_evening_attendance'))
        previous_qualification = int(request.form.get('previous_qualification'))
        
        # Updated feature names for user input
        age = int(request.form.get('age'))
        mother_qualification = int(request.form.get('mother_qualification'))
        father_qualification = int(request.form.get('father_qualification'))
        mother_occupation = int(request.form.get('mother_occupation'))
        father_occupation = int(request.form.get('father_occupation'))
        displaced = int(request.form.get('displaced'))
        debtor = int(request.form.get('debtor'))
        tuition_fees_up_to_date = int(request.form.get('tuition_fees_up_to_date'))
        gender = int(request.form.get('gender'))
        scholarship_holder = int(request.form.get('scholarship_holder'))
        
        # Averages for the two semesters
        avg_enrolled = float(request.form.get('avg_enrolled'))
        avg_approved = float(request.form.get('avg_approved'))
        avg_grade = float(request.form.get('avg_grade'))
        avg_without_evaluations = float(request.form.get('avg_without_evaluations'))

        # Economic factors
        gdp = float(request.form.get('gdp'))

        # Organize input data in the correct order according to feature names
        input_data = np.array([[marital_status, application_mode, application_order, course,
                                daytime_evening_attendance, previous_qualification, mother_qualification, 
                                father_qualification, mother_occupation, father_occupation, displaced, 
                                debtor, tuition_fees_up_to_date, gender, scholarship_holder, age, 
                                avg_enrolled, avg_approved, avg_grade, avg_without_evaluations, gdp]])

        print("Received input data:", input_data)

        # Predict dropout probability
        dropout_probability = model.predict_proba(input_data)[0][1]
        logging.debug(f"Dropout probability: {dropout_probability}")

        # Convert to percentage
        dropout_percentage = dropout_probability * 100
        result = f'Xác suất bỏ học: {dropout_percentage:.2f}%'
        
        return render_template('results.html', result=result)
    
    except Exception as e:
        logging.error(f"Lỗi khi dự đoán: {e}")
        return render_template('results.html', result="Có lỗi xảy ra trong quá trình dự đoán.", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
