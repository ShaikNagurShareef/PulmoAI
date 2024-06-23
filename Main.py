import os
from flask import Flask, render_template, request
from service.generateClinicalDescription import generate_caption
from service import generate_clinc_report_service as clinc_report_gen
from service.email_service import send_email
from werkzeug.utils import secure_filename


app = Flask(__name__)

uploads = os.path.join('static', 'uploads')
app.config['UPLOAD'] = uploads

@app.route("/")
def login():
    return render_template('index.html')

@app.route("/generateClinicalDescription", methods=['POST'])
def generateClinicalDescription():
    try:
        name = request.form['patientName']
        gender = request.form['gender']
        age = request.form['age']
        mobile = request.form['phoneNumber']
        email = request.form['email']
        address = request.form['address']
        med_history = request.form['medicalHistory']
        keyterms = request.form['diagnosticKeyterms']
        scan_type = request.form['retinalScanType']
        file = request.files['scanImageInput']
        file_name = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], file_name))
        img_path = os.path.join(app.config['UPLOAD'], file_name)

        #pre-process keywords
        keywords = keyterms
        keywords = keywords.lower()
        keywords = ",".join([kw.strip() for kw in keywords.split(',')])
        keywords = keywords.replace(',', '[sep]')

        # generate clinical description
        clinical_description = generate_caption(img_path, keywords, img_verbose=False)

        to_mail_list = ['skns.cse@gmail.com']
        if(email != None):
            to_mail_list.append(email)

        # generate report
        report_data = [
        ("Name", name),
        ("Gender", gender),
        ("Age", age),
        ("Mobile", mobile),
        ("Email", email),
        ("Address", address),
        ("Medical History", med_history),
        ("Diagnostic Attributes", keyterms),
        ("Scan Type", scan_type),
        ("Retinal Scan", img_path),
        ("Clinical Description", clinical_description)
        ]
        pdf_file = clinc_report_gen.generate_report(report_data)
        with open(pdf_file, 'rb') as f:
            report = f.read()
        send_email(to_mail_list, name, "Clinc Desc Gen", clinical_description)
        
    except Exception as e:
        print("Exception Occured: "+str(e))
        
    data = (name, gender, age, mobile, email, address, med_history, keyterms, scan_type, img_path, clinical_description)
    return render_template('report.html', data=data)
    
	
if __name__ == "__main__":
    app.run(debug=True)
