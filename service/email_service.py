#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:16:37 2023

@author: reshma shaik
"""

# Python code to illustrate Sending mail with attachments
# from your Gmail account

# libraries to be imported
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import COMMASPACE, formatdate

def send_email(to_addr, name, test, result):

    from_addr = "rhd.reports@gmail.com"
    report_path = "static/Reports/"+name+".pdf"

    # instance of MIMEMultipart
    msg = MIMEMultipart()
    
    # storing the senders email address
    msg['From'] = from_addr
    
    # storing the receivers email address
    msg['To'] = COMMASPACE.join(to_addr)
    
    # message date
    msg['Date'] = formatdate(localtime=True)
    
    # storing the subject
    msg['Subject'] = "RHD Report of "+name+" for - "+ test
    
    # string to store the body of the mail
    body = 'Hello,\n\nDiagnosis report of '+name+' for '+test+' is ready. Please find the attached report and diagnosis result. \n\nDiagnosis Result: '+result+'\n\nCorrelate the information clinically. \n\nNote: Write to rhd.reports@gmail.com for any queries \n\nThanks & Regards,\nPulmoAI Health Diagnostics.'  
    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))
    
    # open the file to be sent
    attachment = open(report_path, "rb")
    
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')
    
    # To change the payload into encoded form
    p.set_payload((attachment).read())
    
    # encode into base64
    encoders.encode_base64(p)
    
    p.add_header('Content-Disposition', "attachment; filename= %s" % name+"_"+formatdate(localtime=True)+".pdf")
    
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)
    
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    
    # start TLS for security
    s.starttls()
    
    # Authentication
    s.login(from_addr, "agndnhbontjsahyd")
    
    # Converts the Multipart msg into a string
    text = msg.as_string()
    
    # sending the mail
    s.sendmail(from_addr, to_addr, text)
    
    # terminating the session
    s.quit()
