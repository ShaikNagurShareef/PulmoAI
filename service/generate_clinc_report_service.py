from fpdf import FPDF
from datetime import date

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('static/pulmoai-logo.png', 10, 8, 33)
        # Times bold 16
        self.set_font('Times', 'B', 16)
        self.set_text_color(255,0,0)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'PulmoAI Visual X-Rays to Clinical Insights', 0, 0, 'C')
        
        # Times bold 12
        self.set_font('Times', 'B', 12)
        self.set_text_color(0,0,0)
        # Move to the left
        self.cell(-30)
        # Tag line
        self.cell(30, 27, 'X-Ray Medical Diagnosis Reports', 0, 0, 'C')
        
        # Move to the left
        self.cell(-110)
        # Add Date & Contact details
        self.set_font('Times', '', 12)
        self.cell(0, 50, 'Date: %s' % date.today().strftime("%b-%d-%Y"), 0, 0, 'L')
        self.cell(0, 50, "Nagur Shareef Shaik & Teja Cherukuri", 0, 0, 'R')
        
        #draw line
        self.set_line_width(0.5)
        self.line(10, 40, 200, 40)
        
        # Line break
        self.ln(35)

    def add_data(self, data):

        self.set_fill_color(255, 255, 255)  # White background for the table
        self.set_text_color(0, 0, 0)  # Black text color

        # Adding Patient Details as a subheading
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, 'Patient Details', 0, 1, 'C')

        for item in data[:-2]:  # Excluding scan type and clinical description
            self.set_font('Times', 'B', 10)
            self.cell(65, 10, item[0], 1, 0, 'C', fill=True)
            self.set_font('Times', '', 10)  # Setting font to normal for right side values
            self.cell(125, 10, item[1], 1, 1, 'C', fill=True)

        self.ln(5)

        # Retinal Scan
        self.set_font('Times', 'B', 12)  # Set font to bold for the section title
        self.cell(0, 10, 'Chest X-Ray', 0, 1, 'C')
        self.set_font('Times', '', 12)  # Set font back to normal for the content
        self.image(data[9][1], x=75, y=None, w=60, h=75)

        self.ln(5)

        # Clinical Description
        self.set_font('Times', 'B', 12)  # Set font to bold for the section title
        self.cell(0, 10, 'Clinical Findings: ', 0, 1, 'L')
        self.set_font('Times', '', 12)  # Set font back to normal for the content
        self.set_text_color(0,0,255)
        self.multi_cell(0, 5, data[10][1])

        self.ln(5)

        # Set Normal Font
        self.set_font("Times", size=10)
        self.set_text_color(0,0,0)
        self.cell(0, 10,"Note: Correlate the information provided clinically, Write to 'rhd.reports@gmail.com' for any queries.", 0, 1)

def generate_report(data):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Times', '', 12)
    pdf.add_data(data)
    pdf_filename = "static/Reports/"+data[0][1]+".pdf"
    pdf.output(pdf_filename, "F")
    return pdf_filename

    
