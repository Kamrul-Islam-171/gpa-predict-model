# import gradio as gr
# import pandas as pd
# import pickle
# import numpy as np

# with open('student_rf_pipeline.pkl', 'rb') as f:
#     model = pickle.load(f)

# def predict_gpa(gender, age, address, famsize, 
#                 Pstatus, M_Edu, F_Edu, M_Job, F_Job, 
#                 relationship, smoker, tuition_fee, time_friends,
#                   ssc_result):
    
#     input_df = pd.DataFrame([[
#          gender, age, address, famsize, Pstatus, 
#         M_Edu, F_Edu, M_Job, F_Job, relationship, 
#         smoker, tuition_fee, time_friends, ssc_result
#     ]], columns=[
#          'gender', 'age', 'address', 'famsize', 'Pstatus', 'M_Edu', 'F_Edu', 'M_Job', 'F_Job', 'relationship', 'smoker', 'tuition_fee', 'time_friends', 'ssc_result'
#     ])  
    
#     prediction = model.predict(input_df)[0]
#     return f"Predicted HSC Result: {np.clip(prediction, 0, 5):.2f}"
# inputs = [
#     gr.Radio(["M", "F"], label="Gender"),
#     gr.Number(label="Age", value=18),
#     gr.Radio(["Urban", "Rural"], label="Address"),
#     gr.Radio(["GT3", "LE3"], label="Family Size"),
#     gr.Radio(["Together", "Apart"], label="Parent Status"),
#     gr.Slider(0, 4, step=1, label="Mother's Edu"),
#     gr.Slider(0, 4, step=1, label="Father's Edu"),
#     gr.Dropdown(["At_home", "Health", "Other", "Services", "Teacher"], label="Mother's Job"),
#     gr.Dropdown(["Teacher", "Other", "Services", "Health", "Business", "Farmer"], label="Father's Job"),
#     gr.Radio(["Yes", "No"], label="Relationship"),
#     gr.Radio(["Yes", "No"], label="Smoker"),
#     gr.Number(label="Tuition Fee"),
#     gr.Slider(1, 5, step=1, label="Time with Friends"),
#     gr.Number(label="SSC Result (GPA)")
# ]

# app = gr.Interface(
#     fn=predict_gpa,
#       inputs=inputs,
#         outputs="text", 
#         title="HSC Predictor")

# app.launch(share=True)

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load model
with open('student_rf_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Custom CSS for styling
custom_css = """
/* Main container styling */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header styling */
.header-text {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
}

.header-text h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-text p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Section cards */
.section-card {
    background: linear-gradient(145deg, #ffffff, #f0f0f0);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}

/* Button styling */
.predict-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-size: 1.2rem !important;
    padding: 15px 40px !important;
    border-radius: 25px !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    transition: transform 0.3s, box-shadow 0.3s !important;
}

.predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
}

/* Output box styling */
.output-box {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    color: white;
    font-size: 1.5rem;
    font-weight: bold;
    box-shadow: 0 4px 20px rgba(17, 153, 142, 0.3);
}

/* Input styling */
input, select, .gr-dropdown {
    border-radius: 10px !important;
    border: 2px solid #e0e0e0 !important;
    transition: border-color 0.3s !important;
}

input:focus, select:focus {
    border-color: #667eea !important;
}

/* Radio button styling */
.gr-radio {
    background: #f8f9fa !important;
    padding: 10px !important;
    border-radius: 10px !important;
}

/* Slider styling */
.gr-slider {
    accent-color: #667eea;
}

/* Section titles */
.section-title {
    color: #667eea;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 0.9rem;
}
"""

def predict_gpa(gender, age, address, famsize, 
                Pstatus, M_Edu, F_Edu, M_Job, F_Job, 
                relationship, smoker, tuition_fee, time_friends,
                ssc_result):
    
    # Validation
    if None in [gender, address, famsize, Pstatus, M_Job, F_Job, relationship, smoker]:
        return """
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;">
            <h3>⚠️ Missing Information</h3>
            <p>Please fill in all required fields</p>
        </div>
        """
    
    input_df = pd.DataFrame([[
        gender, age, address, famsize, Pstatus, 
        M_Edu, F_Edu, M_Job, F_Job, relationship, 
        smoker, tuition_fee, time_friends, ssc_result
    ]], columns=[
        'gender', 'age', 'address', 'famsize', 'Pstatus', 
        'M_Edu', 'F_Edu', 'M_Job', 'F_Job', 'relationship', 
        'smoker', 'tuition_fee', 'time_friends', 'ssc_result'
    ])  
    
    prediction = model.predict(input_df)[0]
    result = np.clip(prediction, 0, 5)
    
    # Determine performance level and color
    if result >= 4.5:
        grade = "A+"
        color = "#00b894"
        emoji = "🏆"
        message = "Outstanding Performance!"
    elif result >= 4.0:
        grade = "A"
        color = "#00cec9"
        emoji = "🌟"
        message = "Excellent Performance!"
    elif result >= 3.5:
        grade = "A-"
        color = "#0984e3"
        emoji = "✨"
        message = "Very Good Performance!"
    elif result >= 3.0:
        grade = "B"
        color = "#6c5ce7"
        emoji = "👍"
        message = "Good Performance!"
    elif result >= 2.5:
        grade = "C"
        color = "#fdcb6e"
        emoji = "📚"
        message = "Average Performance"
    else:
        grade = "D"
        color = "#e17055"
        emoji = "💪"
        message = "Keep Working Hard!"
    
    return f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                padding: 30px; border-radius: 20px; text-align: center; color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <div style="font-size: 3rem; margin-bottom: 10px;">{emoji}</div>
        <h2 style="font-size: 2.5rem; margin: 10px 0;">Predicted GPA: {result:.2f}</h2>
        <div style="font-size: 1.5rem; opacity: 0.9;">Grade: {grade}</div>
        <p style="margin-top: 15px; font-size: 1.1rem;">{message}</p>
    </div>
    """

# Build the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    
    # Header
    gr.HTML("""
        <div class="header-text">
            <h1>🎓 HSC Result Predictor</h1>
            <p>Predict your Higher Secondary Certificate (HSC) result using AI-powered analysis</p>
        </div>
    """)
    
    with gr.Row():
        # Left Column - Inputs
        with gr.Column(scale=2):
            
            # Personal Information Section
            gr.HTML('<div class="section-title">👤 Personal Information</div>')
            with gr.Group():
                with gr.Row():
                    gender = gr.Radio(
                        choices=["M", "F"], 
                        label="Gender",
                        info="Select your gender"
                    )
                    age = gr.Number(
                        label="Age", 
                        value=18,
                        info="Enter your age"
                    )
                with gr.Row():
                    address = gr.Radio(
                        choices=["Urban", "Rural"], 
                        label="Address Type",
                        info="Where do you live?"
                    )
            
            # Family Information Section
            gr.HTML('<div class="section-title">👨‍👩‍👧 Family Information</div>')
            with gr.Group():
                with gr.Row():
                    famsize = gr.Radio(
                        choices=["GT3", "LE3"], 
                        label="Family Size",
                        info="GT3: >3 members, LE3: ≤3 members"
                    )
                    Pstatus = gr.Radio(
                        choices=["Together", "Apart"], 
                        label="Parent Status",
                        info="Are parents living together?"
                    )
            
            # Parents' Education & Job Section
            gr.HTML('<div class="section-title">🎓 Parents\' Education & Occupation</div>')
            with gr.Group():
                with gr.Row():
                    M_Edu = gr.Slider(
                        minimum=0, maximum=4, step=1, 
                        label="Mother's Education Level",
                        info="0: None → 4: Higher Education"
                    )
                    F_Edu = gr.Slider(
                        minimum=0, maximum=4, step=1, 
                        label="Father's Education Level",
                        info="0: None → 4: Higher Education"
                    )
                with gr.Row():
                    M_Job = gr.Dropdown(
                        choices=["At_home", "Health", "Other", "Services", "Teacher"],
                        label="Mother's Job",
                        info="Select mother's occupation"
                    )
                    F_Job = gr.Dropdown(
                        choices=["Teacher", "Other", "Services", "Health", "Business", "Farmer"],
                        label="Father's Job",
                        info="Select father's occupation"
                    )
            
            # Lifestyle Section
            gr.HTML('<div class="section-title">🌟 Lifestyle & Habits</div>')
            with gr.Group():
                with gr.Row():
                    relationship = gr.Radio(
                        choices=["Yes", "No"], 
                        label="In a Relationship?",
                        info="Are you in a romantic relationship?"
                    )
                    smoker = gr.Radio(
                        choices=["Yes", "No"], 
                        label="Smoker?",
                        info="Do you smoke?"
                    )
                with gr.Row():
                    time_friends = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Time Spent with Friends",
                        info="1: Very Low → 5: Very High"
                    )
            
            # Academic Information Section
            gr.HTML('<div class="section-title">📚 Academic Information</div>')
            with gr.Group():
                with gr.Row():
                    tuition_fee = gr.Number(
                        label="Monthly Tuition Fee (৳)",
                        value=0,
                        info="Enter monthly tuition fee amount"
                    )
                    ssc_result = gr.Number(
                        label="SSC Result (GPA)",
                        value=4.0,
                        info="Enter your SSC GPA (0-5)"
                    )
            
            # Submit Button
            predict_btn = gr.Button(
                "🔮 Predict HSC Result", 
                variant="primary",
                elem_classes=["predict-btn"]
            )
        
        # Right Column - Output
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title">📊 Prediction Result</div>')
            output = gr.HTML(
                value="""
                <div style="background: linear-gradient(145deg, #f0f0f0, #ffffff); 
                            padding: 40px; border-radius: 20px; text-align: center;
                            border: 2px dashed #ccc;">
                    <div style="font-size: 3rem; margin-bottom: 15px;">🎯</div>
                    <p style="color: #666; font-size: 1.1rem;">
                        Fill in your information and click<br>"Predict HSC Result" to see your prediction
                    </p>
                </div>
                """
            )
            
            # Tips Section
            gr.HTML("""
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                            padding: 20px; border-radius: 15px; margin-top: 20px;">
                    <h4 style="color: #333; margin-bottom: 10px;">💡 Tips for Better Results</h4>
                    <ul style="color: #555; font-size: 0.9rem; padding-left: 20px;">
                        <li>Focus on your studies regularly</li>
                        <li>Maintain a healthy lifestyle</li>
                        <li>Balance social and academic life</li>
                        <li>Seek help when needed</li>
                    </ul>
                </div>
            """)
    
    # Examples Section
    gr.HTML('<div class="section-title">📋 Example Inputs</div>')
    gr.Examples(
        examples=[
            ["M", 17, "Urban", "GT3", "Together", 3, 4, "Teacher", "Business", "No", "No", 2000, 3, 4.5],
            ["F", 18, "Rural", "LE3", "Together", 2, 2, "At_home", "Farmer", "No", "No", 500, 2, 3.8],
            ["M", 19, "Urban", "GT3", "Apart", 4, 3, "Health", "Services", "Yes", "No", 3000, 4, 4.2],
        ],
        inputs=[gender, age, address, famsize, Pstatus, M_Edu, F_Edu, M_Job, F_Job, relationship, smoker, tuition_fee, time_friends, ssc_result],
    )
    
    # Footer
    gr.HTML("""
        <div class="footer">
            <p>🎓 HSC Result Predictor | Built with ❤️ using Gradio & Machine Learning</p>
            <p style="font-size: 0.8rem; color: #999;">
                Note: This prediction is based on statistical analysis and should be used for guidance only.
            </p>
        </div>
    """)
    
    # Connect the button to the function
    predict_btn.click(
        fn=predict_gpa,
        inputs=[gender, age, address, famsize, Pstatus, M_Edu, F_Edu, M_Job, F_Job, relationship, smoker, tuition_fee, time_friends, ssc_result],
        outputs=output
    )

# Launch the app
app.launch(share=True)