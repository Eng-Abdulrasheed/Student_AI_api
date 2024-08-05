from pycaret.classification import *
import pandas as pd

# Load the dataset
data = pd.read_csv('Students_Data/Student_performance_data _.csv')

# Set up the classification environment
clf = setup(data, target='GradeClass', session_id=123,
            numeric_features=['Age', 'StudyTimeWeekly', 'Absences'],
            categorical_features=['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring',
                                  'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'],
            ignore_features=['StudentID', 'GPA'],
            
           )

# Create and train the logistic regression model
lr_model = create_model('lr')
# Create and train the gradient boosting classifier model
gbc_model = create_model('gbc')

# Compare models
best_model = compare_models(include=['lr', 'gbc'])

# Save the best model
save_model(best_model, 'student_performance_best_model')

# Optionally, save both models
save_model(lr_model, 'student_performance_lr_model')
save_model(gbc_model, 'student_performance_gbc_model')

print('Models saved successfully')
