 # Predictive Maintainer Chatbot & Main Application

 This repository provides two main Streamlit applications:

 1. **Chatbot Application**
 2. **Main Predictive Maintenance Application**

 ## Prerequisites

 - Python 3.8 or higher
 - [Streamlit](https://streamlit.io/) installed
 - All required Python packages (see below)

 ## Setup Instructions

 1. **Clone the Repository**
   ```powershell
   git clone https://github.com/saivarunkonda/Predictive_maintainer_chatbot.git
   cd Predictive_Maintainer_chatbot
   ```

 2. **Install Required Packages**
   It is recommended to use a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

 ## Running the Applications

 ### 1. Run the Chatbot

 To start the chatbot application, use the following command:

 ```powershell
 streamlit run streamlit_synthetic_generator.py
 ```

 This will launch the chatbot interface in your browser.

 ### 2. Run the Main Application

 To start the main predictive maintenance application, use:

 ```powershell
 streamlit run app.py
 ```

 This will launch the main dashboard for predictive maintenance.

 ## Additional Notes

 - Ensure all dependencies listed in `requirements.txt` are installed.
 - If you encounter issues, check your Python and Streamlit installation.
 - For custom data or document support, place your files in the appropriate folders (`test_documents/`, `test_files/`, etc.).
 - For database or advanced features, review the code and documentation for additional setup.

 ## File Structure

 - `app.py` - Main application entry point
 - `streamlit_synthetic_generator.py` - Chatbot application entry point
 - `requirements.txt` - Python dependencies
 - `assets/`, `pages/`, `test_documents/`, etc. - Supporting files and data

 ## License

 See `LICENSE` for details.

 ## Support

 For issues or questions, please open an issue in the repository.