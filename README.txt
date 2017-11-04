Kevin de Haan, 2017

Usage instructions:


If required dependencies are installed (on any system), simply run 'python3 main.py':
    -NumPy
    -SciPy
    -SciKit-learn

On linux, with python >= 3.3 installed:
    extract files to appropriate file location
    navigate to file location
    type 'venv/bin/python3.5 main.py' to create a model
    type 'venv/bin/python3.5 model.py' to test a model

    
On Windows, with python >= 3.3 installed:
    extract files to appropriate file location
    navigate to file location
    type '.\win_env\Scripts\activate'
    this will run the appropriate file for cmd or powershell, whichever you are using
    with the virtual environment active, type 'python .\main.py' to create a model
    type '.\model.py' to test a model

    to exit the virtual environment, type 'deactivate'

    if you encounter a script permission error, it may be bypassed by changing your system settings in powershell:
	'Set-ExecutionPolicy Unrestricted'
    which you can reset afterwards.
    
    
Description of files:
    main.py: model creation and training
    model.py: model testing
    data.csv: provided data
    requirements.txt: exact versions of modules
    results.csv: results of data testing created immediately after model creation
    sample_results.csv: an example of a typical results.csv output
    model_results.csv: results of model testing when target results are provided
    sample_model_results.csv: model_results when run with sample_model.pkl
    predictions.csv: results of model testing when target results are unknown
    sample_predictions.csv: predictions when run with sample_model.pkl
    model.pkl: created when main.py is run
    sample_model.pkl: one of the better models created from main.py
    sample_tests.csv: an example csv with the first column (actual outputs) removed
    
 
    

    

