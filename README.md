# DBS-Msc-Applied-Research-2025
This repository contains the implementation of a DBS-specific chatbot developed as part of the MSc Applied Research Project (2025). The project compares Transformer (SentenceTransformers + T5 with FAISS) against traditional LSTM for text and voice-based interactions, including model development, evaluation, and user interface integration.

# ################# Installation & Setup #########################

# Option 1 - Manual

# 1.1 Copy the folder contents directly into the C drive. 
# 1.2 run installation file, It will install alll the livraries required in all the files.
        utilities/install_dependencies.py
# 1.3 To run the Transformer version of the chatbot use command 
        python src/Transformer_DBS_Chatbot_2025.py
# 1.4 To run the LSTM version of the chatbot use command 
        python src/LSTM_DBS_Chatbot_2025.py
# 1.5 To run the the chatbot user interface application use command . 
        python src/UI_Chatbot_Interface.py
# 1.6 To run the the Result comparison and hypothesis testing results
        python src/ResultComparison.py

# Option 2 - Auto

# 2.1 Open the installation file below and set create_dir_and_uploaf_files_flag = True. It will 
#      first install the libraries, then create necessary folders, and finally upload files from github

        p ython  utilities/install_dependencies.py     

# 2.2 To run the Transformer version of the chatbot use command 
        python src/Transformer_DBS_Chatbot_2025.py
# 2.3 To run the LSTM version of the chatbot use command 
        python src/LSTM_DBS_Chatbot_2025.py
# 2.4 To run the the chatbot user interface application use command . 
        python src/UI_Chatbot_Interface.py
# 2.5 To run the the Result comparison and hypothesis testing results
        python src/ResultComparison.py

# ################# Folder Structure #########################

--DBS-Msc-Applied-Research-2025/
    --README.md
    --colab_notebooks/                          # google colab notebooks versions 
        -- LSTM_DBS_Chatbot_2025.py
        -- LSTM_DBS_Chatbot_2025.ipynb
        -- Transformer_DBS_Chatbot_2025.py
        -- Transformer_DBS_Chatbot_2025.ipynb
        -- ResultComparison.ipynb
    --content                                   # Files n str required for direct copy paste         
    --data/                                     # brochures, test, training data files
        -- brochures-data-collected/
            -- 1-Dbs-postgraduate-programmes.pdf
            -- 2-FAQs-DBS.pdf
            -- 3-LTA-Newsletter-Issue.pdf
            -- 4-dbs-fees-2024-2025.pdf
            -- 5-dbs-college-handbook-2021-2022.pdf
            -- 6-dbs-fee-sheet-international-2024-2025.pdf
            -- 7-Quality.pdf
            -- 8-Pre-Arrival FAQ Guide for new Students to DBS.pdf
            -- Brochures1-8_combined-mdtext-Cleaned-final.txt
        -- training-testing-data-files/
            -- LSTM_Testing_DataSet.json
            -- LSTM_Training_DataSet.json
            -- Transformer_Test_DataSet.json
            -- Transformer_Training_DataSet-1.txt
            -- Transformer_Training_DataSet-2.txt
            -- Intent-Common.json                 # Intent files
        -- web-scraped-data-collected/
            -- 1. why_dbs_scraped_dbs.txt
            -- 2. contact_us_dbs_scraped_dbs.txt
            -- 3. httpswww.dbs.iedbs-staff.txt
            -- 4. httpswww.dbs.iepostgraduate.txt
            -- 5. lib_dbs_scraped_dbs.json
            -- 6. lib_dbs_scraped_dbs.txt
            -- 7. news_dbs_scraped_dbs.txt
            -- 8. stud_exp_dbs_scraped_dbs.txt
            -- ScrapedData-Cleaned.txt

    --documentation/
            -- Sandeep_20049275_Presentation.pptx                          # all documentations
            -- Sandeep_20049275_Report.docx
            -- Sandeep_20049275_Report.pdf
    --results/
        --LSTM_DBS_Chatbot_2025_Results.docx
        --LSTM_DBS_Chatbot_Output_2025.txt
        --Transformer_DBS_Chatbot_2025_Results.docx
        --Transformer_DBS_Chatbot_Output_2025.txt
    --src/                                                                  # source code in python
        --Intent_Rule_Responses.py
        --LSTM_DBS_Chatbot_2025.py
        --Model_Response_Predictor.py
        --ResultComparison.py
        --Transformer_DBS_Chatbot_2025.py
        --UI_Chatbot_Interface.py
        --Voice_IO_Logic.py
    --utilities/                                                             # setup, installations etc
        --DBSWebScraping.ipynb
        --dbswebscraping.py
        --doc_converter_1.py
        --Imports and Purpose.txt
        --install_dependencies.py

