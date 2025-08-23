# DBS-Msc-Applied-Research-2025
This repository contains the implementation of a DBS-specific chatbot developed as part of the MSc Applied Research Project (2025). The project compares Transformer (SentenceTransformers + T5 with FAISS) against traditional LSTM for text and voice-based interactions, including model development, evaluation, and user interface integration.

# Folder Strcture 

--DBS-Msc-Applied-Research-2025/
    --colab_notebooks/                                              # google colab notebooks versions 
        -- LSTM_DBS_Chatbot_2025.py
        -- LSTM_DBS_Chatbot_2025.ipynb
        -- Transformer_DBS_Chatbot_2025.py
        -- Transformer_DBS_Chatbot_2025.ipynb
        -- ResultComparison.ipynb
    --data/                                                          # brochures, test, training data files
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
            --Intent-Common.json                                                # Intent json             
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

    --documentation/                                                        # all documentations
    --results/
        --LSTM_DBS_Chatbot_2025_Results.docx
        --LSTM_DBS_Chatbot_Output_2025.txt
        --Transformer_DBS_Chatbot_2025_Results.docx
        --Transformer_DBS_Chatbot_Output_2025.txt
    --src/                                                                  # source code in python
        --Intent_Rule_Responses.py
        --LSTM_DBS_Chatbot_2025.py
        --Model_Response_Predictor.py
        --Transformer_DBS_Chatbot_2025.py
        --UI_Chatbot_Interface.py
        --Voice_IO_Logic.py
    --utilities/                                                             # setup, installations etc
        --DBSWebScraping.ipynb
        --dbswebscraping.py
        --doc_converter_1.py
        --Imports and Purpose.txt
        --install_dependencies.py

#Installation & Setup

# 1. From utilities folder run installations. 
     python  utilities/install_dependencies.py
# 2. Running transformr version of the chatbot
     python src/Transformer_DBS_Chatbot_2025.py
# 3. Running LSTM version of the chatbot
     python src/LSTM_DBS_Chatbot_2025.py
# 4. Running the chatbot user interface application. 
     python src/UI_Chatbot_Interface.py
# 5. Result comparison and hypothesis testing results
     python src/UI_Chatbot_Interface.py