config = {'luna_raw': 'luna/dataset/raw/', # luna raw data path         
          'luna_segment': 'luna/dataset/seg-lungs-LUNA16/',   # luna segment path
          'luna_data': 'luna/dataset/allset',  # luna allset
          'preprocess_result_path': 'prep_luna_full', # preprocessed data dir

          'luna_abbr': './labels/shorter.csv',
          'luna_label': './labels/annos.csv',
          'stage1_annos_path': ['./labels/label_job5.csv',
                                './labels/label_job4_2.csv',
                                './labels/label_job4_1.csv',
                                './labels/label_job0.csv',
                                './labels/label_qualified.csv'],
          
          'preprocessing_backend': 'python'
          }
