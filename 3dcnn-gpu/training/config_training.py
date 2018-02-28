config = {#'stage1_data_path': '/home/wuhui/3dcnn/cd/kaggle/stage1/',
          #'luna_raw': '/home/wuhui/sda/luna/dataset/raw/',
         
          #'luna_segment': '/home/wuhui/sda/luna/dataset/seg-lungs-LUNA16/',
          #'luna_data': '/home/wuhui/sda/luna/dataset/allset',
          'preprocess_result_path': '/home/wuhui/3dcnn/result-wh/prep_luna_full_075/',

          'luna_abbr': './detector/labels/shorter.csv',
          'luna_label': './detector/labels/annos.csv',
          'stage1_annos_path': ['./detector/labels/label_job5.csv',
                                './detector/labels/label_job4_2.csv',
                                './detector/labels/label_job4_1.csv',
                                './detector/labels/label_job0.csv',
                                './detector/labels/label_qualified.csv'],
          'bbox_path': '/home/wuhui/3dcnn/lung-detection-grt/training/detector/results/res18/bbox/',
          'preprocessing_backend': 'python'
          }
