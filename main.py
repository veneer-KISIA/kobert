import easydict

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    
    tokenizer = load_tokenizer(args)

    train_dataset = None
    dev_dataset = None
    test_dataset = None

    if args.do_train or args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")


if __name__ == '__main__':
    args = easydict.EasyDict({
        'task':'veneer-ner',
        'model_dir':'./model',
        'data_dir':'./data',
        'pred_dir':'./preds',
        'train_file':'train.tsv',
        'test_file':'test.tsv',
        'label_file':'label.txt',
        'write_pred':'store_true',
        'model_type':'kobert',
        'model_name_or_path':'monologg/kobert',
        'seed': 42,
        'train_batch_size': 128,
        'eval_batch_size': 64,
        'max_seq_len': 50,
        'learning_rate': 5e-5,
        'num_train_epochs': 20.0,
        'weight_decay': 0.0,
        'gradient_accumulation_steps': 1,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'max_steps': -1,
        'warmup_steps': 0,
        'logging_steps': 1000,
        'save_steps': 1000,
        'do_train': True,
        'do_eval': True,
        'no_cuda': False,
    })

    main(args)


'''
Iteration: 100%|██████████| 302/302 [03:14<00:00,  1.56it/s]
Epoch: 100%|██████████| 20/20 [58:24<00:00, 175.21s/it] 
08/29/2023 18:48:05 - INFO - trainer -   ***** Model Loaded *****
08/29/2023 18:48:05 - INFO - trainer -   ***** Running evaluation on test dataset *****
08/29/2023 18:48:05 - INFO - trainer -     Num examples = 9606
08/29/2023 18:48:05 - INFO - trainer -     Batch size = 64
Evaluating: 100%|█████████| 151/151 [00:21<00:00,  7.09it/s] 
08/29/2023 18:48:27 - INFO - trainer -   ***** Eval results *****
08/29/2023 18:48:27 - INFO - trainer -     f1 = 0.9479615245301244
08/29/2023 18:48:27 - INFO - trainer -     loss = 0.03217430213629731
08/29/2023 18:48:27 - INFO - trainer -     precision = 0.9378832523691965
08/29/2023 18:48:27 - INFO - trainer -     recall = 0.9582587469487388
08/29/2023 18:48:28 - INFO - trainer -   
                       precision    recall  f1-score   support

        AFA_ART_CRAFT       0.00      0.00      0.00         1
         AFA_DOCUMENT       0.00      0.00      0.00         3
            AFA_MUSIC       0.75      0.64      0.69        14
      AFA_PERFORMANCE       0.00      0.00      0.00         2
            AFA_VIDEO       0.94      0.94      0.94       309
   AFW_OTHER_PRODUCTS       0.67      0.73      0.70        11
 AFW_SERVICE_PRODUCTS       0.00      0.00      0.00         2
          AF_BUILDING       0.40      0.57      0.47        14
    AF_CULTURAL_ASSET       0.29      0.50      0.36         4
AF_MUSICAL_INSTRUMENT       0.89      1.00      0.94         8
              AF_ROAD       0.50      0.38      0.43         8
         AF_TRANSPORT       0.93      0.95      0.94       156
            AF_WEAPON       0.85      1.00      0.92        17
          AM_AMPHIBIA       0.00      0.00      0.00         3
              AM_BIRD       0.93      0.88      0.90        16
              AM_FISH       1.00      1.00      1.00        11
            AM_INSECT       0.70      0.78      0.74         9
          AM_MAMMALIA       0.98      0.98      0.98       408
            AM_OTHERS       0.86      0.92      0.89        26
              AM_PART       0.96      0.97      0.97       359
          AM_REPTILIA       0.75      0.86      0.80         7
              AM_TYPE       0.80      0.89      0.84         9
               CV_ART       0.93      0.93      0.93       229
     CV_BUILDING_TYPE       0.00      0.00      0.00         1
          CV_CLOTHING       0.90      0.97      0.94        38
          CV_CURRENCY       0.00      0.00      0.00         1
             CV_DRINK       0.96      0.97      0.96       154
              CV_FOOD       0.97      0.98      0.97      1077
        CV_FOOD_STYLE       0.87      0.94      0.91        36
             CV_FUNDS       0.00      0.00      0.00         3
          CV_LANGUAGE       1.00      1.00      1.00        24
               CV_LAW       0.54      0.68      0.60        22
        CV_OCCUPATION       0.97      0.98      0.98       722
            CV_POLICY       0.89      1.00      0.94        42
          CV_POSITION       0.93      0.96      0.95       164
             CV_PRIZE       0.43      0.86      0.57         7
          CV_RELATION       0.96      0.97      0.96      1747
            CV_SPORTS       0.95      0.99      0.97       287
       CV_SPORTS_INST       0.55      0.75      0.63         8
   CV_SPORTS_POSITION       1.00      0.91      0.95        11
               CV_TAX       1.00      1.00      1.00         1
             CV_TRIBE       0.78      0.70      0.74        20
               DT_DAY       0.97      0.98      0.97       175
          DT_DURATION       0.95      0.96      0.95       420
             DT_MONTH       0.92      0.98      0.95        56
            DT_OTHERS       0.86      0.86      0.86        96
            DT_SEASON       0.97      0.99      0.98       508
              DT_WEEK       1.00      0.92      0.96        13
              DT_YEAR       0.96      0.97      0.97       131
          EV_FESTIVAL       0.75      0.71      0.73        21
            EV_OTHERS       0.59      0.92      0.72        25
            EV_SPORTS       0.90      0.90      0.90        40
    EV_WAR_REVOLUTION       0.00      0.00      0.00         4
               FD_ART       0.67      1.00      0.80         6
        FD_HUMANITIES       1.00      1.00      1.00        18
          FD_MEDICINE       1.00      0.17      0.29         6
            FD_OTHERS       0.00      0.00      0.00         1
           FD_SCIENCE       0.55      0.94      0.69        18
    FD_SOCIAL_SCIENCE       0.00      0.00      0.00         6
              LCG_BAY       0.00      0.00      0.00         1
        LCG_CONTINENT       1.00      1.00      1.00        47
           LCG_ISLAND       0.89      0.96      0.93        26
         LCG_MOUNTAIN       1.00      1.00      1.00        42
            LCG_OCEAN       0.00      0.00      0.00         1
            LCG_RIVER       0.83      1.00      0.91        15
      LCP_CAPITALCITY       1.00      1.00      1.00        58
             LCP_CITY       0.99      0.99      0.99       190
          LCP_COUNTRY       0.97      0.99      0.98       360
           LCP_COUNTY       0.92      0.96      0.94        76
         LCP_PROVINCE       0.98      1.00      0.99       115
            LC_OTHERS       0.87      0.93      0.90        82
             LC_SPACE       1.00      1.00      1.00        12
          MT_CHEMICAL       0.75      0.79      0.77        19
             MT_METAL       1.00      0.67      0.80         3
              OGG_ART       0.39      0.78      0.52         9
          OGG_ECONOMY       0.88      0.94      0.91        65
        OGG_EDUCATION       0.77      0.96      0.86        25
             OGG_FOOD       0.95      0.98      0.96        83
            OGG_HOTEL       0.00      0.00      0.00         5
              OGG_LAW       0.00      0.00      0.00         2
            OGG_MEDIA       0.85      0.79      0.81        14
         OGG_MEDICINE       0.50      1.00      0.67         1
         OGG_MILITARY       0.79      0.79      0.79        14
           OGG_OTHERS       0.00      0.00      0.00         7
         OGG_POLITICS       0.92      0.89      0.90        63
         OGG_RELIGION       1.00      1.00      1.00         3
          OGG_SCIENCE       0.00      0.00      0.00         1
           OGG_SPORTS       0.93      0.98      0.95        42
         PS_CHARACTER       0.65      0.76      0.70        29
              PS_NAME       0.97      0.98      0.97       424
               PS_PET       0.97      1.00      0.99        74
            PT_FLOWER       0.86      0.86      0.86        14
             PT_FRUIT       0.88      0.81      0.84        26
             PT_GRASS       0.67      0.80      0.73        15
            PT_OTHERS       1.00      1.00      1.00         1
              PT_PART       0.94      1.00      0.97        32
              PT_TREE       0.00      0.00      0.00         2
               QT_AGE       0.94      0.95      0.95       234
             QT_COUNT       0.93      0.95      0.94       533
            QT_LENGTH       0.43      0.50      0.46         6
         QT_MAN_COUNT       0.95      0.98      0.96       232
             QT_ORDER       0.95      0.96      0.96       242
            QT_OTHERS       0.68      0.75      0.71        20
        QT_PERCENTAGE       0.91      0.98      0.95        54
             QT_PRICE       0.97      0.98      0.97       132
            QT_SPORTS       1.00      0.90      0.95        10
       QT_TEMPERATURE       0.50      0.75      0.60         4
            QT_VOLUME       0.00      0.00      0.00         1
            QT_WEIGHT       0.88      0.92      0.90        24
          TI_DURATION       0.94      0.98      0.96       233
              TI_HOUR       0.86      0.86      0.86        21
            TI_OTHERS       0.74      0.88      0.80        16
           TMIG_GENRE       0.00      0.00      0.00         2
               TMI_HW       0.94      0.97      0.95       162
          TMI_SERVICE       0.85      0.93      0.89       127
               TMI_SW       0.80      0.80      0.80         5
          TMM_DISEASE       0.95      0.97      0.96       358
             TMM_DRUG       0.93      0.96      0.95        74
 TM_CELL_TISSUE_ORGAN       0.93      0.93      0.93        69
             TM_COLOR       0.85      1.00      0.92        17
         TM_DIRECTION       0.88      0.96      0.92       126
             TM_SHAPE       0.00      0.00      0.00         1
            TM_SPORTS       0.65      0.76      0.70        17
               TR_ART       0.00      0.00      0.00         1
          TR_MEDICINE       0.70      0.82      0.76        17
            TR_OTHERS       0.00      0.00      0.00         1
           TR_SCIENCE       0.67      0.40      0.50         5
    TR_SOCIAL_SCIENCE       0.75      0.92      0.83        13

            micro avg       0.94      0.96      0.95     12290
            macro avg       0.69      0.73      0.70     12290
         weighted avg       0.94      0.96      0.95     12290
'''