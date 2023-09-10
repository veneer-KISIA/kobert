import argparse
from predict2 import predict

def get_predict(text):

    pred_config = argparse.Namespace()
    predict_labels = ['LCP_COUNTY', 'OGG_EDUCATION', 'OGG_MEDICINE','PS_NAME', 'PS_PET', 'QT_AGE', 'TMM_DISEASE', 'TMM_DRUG']

    pred_config.input_text = text
    pred_config.model_dir = './model'
    pred_config.batch_size = 32
    pred_config.no_cuda = True
    pred_config.predict_labels = predict_labels

    return predict(pred_config)

if __name__ == "__main__":
    print(get_predict('안녕하세요 단국대학교 학생 홍길동입니다.'))
