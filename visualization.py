import pickle, cv2
import pandas as pd

video = '7T5G0CmwTPo'
time = '1455'

predict_file = '/home/local/KHQ/chen.zhao/ACAR-Net-copy/experiments/AVA/SLOWFAST_R50_ACAR_HR2O/predict_epoch1.csv'
ground_truth = 'annotations/ava_val_v2.2.csv'
ava_data_path = '/data/datasets/AVA/frames/' + video + '/' + 'image_' + f"{int(time):06}" + '.jpg'


def select(infile, video, time):
    df = pd.read_csv(infile, header=None)
    df.columns = ['video', 'time', 'bbx1', 'bbx2', 'bbx3', 'bbx4', 'label', 'precision']
    df = df[(df['video'] == video) & (df['time'] == int(time)) & (df['precision'] > 0.5)]
    return df


def draw_bbx(df, image, color):
    h, w, c = image.shape
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    bbxes = []
    ind = 0
    for index, row in df.iterrows():
        bbx = [row['bbx1'], row['bbx2'], row['bbx3'], row['bbx4']]
        label = row['label']
        if bbx not in bbxes:
            bbxes.append(bbx)
            start_point = (int(row['bbx1'] * w), int(row['bbx2'] * h))
            end_point = (int(row['bbx3'] * w), int(row['bbx4'] * h))
            image = cv2.rectangle(image, start_point, end_point, color, thickness=2)
            cv2.putText(image, '(' + str(label) + ')', (start_point[0], start_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, thickness=1)
        else:
            ind += 1
            cv2.putText(image, '(' + str(label) + ')', (start_point[0] + ind * 20, start_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, thickness=1)
    return image


df_pre = select(predict_file, video, time)
df_gt = select(ground_truth, video, time)
print(df_pre)
print(df_gt)
print(ava_data_path)
image = cv2.imread(ava_data_path)
image = draw_bbx(df_gt, image, (0, 255, 0))  # green
image = draw_bbx(df_pre, image, (0, 0, 255))  # red
cv2.imwrite('visualizations/' + video + '_' + time + '.jpg', image)
