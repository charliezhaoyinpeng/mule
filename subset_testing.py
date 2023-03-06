import pickle, copy
import pandas as pd


def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_ids(l):
    ans = list()
    for item in l:
        ans.append(item['id'])
    return ans


def subset_frames(frames, gt_df, ids):
    new_frames = list()
    for frame in frames:
        video = frame["video"]
        time = frame["time"]
        actors = frame["labels"]
        actors_copy = copy.deepcopy(actors)
        temp_gt_df = gt_df.loc[(gt_df[gt_df.columns[0]] == video) & (gt_df[gt_df.columns[1]] == time) & (
                gt_df[gt_df.columns[-2]] <= max(ids))]
        temp_gt_df = temp_gt_df.reset_index(drop=True)
        # temp_gt_df = temp_gt_df.iloc[:1]
        for actor in actors_copy:
            bbx = actor["bounding_box"]
            count = 0
            for i in range(len(temp_gt_df)):
                count += 1
                comp_bbx = [temp_gt_df[temp_gt_df.columns[2]][i], temp_gt_df[temp_gt_df.columns[3]][i],
                            temp_gt_df[temp_gt_df.columns[4]][i], temp_gt_df[temp_gt_df.columns[5]][i]]
                # print(bb_iou(bbx, comp_bbx))
                if bb_iou(bbx, comp_bbx) > 0.5:
                    break
                elif count == len(temp_gt_df):
                    actors.remove(actor)
        new_frames.append(frame)
    return new_frames


def clean_frames(frames):
    new_frames = list()
    for frame in frames:
        actors = frame["labels"]
        if len(actors) != 0:
            for actor in actors:
                actor['label'] = [end_selected_cls - 1]
            new_frames.append(frame)
    return new_frames


if __name__ == '__main__':
    # path = "tools/ava_train_v2.2_person_obj_ud.pkl"
    ano_path = "annotations/ava_val_v2.2_fair_0.85.pkl"
    gt_path = "annotations/ava_val_v2.2.csv"
    start_selected_cls = 0
    end_selected_cls = 40
    save_path = "tools/ava_val_v2.2_fair_0.85_40_cls.pkl"

    ano_df = pd.read_pickle(ano_path)
    gt_df = pd.read_csv(gt_path, header=None)

    frames = ano_df[0]
    cls_info = ano_df[1][start_selected_cls:end_selected_cls]
    ids = get_ids(cls_info)

    new_frames = subset_frames(frames, gt_df, ids)
    print(len(new_frames))
    cleaned_frames = clean_frames(new_frames)
    print(len(cleaned_frames))

    ano_df[0] = frames
    ano_df[1] = cls_info
    with open(save_path, 'wb') as f:
        pickle.dump(ano_df, f)

    print("========================================================================")
    print(len(cls_info), cls_info)
    print(len(ids), ids)
    print(len(ano_df), len(ano_df[1]), ano_df[1])
    print("1: ", len(frames))
    print(frames[0])
