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


def add_labels(frames, gt_df, ids):
    new_frames = list()
    for frame in frames:
        video = frame["video"]
        time = frame["time"]
        actors = frame["labels"]
        temp_gt_df = gt_df.loc[(gt_df[gt_df.columns[0]] == video) & (gt_df[gt_df.columns[1]] == time) & (
                gt_df[gt_df.columns[-2]] >= min(ids))]
        temp_gt_df = temp_gt_df.reset_index(drop=True)
        # temp_gt_df = temp_gt_df.iloc[:1]
        # print(len(temp_gt_df), temp_gt_df)
        for actor in actors:
            bbx = actor["bounding_box"]
            actor["flag"] = False
            count = 0
            for i in range(len(temp_gt_df)):
                count += 1
                comp_bbx = [temp_gt_df[temp_gt_df.columns[2]][i], temp_gt_df[temp_gt_df.columns[3]][i],
                            temp_gt_df[temp_gt_df.columns[4]][i], temp_gt_df[temp_gt_df.columns[5]][i]]
                # print(bb_iou(bbx, comp_bbx))
                gt_label = temp_gt_df[temp_gt_df.columns[-2]][i]
                if (bb_iou(bbx, comp_bbx) > 0.5) and (gt_label in ids):
                    actor["label"].append(gt_label)
                    actor["flag"] = True
        new_frames.append(frame)
    return new_frames


def subset_frames(frames):
    new_frames = list()
    for frame in frames:
        actors = frame["labels"]
        actors_copy = copy.deepcopy(actors)
        for actor in actors_copy:
            if actor["flag"] == False:
                actors.remove(actor)
        if len(actors) != 0:
            new_frames.append(frame)
    return new_frames


def flag_by_kuk(frames):
    new_frames = list()
    k_cls = list(range(start_selected_cls, k_unk_cut))
    unk_cls = list(range(k_unk_cut, end_selected_cls))
    for frame in frames:
        actors = frame["labels"]
        for actor in actors:
            num_actor_labels = len(actor["label"])
            actor["person_id"] = actor["person_id"] * num_actor_labels
            condition1 = all(x in k_cls for x in actor["label"])
            condition2 = all(x in unk_cls for x in actor["label"])
            if condition1:
                actor["uncertainty"] = 0
            elif condition2:
                actor["uncertainty"] = 1
            else:
                actor["uncertainty"] = -1
                actor["flag"] = False
                # print("!---> ", actor)
        new_frames.append(frame)
    return new_frames


def clean_frames(frames, ids, labels):
    id_label = dict()
    for i in range(len(ids)):
        id_label[ids[i]] = labels[i]
    new_frames = list()
    for frame in frames:
        actors = frame["labels"]
        for actor in actors:
            temp = actor["label"][1:]
            actor["label"] = temp
            for i in range(len(actor["label"])):
                actor["label"][i] = id_label[actor["label"][i]]
        new_frames.append(frame)
    return new_frames


def print_check(frames, num):
    for i in range(num):
        print(frames[i])


def check(frames):
    uncertainty0, uncertainty1, uncertainty_others = 0, 0, 0
    for frame in frames:
        actors = frame["labels"]
        for actor in actors:
            if actor["uncertainty"] == 1:
                uncertainty1 += 1
            elif actor["uncertainty"] == 0:
                uncertainty0 += 1
            else:
                uncertainty_others += 1

    print("**************************************")
    print("# of frames: ", len(frames))
    print("# of known actors: ", uncertainty0)
    print("# of unknown actors: ", uncertainty1)
    print("# of error actors: ", uncertainty_others)
    print("**************************************")


if __name__ == '__main__':
    # path = "tools/ava_train_v2.2_person_obj_ud.pkl"
    ano_path = "annotations/ava_val_v2.2_fair_0.85.pkl"
    gt_path = "annotations/ava_val_v2.2.csv"
    start_selected_cls = 20
    end_selected_cls = 60
    k_unk_cut = 40
    save_path = "tools/ava_val_v2.2_fair_0.85_40_ood.pkl"

    ano_df = pd.read_pickle(ano_path)
    gt_df = pd.read_csv(gt_path, header=None)

    frames = ano_df[0]
    cls_info = ano_df[1][start_selected_cls:end_selected_cls]
    ids = get_ids(cls_info)
    labels = list(range(start_selected_cls, end_selected_cls))

    print(len(ids), "ids: ", ids)
    print(len(labels), "labels: ", labels)

    frames_v1 = add_labels(frames, gt_df, ids)
    print("frames_v1: ", len(frames_v1))
    # print_check(frames_v1, 100)
    frames_v2 = subset_frames(frames_v1)
    print("frames_v2: ", len(frames_v2))
    # print_check(frames_v2, 100)
    frames_v3 = clean_frames(frames_v2, ids, labels)
    print("frames_v3: ", len(frames_v3))
    # print_check(frames_v3, 100)
    frames_v4 = subset_frames(flag_by_kuk(frames_v3))
    print("frames_v4: ", len(frames_v4))
    # print_check(frames_v4, 100)


    check(frames_v4)

    ano_df[0] = frames_v4
    ano_df[1] = cls_info
    with open(save_path, 'wb') as f:
        print("saving...")
        pickle.dump(ano_df, f)

    print("========================================================================")
    print(len(cls_info), cls_info)
    print(len(ids), ids)
    print(len(ano_df), len(ano_df[1]), ano_df[1])
    print("1: ", len(frames))
    print(frames[0])
