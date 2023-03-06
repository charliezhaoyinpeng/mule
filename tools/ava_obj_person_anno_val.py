import pandas as pd
import pickle

pkl_file = "../annotations/ava_val_v2.2_fair_0.85.pkl"
obj_anno_file = "../annotations/ava_val_v2.2_obj.csv"

pkl_dict = pd.read_pickle(pkl_file)
data = pkl_dict[0]
obj_anno_df = pd.read_csv(obj_anno_file, header=None)

count = 0
for item in data:
    count += 1
    # print(count)
    df = obj_anno_df.loc[(obj_anno_df[0] == item['video']) & (obj_anno_df[1] == item['time'])]
    # print(df)
    if not df.empty:
        for idx, row in df.iterrows():
            print("%s-%s...added" % (count, idx))
            obj_info = dict()
            obj_info['bounding_box'] = [round(row[2], 3), round(row[3], 3), round(row[4], 3), round(row[5], 3)]
            # obj_info['label'] = [row[6]]
            obj_info['label'] = [1]
            obj_info['person_id'] = [-100]
            item['labels'].append(obj_info)

with open('ava_val_v2.2_person_obj.pkl', 'wb') as f:
    pickle.dump(pkl_dict, f)
#
# print(len(pkl_dict[0]))
# print(pkl_dict[0][0])
# print(pkl_dict[0][0]['labels'])
# print(obj_anno_df.head())
# print(data[:20])

