import pickle, copy
import pandas as pd

path = "tools/ava_train_v2.2_person_obj_ud.pkl"
# path = "annotations/ava_val_v2.2_fair_0.85.pkl"
select_classes = list(range(40))

df = pd.read_pickle(path)
frames = df[0]
print("1: ", len(frames))
print(frames[0])


def print_exception_frame(frames):
    for frame in frames:
        actors = frame['labels']
        person_ids = []
        for actor in actors:
            if len(actor['label']) == 0 or len(actor['person_id']) == 0:
                print(frame)
                break
            else:
                person_ids.append(actor['person_id'][0])
        if (len(actors) == 0) or (len(list(set(person_ids))) == 1 and person_ids[0] == -100):
            print(frame)


for frame in frames:
    actors = frame['labels']
    for actor in actors:
        labels = actor['label']
        labels_copy = copy.deepcopy(labels)
        person_ids = actor['person_id']
        for label in labels_copy:
            if label not in select_classes:
                labels.remove(label)
                person_ids.remove(person_ids[0])

print("2: ", len(frames))
# print_exception_frame(frames)

temp_frames = []
for frame in frames:
    actors = frame['labels']
    temp_actors = []
    for actor in actors:
        if len(actor['label']) == 0:
            temp_actors.append(actor)
    if len(temp_actors) != 0:
        for temp_actor in temp_actors:
            actors.remove(temp_actor)
    person_ids = []
    for actor in actors:
        person_ids.append(actor['person_id'][0])
    if (len(actors) == 0) or (len(list(set(person_ids))) == 1 and person_ids[0] == -100):
        temp_frames.append(frame)
for temp_frame in temp_frames:
    frames.remove(temp_frame)

print("3: ", len(frames))
print_exception_frame(frames)


df[0] = frames
save_name = 'ava_train_v2.2_person_obj_ud_40cls'
with open('tools/' + save_name + '.pkl', 'wb') as f:
    pickle.dump(df, f)


# ans = dict()
# for frame in frames:
#     if frame['video'] not in ans.keys():
#         ans[frame['video']] = [0] * 60
#     actors = frame['labels']
#     for actor in actors:
#         if -100 not in actor['person_id']:
#             for label in actor['label']:
#                 ans[frame['video']][label] += 1
#             # print(actor['person_id'][0], actor['label'])
#
# for k, v in ans.items():
#     print(k, v)
