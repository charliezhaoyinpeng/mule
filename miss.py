import pickle
import copy


file = 'tools/ava_train_v2.2_person_obj.pkl'
import torch

pk_file = pickle.load(open(file, "rb"))
# print(len(pk_file), type(pk_file), pk_file[1])
images = pk_file[0]


def method(images):
    ans = []
    for image in images:
        all_person_ids = []
        items = image['labels']
        for item in items:
            person_ids = item['person_id']
            for id in person_ids:
                all_person_ids.append(id)
        if len(all_person_ids) == 0:
            print(image)
            ans.append(image)
        temp = list(set(all_person_ids))
        if len(temp) == 1 and temp[0] == -100:
            print(image)
            ans.append(image)
    return ans


print(len(images), len(method(images)))
temp_list = copy.deepcopy(method(images))
# print(temp_list)
for image in temp_list:
    images.remove(image)
print(len(images), len(method(images)))
pk_file[0] = images
save_name = 'ava_train_v2.2_person_obj_ud'
with open('tools/' + save_name + '.pkl', 'wb') as f:
    pickle.dump(pk_file, f)


