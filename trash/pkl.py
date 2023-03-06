import pickle
import copy

# file = '../tools/ava_train_v2.2_person_obj_ud_40cls.pkl'
file = '../tools/ava_val_v2.2_fair_0.85_40_ood.pkl'

pk_file = pickle.load(open(file, "rb"))
# print(len(pk_file), type(pk_file), pk_file[1])
images = pk_file[0]

new_images = images[:1000]
print(len(new_images))

pk_file[0] = new_images
save_name = 'val_test'
with open(save_name + '.pkl', 'wb') as f:
    pickle.dump(pk_file, f)