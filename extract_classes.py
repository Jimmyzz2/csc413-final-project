class_id_to_index = {}
lst_classes = []

with open('map_vid.txt', 'r') as f:
    classes = f.readlines()
    for cls in classes:
        cls = cls.split(' ')
        class_id_to_index[cls[0]] = int(cls[1])-1
        lst_classes.append(cls[2][:-1])

print(class_id_to_index)
print(lst_classes)