import cv2
import os
import torch



# train=['dry_train','wet_train','snow_train','unknown_train']
# test=['dry_test','wet_test','snow_test','unknown_test']
# cnt=0
# for i in range(len(train)):
#     if not os.path.exists(train[i]):
#         os.mkdir(train[i])
#     trainfiles=os.listdir("%s2/"%train[i])
#     for fil in trainfiles:
#         img=cv2.imread('%s2/%s'%(train[i],fil),1)
#         cv2.imwrite("%s/%s%d.jpg"%(train[i],train[i],cnt),img)
#         cnt+=1

trainfiles = os.listdir('./img_train')
# print(type(trainfiles), len(trainfiles))
img_total_train = torch.zeros(1, 3, 60, 60)
label_total_train = torch.tensor([0])
print(img_total_train.shape)

ii = 0
for trainfile in trainfiles:
    img = cv2.imread('./img_train/'+trainfile)
    img = torch.from_numpy(img).transpose(0, 2).transpose(1, 2).unsqueeze(dim=0).float()
    img_total_train = torch.cat([img_total_train, img], dim=0)

    if 'dry' in trainfile:
        label = torch.tensor([0])
    elif 'wet' in trainfile:
        label = torch.tensor([1])
    else:
        label = torch.tensor([2])
    label_total_train = torch.cat([label_total_train, label], dim=0)
    ii += 1
    print(ii, img.shape, label)
print('img_total_train:', img_total_train.shape, 'label_total_train:', label_total_train)

img_total_train = img_total_train[torch.arange(img_total_train.size(0)) != 0]   # 删除第一个初始创造的0元素
label_total_train = label_total_train[torch.arange(label_total_train.size(0)) != 0]   # 删除第一个初始创造的0元素
print('img_total_train:', img_total_train.shape, 'label_total_train:', label_total_train)


torch.save(img_total_train, './dataset/train_x.t')
torch.save(label_total_train, './dataset/train_y.t')


#
testfiles = os.listdir('./img_test')
# print(type(testfiles), len(testfiles))
img_total_test = torch.zeros(1, 3, 60, 60)
label_total_test = torch.tensor([0])
print(img_total_test.shape)

ii = 0
for testfile in testfiles:
    img = cv2.imread('./img_test/'+testfile)
    img = torch.from_numpy(img).transpose(0, 2).transpose(1, 2).unsqueeze(dim=0).float()
    img_total_test = torch.cat([img_total_test, img], dim=0)

    if 'dry' in testfile:
        label = torch.tensor([0])
    elif 'wet' in testfile:
        label = torch.tensor([1])
    else:
        label = torch.tensor([2])
    label_total_test = torch.cat([label_total_test, label], dim=0)
    ii += 1
    print(ii, img.shape, label)
print('img_total_test:', img_total_test.shape, 'label_total_test:', label_total_test)

img_total_test = img_total_test[torch.arange(img_total_test.size(0)) != 0]   # 删除第一个初始创造的0元素
label_total_test = label_total_test[torch.arange(label_total_test.size(0)) != 0]   # 删除第一个初始创造的0元素
print('img_total_test:', img_total_test.shape, 'label_total_test:', label_total_test)


torch.save(img_total_test, './dataset/test_x.t')
torch.save(label_total_test, './dataset/test_y.t')


