import os

net = "vgg16"
part = "test_t"
dataset = "cityscape"
begin_epoch =9
end_epoch =9

model_prefix = "./FLDMN/cityscape/model/cityscape_"

commond = "./FLDMN/eval/test.py --net {} --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")
