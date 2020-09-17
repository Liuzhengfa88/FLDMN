import os

net = "res101"
part = "test_t"
dataset = "clipart"
begin_epoch =1
end_epoch =3

model_prefix = "/home/liuzhengfa/code/CR-DA-DET/DA_Faster_ICR_CCR_Clipart1k/clipart/model/clipart_"

commond = "python  /home/liuzhengfa/code/CR-DA-DET/DA_Faster_ICR_CCR_Clipart1k/eval//test.py --net {} --cuda --dataset {} --part {} --model_dir {}".format(net, dataset, part, model_prefix)

for i in range(begin_epoch, end_epoch + 1):
    print("epoch:\t", i)
    os.system(commond + str(i) + ".pth")