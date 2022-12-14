# <center>Implement Image classification with Pytorch</center>

Repo này giúp giải quyết bài toán Image classification bằng các model phổ biến: Efficientnet, VGG, Resnet, GoogleNet. Hơn nữa ta có thể sử dụng transfer learning để tiết kiệm thời gian và tài nguyên tính toán nhưng vẫn đạt được model có độ chính xác cao.

## 0. Clone source code và cài môi trường
```
git clone https://github.com/lynguyenminh/image-classification-pytorch.git 
cd image-classification-pytorch

pip install -r requirements.txt

Nếu gặp lỗi thì pip install -r requirements.txt --no-cache-dir

```

## 1. Cấu trúc thư mục & chuẩn bị data
Hãy chuẩn bị data và code theo cấu trúc sau: 
```
Main-folder/
│
├── data/ - This folder contain data for training model
│   ├── train
|   |   ├── class 1
|   |   |   ├── img_1.jpg
|   |   |   ├── img_2.jpg
|   |   |   └── ...
|   |   └── class 2
|   |       ├── img_1.jpg
|   |       ├── img_2.jpg
|   |       └── ...
│   └── val
|       ├── class 1
|       |   ├── img_3.jpg
|       |   ├── img_4.jpg
|       |   └── ...
|       └── class 2
|           ├── img_3.jpg
|           ├── img_4.jpg
|           └── ...
|
├── predict/ - public test images
│   └── imgs
|       ├── test_1.jpg
|       ├── test_2.jpg
|       └── ...
|
├── src/ - source code
│   ├── config.yaml
│   ├── predict.py - Code predict
│   ├── train.py - Code train model
│   └── utils
|       ├── load_data.py
|       ├── load_model.py
|       ├── load_optim.py
|       ├── load_loss.py
|       └── train_model.py
|
└── weights/ - this folder contains weights after training.
    ├── best.pt
    └── epoch_1.pt
```

## 2. Train model
**CHÚ Ý**

Ta có thể thực hiện tăng cường dữ liệu trực tiếp bằng Pytorch trước khi train. Sửả code ở [đây](https://github.com/lynguyenminh/image-classification-pytorch/blob/master/src/utils/load_data.py#L18)

Có thể thay đổi hàm loss function tại [đây](https://github.com/lynguyenminh/image-classification-pytorch/blob/master/src/train.py#L48). Các hàm có sẵn là: `CrossEntropyLoss`, `NLLLoss`.

Có thể thay đổi hàm optimization tại [đây.](https://github.com/lynguyenminh/image-classification-pytorch/blob/master/src/train.py#L51). Các hàm có sẵn là: `Adam`, `RAdam`, `SGD`, `Adadelta`, `Adagrad`, `AdamW`, `Adamax`, `ASGD`, `NAdam`, `Rprop`.


Chạy script sau để train model: 

```
python3 train.py \
        --model_name "resnet18" \
        --epoch 10 \
        --data ../data \
        --batchsize 16 \
        --save_weights ../weights \
        --numclass 2
```

--model_name: tên thuật toán phân loại: 

* Efficientnet: `efficientnetb0`, `efficientnetb1`, `efficientnetb2`, `efficientnetb3`, `efficientnetb4`, `efficientnetb5`, `efficientnetb6`, `efficientnetb7`
* VGG: `vgg11`, `vgg11bn`, `vgg13`, `vgg13bn`, `vgg16`, `vgg16bn`, `vgg19`, `vgg19bn`
* Resnet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet115`
* Googlenet: `googlenet` 

--epoch: số epoch train model.

--data: path của data train.

--batchsize: batchsize. Chú ý nếu máy yếu thì giá trị batchsize nên nhỏ.

--save_weights: path lưu weight sau khi train.

--numclass: số class muốn train.

Sau khi train, sẽ có file best.pt là file weight có f1_score lớn nhất trên tập val.
## 3. Inference model
```
python3 predict.py \
        --model_name "vgg11" \
        --test_path ./predict \
        --weights ../weights/best.pt \
        --numclass 2
```
--model_name: Giống như ở phần train.

--test_path: path chứa ảnh test. Path này có thể là folder hay file image đều đươc.

--weights: path file weight.

--numclass: số class model cần phân loại (=numclass lúc train).
Sau khi inference, thì kết quả lưu trong `predict.csv`.

Nếu gặp lỗi, vui lòng tạo issue hay liên hệ trực tiếp với tôi: 20521592@gm.uit.edu.vn.
