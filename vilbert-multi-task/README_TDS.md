## Dependencies
* Python 3.6
* Pytorch 1.7.1
* Torchvision==0.8.2
* Dependencise please follow the original [vilibert repo](https://github.com/facebookresearch/vilbert-multi-task)
* We train and evaluate all of the models based on one TITAN Xp GPU

### Download the baseline models pre-trained on VQA-CP v2 training set 

 | Model | Link| Acc (%)|
 | :-: | :-: | :-: |
 |ViLBERT (VQACP-v2)|https://gh.h233.eu.org/https://github.com/Zhiquan-Wen/TDS/releases/download/Pretrained_Model_on_VQACPv2_train_vilbert/pytorch_model_18.bin|40.75|
 |ViLBERT (VQACP-v1)|https://ghps.cc/https://github.com/Zhiquan-Wen/TDS/releases/download/Pretrained_Model_on_VQACPv1_train_vilbert/pytorch_model_12.bin|39.59|

### Download other features
VilBert require the height and width of the image, which can be download [here](https://hub.gitmirror.com/https://github.com/Zhiquan-Wen/TDS/releases/download/img_height_and_width/imgid_to_height_and_width.npy)

### Test-Time Adaptation 
* Evaluate and Train our model simultaneous
```
CUDA_VISIBLE_DEVICES=0 python vilbert-multi-task/TDS.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features (with boxes) --output saved_models_cp2/test.json --batch_size 32 --learning_rate 0.0002 --rate 0.2 --checkpoint_path path_for_pretrained_model
``` 

### Evaluation
* Compute detailed accuracy for each answer type on a json file of results:
```
python comput_score.py --input saved_models_cp2/test.json --dataroot data/vqacp2/
```
* Our results of ViLBERT for [VQACP v1](https://slink.ltd/https://github.com/Zhiquan-Wen/TDS/releases/download/ViLBERT_vqacpv1_results/minval_result.json) and [VQACP v2](https://download.nuaa.cf/Zhiquan-Wen/TDS/releases/download/ViLBERT_vqacpv2_results/minval_result.json) results can be downloaded.


## Quick Reproduce

1. **Preparing enviroments**: we prepare a docker image (built from [Dockerfile](https://github.com/Zhiquan-Wen/D-VQA/blob/master/docker/Dockerfile)) which has included above dependencies, you can pull this image from dockerhub or aliyun registry:

```
docker pull zhiquanwen/debias_vqa:v1
```


```
docker pull registry.cn-shenzhen.aliyuncs.com/wenzhiquan/debias_vqa:v1
docker tag registry.cn-shenzhen.aliyuncs.com/wenzhiquan/debias_vqa:v1 zhiquanwen/debias_vqa:v1
```

2. **Start docker container**: start the container by mapping the dataset in it:

```
docker run --gpus all -it --ipc=host --network=host --shm-size 32g -v /host/path/to/data:/xxx:ro zhiquanwen/debias_vqa:v1
```

3. **Running**: refer to `Download and preprocess the data`, `Training` and `Evaluation` steps in `Getting Started`.

## Reference
If you found this code is useful, please cite the following paper:
```
@inproceedings{TDS,
  title     = {Test-Time Model Adaptation for Visual Question Answering with Debiased Self-Supervisions},
  author    = {Zhiquan Wen, 
               Shuaicheng Niu, 
               Ge Li,
               Qingyao Wu, 
               Mingkui Tan, 
               Qi Wu},
  booktitle = {IEEE TMM},
  year = {2023}
}
```