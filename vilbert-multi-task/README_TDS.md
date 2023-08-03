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
VilBert require the height and width of the image (imgid_to_height_and_width.npy), which can be download [here](https://hub.gitmirror.com/https://github.com/Zhiquan-Wen/TDS/releases/download/img_height_and_width/imgid_to_height_and_width.npy)

### Test-Time Adaptation 
* Evaluate and Train our model simultaneous

First, fill out the necessary hyper-parameters in the vilbert_tasks.yml file based on our paper.

```
CUDA_VISIBLE_DEVICES=0 python vilbert-multi-task/TDS.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 20 --batch_size 32 --output_dir <path to save results>
``` 

### Evaluation
* Compute detailed accuracy for each answer type on a json file of results:
```
python comput_score.py --input saved_models_cp2/test.json --dataroot data/vqacp2/
```
* Our results of ViLBERT for [VQACP v1](https://slink.ltd/https://github.com/Zhiquan-Wen/TDS/releases/download/ViLBERT_vqacpv1_results/minval_result.json) and [VQACP v2](https://download.nuaa.cf/Zhiquan-Wen/TDS/releases/download/ViLBERT_vqacpv2_results/minval_result.json) results can be downloaded.


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