# EfficientNetV2++: An efficient handwritten Chinese character recognition model based on capturing multi-style attention
## License
This work is only allowed for academic research use. For commercial use, please contact the author.
## Structure
- CDC_Dataset: Chinese character dataset
- checkpoints: Weight files
- confidencs: Confidence data
- enhance: Character style enhancement
- logs: Logs
- models: Model zoo
  - network: EfficientNetv2++ and MSA modules
- utils: Utilities
- scripts: Additional scripts
- dataloder.py
- train.py
- test.py
## Download
- Clone this repo:
```bash
git clone https://github.com/bzlmh/CHCRecognition
cd CHCRecognition
```
## Requirements
- Install the requirements.txt

## How to get CHC-Dataset
- [CDCDataset](https://pan.baidu.com/s/1AG990agFmgl6jJhNrWJTkQ?pwd=vrdn)
#### Detailed information of CDC-Dataset
| Dataset    | Character number | Sample number | Sample size |
|:----------:|:----------------:|:-------------:|:-----------:|
| Augmented  |       600        |    19,958     |    64×64    |
| Train_data |       600        |    14,253     |    64×64    |
| Test_data  |       600        |     3,644     |    64×64    |
#### Training on the CDC-Dataset
```bash
- Command to train:
python train.py 
- Command to test: 
python test.py
- Command to visualization:  we provide the grad-cam tools to explain our attention mechanism
python visualization_msa.py
```

#### Training on your own data
If you need to expand the data, you could use the ./enhance/augment.py. Augment.py supports a variety of style change methods.

```bash
- Command to train:
python train.py 
- Command to test:
python test.py
```
- Specifying the batch size and the number of epochs could be done inside the code.
#### Other Datasets and Papers
- [Multi-scene ancient Chinese text recognition](https://1drv.ms/u/s!AqAU14ep3HF7bhNy8KcOtjfEpbI) 
- [OBC306: A Large-Scale Oracle Bone Character Recognition Dataset](https://www.omniglot.com/chinese/jiaguwen.htm) 
- [CASIA-AHCDB: A Large-Scale Chinese Ancient Handwritten Characters Database](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html)

