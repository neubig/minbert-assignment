# minbert Assignment
by Zhengbao Jiang, Shuyan Zhou, and Ritam Dutt

This is an exercise in developing a minimalist version of BERT, part of Carnegie Mellon University's [CS11-747: Neural Networks for NLP](http://www.phontron.com/class/nn4nlp2020/).

In this assignment, you will implement some important components of the BERT model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with the BERT model.

## Assignment Details

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use `torch`, no other external libraries are allowed (e.g., `transformers`).
* We will run your code with the following commands, so make sure that whatever your best results are reproducible using these commands (where you replace ANDREWID with your andrew ID):
```
mkdir -p ANDREWID

python3 classifier.py --option [pretrain/finetune] --epochs NUM_EPOCHS --lr_pretrain LR_FOR_PRETRAINING --lr_finetune LR_FOR_FINETUNING --seed RANDOM_SEED
```
## Reference accuracies: 

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

Pretraining for SST:
Dev Accuracy   : 0.387 (0.008)
Test Accuracy  : 0.397 (0.013)

Finetuning for SST :
Dev Accuracy   : 0.520 (0.006)
Test Accuracy  : 0.525 (0.007)


### Submission
The submission file should be a zip file with the following structure (assuming the andrew id is ``ANDREWID``):
```
ANDREWID/
├── base_bert.py
├── bert.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── sst-dev-output.txt 
├── sst-test-output.txt 
├── cfimdb-dev-output.txt 
├── cfimdb-test-output.txt 
└── setup.py
```

### Grading
* A+: You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements:
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the MLM objective to do domain adaptation
    * try [alternative fine-tuning algorithms](https://www.aclweb.org/anthology/2020.acl-main.197)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option finetune`` code that achieves comparable accuracy to our reference implementation
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` code that achieves comparable accuracy to our reference implementation
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py``, but accuracy is not comparable to the reference.
* B or below: Some parts of the missing pieces are not implemented.

### Acknowledgement
Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
