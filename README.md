# minbert Assignment
by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt and Brendon Boldt

This is an exercise in developing a minimalist version of BERT, part of Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html).

In this assignment, you will implement some important components of the BERT model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with the BERT model.

## Assignment Details

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* We will run your code with the following commands, so make sure that whatever your best results are reproducible using these commands (where you replace ANDREWID with your lowercase Andrew ID):
    * Do not change any of the existing command options (including defaults) or add any new required parameters
```
mkdir -p ANDREWID

python3 classifier.py --option [pretrain/finetune] --epochs NUM_EPOCHS --lr LR --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt
```
## Reference accuracies: 

Pretraining for SST:
Dev Accuracy: 0.391 (0.007)
Test Accuracy: 0.403 (0.008)

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

Finetuning for SST:
Dev Accuracy: 0.515 (0.004)
Test Accuracy: 0.526 (0.008)

Finetuning for CFIMDB:
Dev Accuracy: 0.966 (0.007)
Test Accuracy: -

### Submission

We are asking you to submit in two ways:
1. *Canvas:* a full code package, which will be checked by the TAs in the 1-2 weeks 
   after the assignment for its executability.
2. *ExplainaBoard:* which will grade your assignment immediately, so you can make sure
   that your accuracy matches what you would expect.

#### Canvas Submission

For submission via [Canvas](https://canvas.cmu.edu/),
the submission file should be a zip file with the following structure (assuming the
lowercase Andrew ID is ``ANDREWID``):
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

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It
will throw assertion errors if the format is not expected, and *submissions that fail
this check will be graded down*.

Usage:
1. To create and check a zip file with your outputs, run
   `python3 prepare_submit.py path/to/your/output/dir ANDREWID`
2. To check your zip file, run
   `python3 prepare_submit.py path/to/your/submit/zip/file.zip ANDREWID`

Please double check this before you submit to Canvas; most recently we had about 10/100
students lose a 1/3 letter grade because of an improper submission format.

#### ExplainaBoard Submission

To submit your outputs via [ExplainaBoard](https://explainaboard.inspiredco.ai), first
click the top-right of the site to log in, and then again click the top-right to view
your API key. Run the following to save your email and API key to environmental
variables:

```
export EB_EMAIL=your_email_used_for_explainaboard
export EB_API_KEY=your_api_key_for_explainaboard
export EB_ANDREW_ID=your_andrew_id
```

Now you can upload the outputs of your model with the `upload_results.py` script. There
are the following options.

* `--system_name` a name that you can choose for your system. Your final system name
  will be `anlp_{andrewid}_{system_name}`.
* `--dataset` the dataset name (sst/cfimdb).
* `--split` the split (dev/test).
* `--output` the system output you're uploading.
* `--public` if you want your output listed on the public site so people in the class
  can compare and contrast with it add this flag. But it is off by default (and has no
  effect on your grade).

Here is an example of uploading all of the datasets with a system name of `baseline`.

```
python upload_results.py --system_name baseline --dataset sst --split dev --output sst-dev-output.txt
python upload_results.py --system_name baseline --dataset sst --split test --output sst-test-output.txt
python upload_results.py --system_name baseline --dataset cfimdb --split dev --output cfimdb-dev-output.txt
python upload_results.py --system_name baseline --dataset cfimdb --split test --output cfimdb-test-output.txt
```

You can then go to the ExplainaBoard systems page to confirm that the results are
uploaded properly.

### Grading
* A+: You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements. Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the MLM objective to do domain adaptation
    * try [alternative fine-tuning algorithms](https://www.aclweb.org/anthology/2020.acl-main.197)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` and ``--option finetune`` code that achieves comparable accuracy to our reference implementation.
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` and ``--option finetune`` code but accuracy is not comparable to the reference.
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py`` (bert implementation) and ``optimizer_test.py`` (optimizer implementation)
* B or below: Some parts of the missing pieces are not implemented.

If your results can be confirmed through ExplainaBoard, but there are problems with your
code submitted through Canvas, such as not being properly formatted, not executing in
the appropriate amount of time, etc., you will be graded down 1/3 grade.

All assignments must be done individually and we will be running plagiarism detection
on your code. If we confirm that any code was plagiarized from that of other students
in the class, you will be subject to strict measure according to CMUs academic integrity
policy.

### Acknowledgement
Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
