# trend_ml_toolkit_xgboost_v1.1 说明文档

[TOC]

## 前言

* 该工具集主要依托[xgboost](https://github.com/dmlc/xgboost)与[sklearn](http://scikit-learn.org/stable/)进行xgboost模型训练，进行***文件是/否病毒文件***的二分类任务；并且针对测试结果进行画图，与其他模型，譬如svm，进行效果比较
* 并且具备一些子功能，譬如不同格式之间数据的转换；对文件进行分拣，对特征进行合并，对文件内容进行hash编码等等。

## 原始数据格式说明

* 原始数据，是由上游交付，其生成***完全独立于本工具集***，是我们数据处理的最起始点。

* 对于原始数据的数据格式，我们称之为NN格式

* 一个NN格式的数据集，包含两个文本文件（下面以训练集为例）

  * NN_train.txt：存储3个信息：样本的维度；样本的label；样本的features。具体的为，文件第一行存储一个数值，表示该数据集中样本的维度，也即为样本的总特征数量；接下来N行，表示存储N个样本的信息，每一行具体的表示为：[***样本label***];[***样本第一个非0特征的索引***];[***样本第一个非0特征***];[***样本第二个非0特征的索引***];[**样本第二个非0特征**];...;[***样本第D个非0特征的索引***];[***样本第D个非0特征***];

    可见，NN数据格式是非稀疏的，因为并没有直接存储0值特征，并且需要注意的是**索引是以1开头的，不是0开始索引**。

  * NNAI_train.txt：存储2个信息：样本的label；该样本的来源。具体的为，该文件有N行，表示N个样本，每一行的表示为：[***样本标签***]|[***样本的来源路径***]

* 示例

  若一个样本，其label为1，features为 [3.3,0.5,0.0,22]，来源路径为D:\\temp\temp.txt，那么其在NN_train.txt中表示为

  ```
  1;1;3.3;2;0.5;4;22
  ```

  在NNAI_test.txt中表示为

  ```
  1|D:\temp\temp.txt
  ```

  真实数据示例展示如下

  NN_train.txt

  ```
  1108
  1;1;45;3;0.0027597123;4;0.0020697842;5;0.057264031;6;0.0041395685;8;0.013798562;9;0.0013798562;10;0.0041395685;20;0.019317986;21;0.036566188;28;0.065543168;30;0.0013798562;33;0.016558274;38;0.0027597123;40;0.042085613;41;0.0027597123;42;0.00068992808;51;0.0068992808;55;0.012418705;57;0.96658924;59;0.075892089;61;0.043465469;63;0.05381439;64;0.00068992808;74;0.033116548;88;0.0048294966;89;0.033116548;106;0.17593166;107;0.029666907;129;0.0013798562;132;0.00068992808;137;0.080031657;141;0.031046764;143;0.022077699;155;0.011038849;161;0.0013798562;174;0.00068992808;177;0.0013798562;182;0.00068992808;212;0.0034496404;220;0.013108634;234;0.00068992808;261;0.033806476;278;0.0013798562;337;0.0027597123;370;0.00068992808;
  1;1;37;5;0.053668617;6;0.0022361924;8;0.0011180962;9;0.0027952405;10;0.0011180962;20;0.010621914;21;0.0016771443;28;0.30579931;30;0.0016771443;33;0.006149529;39;0.0011180962;40;0.045841944;41;0.00055904809;42;0.0011180962;55;0.028511453;57;0.87323312;59;0.074912444;61;0.04696004;63;0.025157164;74;0.14311631;89;0.025157164;101;0.00055904809;106;0.27728785;107;0.0055904809;132;0.00055904809;137;0.06149529;141;0.0022361924;143;0.16603728;146;0.0044723847;161;0.0011180962;216;0.010621914;223;0.00055904809;261;0.0016771443;429;0.00055904809;440;0.00055904809;474;0.00055904809;597;0.0016771443;
  ```

  NNAI_train.txt

  ```
  1|G:\MacX\d3_oc\Test\bad\test\003e6dc030f5c63db601fc871b48562b308891029dd99ce412f3ce54d1d7ad0c.opcode
  1|G:\MacX\d3_oc\Test\bad\test\00872dfe996f3465de366ee3f1f3312970b2dbb625202df4ff2b9c4aa312613d.opcode
  ```

  ​

## 适配数据格式说明

* 所谓**适配数据格式**，指的是为了训练xgboost模型，所需的数据格式

* 本工具集目前只支持**libsvm**数据格式，所以为了训练xgboost模型，所有的数据格式都必须先转换为libsvm数据格式

* libsvm数据格式

  ```
  Label 1:value 2:value ...
  # 需要注意的是，libsvm也是非稀疏的，意即不直接存储0值特征
  ```

* 对于libsvm，若仍有疑问，详见[这里](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

## 目录结构说明

* Data文件夹：储存样本数据，包括原始的训练数据（譬如NN_train.txt NNAI_train.txt），适配于xgboost的数据（libsvm数据格式）
* Figures文件夹：存储模型在数据上的画图结果
* Models文件夹：存储训练好的模型
* Old文件夹：存放一些旧的脚本，该文件夹下的脚本不会用到，一般仅作用代码阅读
* Output文件夹：存放一些脚本的输出
* Temp文件夹：存放一些临时文件，譬如log日志等等
* tools.py：该脚本存放一些工具函数，并且提供了命令行形式的将***NN数据格式转换为 libsvm数据格式***的接口
* xg_train_cv.py，xg_train_cv.config，xg_train_cv.sh：根据配置文件——xg_train_cv.config，训练xgboost模型
* xg_predict.py：使用xgboost模型进行预测
* compare.config，compare.py，compare.sh：根据配置文件——compare.config，比较模型的效果
* feature_hash.py：将文件内容哈希到指定长度，作为一种特征工程的手段。
* dataset_shake_to_NN.py：将哈希之后得到的内容，转换为NN数据格式。
* rf_train.py：给定样本，训练**随机森林**分类器（**该脚本可正常使用，但是时间复杂度较高**）
* xg_train_untuned.py ：该脚本训练未调参，即默认参数的xgboost模型。
* 其他的一些说明
  * xg_train_untuned.config  xg_train_untuned.py  xg_train_untuned.sh： 根据指定任务，训练未调参的xgboost模型
  * xg_train_slower.config  xg_train_slower.py ：一种较慢方式的xgboost模型方式
  * xg_train.config  xg_train.py ：一种较为繁琐的xgboost模型训练方式

## pipeline说明

* NN数据格式转化为libsvm数据格式

  * 因为现有的数据格式为NN格式，为了训练xgboost模型，系统先将NN数据格式转换为libsvm数据格式

  * 实例

    ```shell
    # -s 表示 NN格式样本的特征数据 -l 表示 NN格式样本的label数据；生成的libsvm数据存放在./Data路径下
    python tools.py -s ./Data/NN_train.txt -l ./Data/NNAI_train.txt
    ```

* xgboost模型训练

  * 设置xg_train_cv.config中的配置项，详细的xgboost参数含义，可以查看[这里](http://xgboost.readthedocs.io/en/latest/parameter.html)

    ```
    [xg_conf]
    # DO NOT DELET OR ADD ANY PARAMETERS HERE. IF YOU HAVE TO, PLEASE REVISE THE CODE: xg_train_cv.py

    # ==========   General Parameters, see comment for each definition  ===========
    # choose the booster, can be gbtree or gblinear
    booster = gbtree
    # Do not show the detailed information[1 Yes, 0 NO]
    silent = 1
    # ===============   Task Parameters   =================
    # choose logistic regression loss function for binary classification
    objective = binary:logistic
    base_score = 0.5
    seed = 0

    # =============== common Parameters ====================
    # 0 means do not save any model except the final round model
    save_period = 0
    # The path of training data
    # Is the training data xg format? [1 Yes, 0 No]
    xgmat = 0
    data = Data/OC-vuq2/NN_train.txt
    label = Data/OC-vuq2/NNAI_train.txt
    xgdata = Data/OC-vuq2/NN_train.txt.libsvm
    eval_metric = logloss
    ascend = 0
    # eval: show the train error in each round[0 no]
    eval = 1
    cv = 5
    #  MultiThread
    nthread = 4
    [xg_tune]
    #===============  parameters need to be tuned =================
    # the number of round to do boosting
    num_round = 500
    # maximum depth of a tree
    max_depth = 4,6,8,10,15
    # max_depth = 8
    subsample = 0.7,0.8,0.9,1.0
    #subsample = 1.0
    min_child_weight = 0.3,0.8,1,2
    # min_child_weight = 0.1
    colsample_bytree = 0.7,0.8,0.9,1.0
    #colsample_bytree = 0.7
    ```

    * xgmat：bool。0或者1，表示训练数据是否已经是xgboost所需的数据格式，即是否为libsvm格式；0表示否，1表示是

    * data：string。NN数据格式中的features数据，该设置仅在xgmat设置为0的情况下有效。

    * label：string。NN数据格式中的label数据，该设置仅在xgmat设置为0的情况下有效。

    * xgdata：string。直接进行xgboost训练的数据，即为libsvm数据格式，该设置仅在xgmat设置为1的情况下有效。

    * eval_metric：string。训练模型时所采用的评估指标。具体可设置选项可以查看[这里](http://xgboost.readthedocs.io/en/latest/parameter.html)

    * ascend：bool。0或者1。表示是以***升序或者降序***的方式选出最后一个作为***最优结果***。其中，0表示降序方式；1表示升序方式。

    * eval：bool。是否在每一个训练周期中展示训练误差。

    * cv：int。表示交叉验证的时候，设置多少个fold。

    * nthread：int。设置多线程，进行交叉验证时设置多少个线程。

    * num_round：int。待调参数，表示训练xgboost时，最大的迭代次数，也就是树的棵数。系统将在num_round次迭代中找到最优的迭代次数。

    * max_depth：一个int，或者多个int，用半角逗号进行分隔。待调参数，表示训练xgboost时每一棵树的最大深度。如果为一个int，表示该参数已经调参完毕；如果为多个int，譬如

      > max_depth = 10,20,40

      系统将在这几个数中选出最优的作为调参结果。

    * subsample：一个int，或者多个int，用半角逗号进行分隔。用法同max_depth。

    * min_child_weight：一个int，或者多个int，用半角逗号进行分隔。用法同max_depth。

    * colsample_bytree：一个int，或者多个int，用半角逗号进行分隔。用法同max_depth。


*   模型训练：

    ```shell
    sh xg_train_cv.sh 
    ```

    系统将执行xg_train_cv.py，并且将终端输出存储到Temp文件夹下的日志文件中。系统将最终训练好的xgboost模型存储到磁盘上，Models目录下

*   模型预测

    * 预测配置：用户在xg_predict.py的get_config函数中设置用户自定义项

      ```python
      def get_config():
          config = dict()
          # 要执行预测功能的模型的存放路径
          config['model_path'] = './Models/OC-vuq1/2017_05_25_12_52_34.xgmodel'
        # 测试数据集存放路径，注意，必须是libsvm数据格式，不可以是NN数据格式，如果只有NN格式的数据，先要进行格式转换
          config['data_path'] = './Data/OC-vuq1/NN_test.txt.libsvm'
          # 保存预测结果的文件存放路径（无需事先建立文件）
          config['result_path'] = './Output/result.csv'
          # 测试数据NN数据格式的label数据，为的是对预测的样本能够进行回溯
          config['label_path'] = './Data/OC-vuq1/NNAI_test.txt'
      ```

    * 执行预测

      ```shell
      python xg_predict.py 
      ```

*   模型比较

    * 在compare.config中设置用户自定义项目

      ```
      [compare]
      model_paths = Models/OC-vuq1/2017_05_25_12_52_34.xgmodel,Models/OC-vuq2/2017_05_25_22_24_20.xgmodel,Models/OC-vuq3/2017_05_25_18_38_09.xgmodel
      datasets = Data/OC-vuq1/NN_test.txt.libsvm,Data/OC-vuq2/NN_test.txt.libsvm,Data/OC-vuq3/NN_test.txt.libsvm
      #labels = Data/OC-vuq1/NNAI_test.txt,Data/OC-vuq2/NNAI_test.txt,Data/OC-vuq3/NNAI_test.txt
      dataset_formats = xgboost,xgboost,xgboost
      model_names = OC-vuq1,OC-vuq2,OC-vuq3
      thres = 0.5,0.5,0.5
      markers = g-,r-,b-
      ```

      * model_paths：一个string，或者多个string，之间用逗号进行分隔。表示要进行比较的模型的存放路径，一个string表示仅仅查看一个模型的效果，多个string表示将多个模型进行比较。
      * datasets：设置同model_paths，表示评估模型所需要的数据集。如果对应的模型是xgboost模型，那么datasets的该项必须是libsvm格式的数据集。
      * dataset_formats：表示datasets中设置的各个数据对应的数据形式，即适配于***何种分类器***，目前系统仅仅支持svm以及xgboost两种分类器，详情可以查看源码
      * model_names：设置同model_paths。表示代表各个模型的名称。用户自定义。
      * thres：一个int，或者多个int，之间用逗号进行分隔。表示进行二分类时的切分阈值。譬如，0.5，如果预测得分大于0.5，表示该样本被标记为1，否则被标记为0。
      * markers：设置同model_paths，表示评估模型画图时所采用的线条。可设置值详见[这里](https://matplotlib.org/api/pyplot_api.html)
      * 注意：配置文件中，每项配置的可取值数目必须相等

    * 执行比较

      ```shell
      python compare.py -c compare.config
      # 或者
      # ./compare.sh
      ```

*   哈希编码

    * 文件分拣

      * 在源文件file_classifier.py中设置配置项

        ```python
        def get_config():
            config = dict()
            # 分拣哪一个数据集的文件，可供设置的选项为 'train' 'test'
            config['phase'] = 'test'
            # 存放待分拣文件的文件夹
            config['handle_path'] = '/macml-data/features/opcode'
            # train数据集的0类文件存储地址
            config['result_path0'] = '/home/lili/opcode-2017-05/train/0'
            # train数据集的1类文件存储地址
            config['result_path1'] = '/home/lili/opcode-2017-05/train/1'
            # test数据集的0类文件存储地址
            config['result_path2'] = '/home/lili/opcode-2017-05/test/0'
            # test数据集的1类文件存储地址
            config['result_path3'] = '/home/lili/opcode-2017-05/test/1'
            # 分拣train数据集的参考文件
            config['train_csv'] = '/home/lili/datasets/2017-05_train.csv'
             # 分拣test数据集的参考文件
            config['test_csv'] = '/home/lili/datasets/2017-05_test.csv'
            # 处理器的数目设置，用于多线程；源代码也设置了单线程的处理方法
            config['processes'] = None
            return config
        ```

    * 执行分拣

      ```bash
      python file_classifier.py
      ```

      分拣完毕后将得到4个文件夹

      xxxx/train/0

      xxxx/train/1

      xxxx/test/0

      xxxx/test/1

    * 哈希编码：将每一个文件的内容哈希编码为1024个bit，使用了SHA3-128算法，SHA算法簇情况看[SHA](https://en.wikipedia.org/wiki/Secure_Hash_Algorithms)

      * 在feature_hash.py中设置配置项

        ```python
        def get_config():
            config = dict()
            # 要处理的文件夹路径,这里一定要注意：config['handle_path']下的目录结构必须为
            # config['handle_path']/train/0
            # config['handle_path']/train/1
            # config['handle_path']/test/0
            # config['handle_path']/test/1
            config['handle_path'] = '/home/raymon/trend_ml_toolkit_xgboost/Data/'
            # 处理结果存放路径，将具有如下目录结构：
            # config['re_path']/train/0
            # config['re_path']/train/1
            # config['re_path']/test/0
            # config['re_path']/test/1
            config['re_path'] = '/home/raymon/trend_ml_toolkit_xgboost/Data/'
            # 编码长度
            config['length'] = 1024
            # 设置线程数，用于多线程处理，None表示利用最大核心数；源代码中也实现了单线程函数handle_f_single_thread
            config['processes'] = None
            return config
        ```

    * 将哈希编码的结果整合为NN数据集

      * 设置配置项

        ```python
        def get_config():
            config = dict()
            # 哈希结果存放路径
            config['data_path'] = '/home/lili/opcode-2017-05-hash/'
            config['NN_train'] = './Data/opcode-2017-05-hash/NN_train.txt'
            config['NNAI_train'] = './Data/opcode-2017-05-hash/NNAI_train.txt'
            config['NN_test'] = './Data/opcode-2017-05-hash/NN_test.txt'
            config['NNAI_test'] = './Data/opcode-2017-05-hash/NNAI_test.txt'
            return config
        ```

      * 执行转换

        ```shell
        python dataset_shake_to_NN.py
        ```


## 其他

* feature_clean_Normalize-Opcode.py：去除文件中连续重复的指令（可选），以及将隶属于同一个组的指令映射到该组中的第一个指令（可选），去除非法指令。

  * 举例：xxx.opcode文件内容如下

    ```
    mov ebp,eap
    mov
    mov 
    add ebp,eap
    ins1
    mov
    ins2
    abc
    ```

    其中，前3个mov指令连续出现，则保留一个；ins1与ins2指令属于同一个组（我们假定是这样的），那么ins1与ins2都映射到ins1，abc为非法指令，理应去除。

    处理结果理应如下：

    ```
    mov
    add
    ins1
    mov
    ins2
    ```

  * 用法：feature_clean_Normalize-Opcode.py的选项说明为

    ```python
    # parser
    def arg_parser():
        parser = argparse.ArgumentParser()
        # 待处理路径
        parser.add_argument('-i','--inputFolder', required=True)
        # 输出目录
        parser.add_argument('-o', '--outputFolder', required=True)
        # 指令集合文件:工程中已经给出 instructions.txt
        parser.add_argument('-is', '--instructFile', required=True)
        # 是否去除 连续重复的 指令
        parser.add_argument('-rs','--remove',default=1)
        # 是否将同组的指令映射到该组的第一个指令
        parser.add_argument('-g','--group',default=1)
        return parser.parse_args()
    ```

    ```shell
    python -i InputFolder/ -o OutputFolder/ -is instructions.txt -rs 1 -g 1
    ```

  * 该脚本可以稍加修改，改成多线程处理

## 附录

- scikit-learn 模型评估指标

    ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

- 项目github地址：https://github.com/raymon-tian/trend_ml_toolkit_xgboost

- xgboost调参

  - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
  - http://blog.csdn.net/wzmsltw/article/details/52382489
  - https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/19083

- xgboost参数详解文档：http://xgboost.readthedocs.io/en/latest/parameter.html

- xgboost Python API文档：http://xgboost.readthedocs.io/en/latest/python/python_intro.html