# MyNet

通过分类网络的实现, 学习tensorflow :smile:

## 各个网络

* vgg11

    * train_acc: 0.9909;
    * test_acc_from_testdataset: 0.9119
    * evel_acc_from_alldataset: 0.9120

    * 'batch_size'   : 128,
    * 'image_size'   : (32, 32),
    * 'learning_rate': 0.01,
    * 'moment'       : 0.9,
    * 'display_step' : 100,
    * 'num_epochs'   : 500

* vgg13

    * train_acc: 0.9882
    * test_acc_from_testdataset: 0.9288
    * evel_acc_from_alldataset: 0.9291

    * 'batch_size'   : 128,
    * 'image_size'   : (32, 32),
    * 'learning_rate': 0.01,
    * 'moment'       : 0.9,
    * 'display_step' : 100,
    * 'num_epochs'   : 300

* resnet
    
    * 论文(error)
        
        * 20: 8.75
        * 32: 7.51
        * 44: 7.17
        * 56: 6.97