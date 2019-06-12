# unsupervised-satellite-image-classfication-based-on-partial-domain-adaptation
machine learning project group 5
# Code for course project Unsupervised Satellite Image Classification based on Partial Adversarial Domain Adaptation
Team member:Hu Jian;  Clinton Elian Gandana;  
Joel Dzidzorvi Kwame Disu;  Chen Junjie; Zheng Cheng  

# Prerequisites
Linux or OSX

NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework (version 0.3.1)

Python 2.7/3.5

# Datasets
We use NWPU-RESISC45 and UC Merced Land dataset in our experiments. 
For our mession, "data/NWPU-RESISC45/NWPU-45.txt" is the source list file and "/data/UCMercedLand-share/UCMerced-19.txt" is the target list file.


# Training and Evaluation
First, you can manually download the PyTorch pre-trained model introduced in `torchvision' library or if you have connected to the Internet, you can automatically downloaded them.
Then, you can train the model for each dataset using the followling command.
```
cd src
python train_pada.py --gpu_id 2 --net ResNet50 --dset NWPU-RESISC45 --s_dset_path ../data/NWPU-RESISC45/NWPU-45.txt --t_dset_path ../data/UCMercedLand-share/UCMerced-19.txt --test_interval 500 --snapshot_interval 5000 --output_dir san1
```
You can set the command parameters to switch between different experiments. 
- "gpu_id" is the GPU ID to run experiments.
- "dset" parameter is the dataset selection. In our experiments, it is NWPU-RESISC45
- "s_dset_path" is the source dataset list.
- "t_dset_path" is the target dataset list.
- "test_interval" is the interval of iterations between two test phase.
- "snapshot_interval" is the interval of iterations between two snapshot models.
- "output_dir" is the output directory of the log and snapshot.
- "net" sets the base network. For details of setting, you can see network.py.
    - For AlexNet, "net" is AlexNet.
    - For VGG, "net" is like VGG16. Detail names are in network.py.
    - For ResNet, "net" is like ResNet50. Detail names are in network.py.
    
# contribution of our members:
Hu Jian:build up the model framework,finish the part of train_PADA_pic.py and loss.py.  
Clinton Elian Gandana:build up the Alexnet network in network.py.  
Joel Dzidzorvi Kwame Disu:foucus on preprocss,finish the pre_process.py.  
Chen Junjie:finish the comparation experiments,mainly focus on train_DaNN_pic.py and data_list.py.  
Zheng Cheng:build up the comparation experiments in network,mainly focus on tran_ResNet_pic.py and lr_schedule.py.
