# [CVPR 2024] MRFP
Official code implementation of "MRFP: Learning Generalizable Semantic Segmentation from Sim-2-Real with Multi-Resolution Feature Perturbation"

Deep neural networks have shown exemplary performance on semantic scene understanding tasks on source domains, but due to the absence of style diversity during training, enhancing performance on unseen target domains using only single source domain data remains a challenging task. Generation of simulated data is a feasible alternative to retrieving large style-diverse real-world datasets as it is a cumbersome and budget-intensive process. However, the large domain-specific inconsistencies between simulated and real-world data pose a significant generalization challenge in semantic segmentation. In this work, to alleviate this problem, we propose a novel Multi-Resolution Feature Perturbation (MRFP) technique to randomize domain-specific fine-grained features and perturb style of coarse features. Our experimental results on various urban-scene segmentation datasets clearly indicate that, along with the perturbation of style-information, perturbation of fine-feature components is paramount to learn domain invariant robust feature maps for semantic segmentation models. MRFP is a simple and computationally efficient, transferable module with no additional learnable parameters or objective functions, that helps state-of-the-art deep neural networks to learn robust domain invariant features for simulation-to-real semantic segmentation.

Check out our paper [here](https://openaccess.thecvf.com/content/CVPR2024/papers/Udupa_MRFP_Learning_Generalizable_Semantic_Segmentation_from_Sim-2-Real_with_Multi-Resolution_Feature_CVPR_2024_paper.pdf), presentation video [here](https://www.youtube.com/watch?v=63sYr5LoHvo) and the poster [here](https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202024/29708.png?t=1717342777.2902172).

![MRFP_based_architecture](https://github.com/airl-iisc/MRFP/blob/main/pngs/arch.png)

We advise you to use conda environment to run the package. Run the following command to install all the necessary modules:

```sh
conda env create -f SDG.yml 
conda activate solaris_new
```
## Checkpoint
Update the data paths in the 'mypath.py' file.
To run the inference for MRFP+, use the checkpoint provided [here](https://drive.google.com/file/d/1lYDR4bjBUmrUqyTZDpLJDZTRzBiCgtC2/view?usp=sharing). Update the main.py file with the correct MODEL_PATH. Run the main_script.sh file.


To train the model from scratch, follow the comments given in the main.py file. Then run the main_script.sh file.

## Results
![Main Results](https://github.com/airl-iisc/MRFP/blob/main/pngs/mrfp_maintable.png)


![Adverse Weather Results](https://github.com/airl-iisc/MRFP/blob/main/pngs/mrfp_table2.png)


## Citation 

If you find this repo useful for your work, please cite our paper:

```shell
@inproceedings{udupa2024mrfp,
  title={MRFP: Learning Generalizable Semantic Segmentation from Sim-2-Real with Multi-Resolution Feature Perturbation},
  author={Udupa, Sumanth and Gurunath, Prajwal and Sikdar, Aniruddh and Sundaram, Suresh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5904--5914},
  year={2024}
}
```

