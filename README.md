# [CVPR 2024] MRFP
Official code implementation of "MRFP: Learning Generalizable Semantic Segmentation from Sim-2-Real with Multi-Resolution Feature Perturbation"

Deep neural networks have shown exemplary performance on semantic scene understanding tasks on source domains, but due to the absence of style diversity during training, enhancing performance on unseen target domains using only single source domain data remains a challenging task. Generation of simulated data is a feasible alternative to retrieving large style-diverse real-world datasets as it is a cumbersome and budget-intensive process. However, the large domain-specific inconsistencies between simulated and real-world data pose a significant generalization challenge in semantic segmentation. In this work, to alleviate this problem, we propose a novel MultiResolution Feature Perturbation (MRFP) technique to randomize domain-specific fine-grained features and perturb style of coarse features. Our experimental results on various urban-scene segmentation datasets clearly indicate that, along with the perturbation of style-information, perturbation of fine-feature components is paramount to learn domain invariant robust feature maps for semantic segmentation models. MRFP is a simple and computationally efficient, transferable module with no additional learnable parameters or objective functions, that helps state-of-the-art deep neural networks to learn robust domain invariant features for simulation-to-real semantic segmentation.

Check out our paper [here](https://openaccess.thecvf.com/content/CVPR2024/papers/Udupa_MRFP_Learning_Generalizable_Semantic_Segmentation_from_Sim-2-Real_with_Multi-Resolution_Feature_CVPR_2024_paper.pdf), presentation video [here](https://www.youtube.com/watch?v=63sYr5LoHvo) and the poster [here](https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202024/29708.png?t=1717342777.2902172).




