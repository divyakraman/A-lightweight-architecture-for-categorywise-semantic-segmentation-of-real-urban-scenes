## A lightweight architecture for category-wise semantic segmentation of real urban scenes

This repository contains the code for 'A lightweight architecture for category-wise semantic segmentation of real urban scenes'.
<br><br>
Abstract: <br><br>
Current state-of-the-art methods for semantic segmentation use deep architectures which typically have a large number of parameters. In the case of autonomous driving scenes, the image sizes are particularly large and this necessitates large number of GPUs in order to train with a reasonable mini batch size. The main goal of semantic segmentation in the case of autonomous driving is to segment out roads, vehicles, buildings and vegetation. In reality, the autonomous driving system doesn't care whether the vehicle in front of it is a car or a bike. Similarly, whether it's a building or a wall is irrelevant as long as it is conveyed that a hard inanimate structure is present. For safe driving and control in case of incidents like brake failure, it is essential to know the category of object/ scene, trivial information about the sub-category can largely be ignored. 
<br><br>
We use cityscapes' category description to group similar classes. We then propose a lightweight architecture for semantic segmentation. The architecture leverages feature pyramid networks in its first stage to extract feature maps at various scales followed by dilated residual blocks for further refinement. Our experiments show that our method performs well on not just relatively easy datasets such as Cityscapes, but also large and extremely diverse datasets such as Berkeley Deep Drive that are particularly hard to segment.
