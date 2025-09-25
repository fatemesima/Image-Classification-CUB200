# Image-Classification-CUB200

This project is an example of **Image Classification on the CUB-200-2011 dataset** using **Knowledge Distillation**.

---

## 🔹 Project Overview

1. **Teacher Model (EfficientNet-B4)**  
   - Pretrained on ImageNet  
   - Last fully connected layer modified for 200-class output  
   - Trained to classify bird species in the CUB-200 dataset  

2. **Student Model (EfficientNet-B0)**  
   - Smaller and faster  
   - Trained with **Knowledge Distillation** from the Teacher  
   - Combines **Soft Target** (Teacher logits with Temperature) and **Hard Target** (ground truth labels)  

3. **Distillation Loss**  
```python
Loss = alpha * KLDiv(Student_logits, Teacher_logits) + (1-alpha) * CrossEntropy(Student_logits, labels)


| Model           | Method                           | Validation Accuracy   |
| --------------- | -------------------------------- | --------------------- |
| EfficientNet-B0 | Standard Training                | 39%                   |
| EfficientNet-B0 | Knowledge Distillation (from B4) | 68%                   |
| EfficientNet-B4 | Teacher                          | 68%+ (train accuracy) |

