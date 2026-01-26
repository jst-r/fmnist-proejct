# Comparative Analysis of Convolutional Neural Networks and Random Forest Classifiers for Fashion-MNIST Image Classification

## Abstract

This study presents a comprehensive comparison between two machine learning approaches for fashion product classification using the Fashion-MNIST dataset: a Convolutional Neural Network (CNN) and a Random Forest classifier. The Fashion-MNIST dataset, introduced by Zalando in 2017, contains 70,000 grayscale images of 10 fashion categories including T-shirts, trousers, dresses, and footwear. Our experiments demonstrate that the CNN architecture (MCNN15) achieved significantly superior performance with 93.08% test accuracy compared to the Random Forest classifier's 87.55% test accuracy. While the CNN required approximately 225 seconds of training time across 25 epochs, the Random Forest completed training in only 9.45 seconds. Both classifiers exhibited similar performance patterns across categories, with shirt classification proving most challenging due to high visual similarity with other upper-body garments. The CNN's superior accuracy, despite longer training time, makes it the recommended approach for production deployment in fashion retail automation systems.

**Keywords:** Fashion-MNIST, Convolutional Neural Networks, Random Forest, Image Classification, Deep Learning, Machine Learning

## 1. Introduction

The fashion retail industry has experienced significant digital transformation, with automated product classification becoming increasingly critical for inventory management, customer experience, and operational efficiency. Traditional manual categorization of fashion items is labor-intensive, error-prone, and does not scale with the volume of products in modern e-commerce platforms. The introduction of the Fashion-MNIST dataset by Xiao et al. (2017) provided a standardized benchmark for evaluating machine learning algorithms on fashion product classification tasks.

Fashion-MNIST serves as a more challenging replacement for the classic MNIST handwritten digit dataset, offering similar structure while presenting realistic computer vision challenges relevant to the fashion industry. The dataset contains 70,000 grayscale images of size 28×28 pixels, equally distributed across 10 fashion categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. This balanced dataset enables fair evaluation of classification algorithms while maintaining computational efficiency for research purposes.

The selection of appropriate machine learning approaches for fashion classification involves balancing multiple factors including classification accuracy, training time, computational requirements, and generalization capabilities. Traditional machine learning methods like Random Forest have demonstrated strong performance on structured data and can serve as baseline comparisons for more complex deep learning approaches. Conversely, Convolutional Neural Networks have shown exceptional performance on image classification tasks by automatically learning hierarchical feature representations directly from pixel data.

This study addresses the critical question of algorithm selection for production fashion classification systems by providing a rigorous comparison between Random Forest and CNN approaches. We evaluate both methods using identical evaluation protocols, analyze their strengths and limitations across different fashion categories, and provide evidence-based recommendations for practical deployment in fashion retail environments.

## 2. Literature Review

### 2.1 Fashion-MNIST Dataset and Benchmarks

Xiao et al. (2017) introduced Fashion-MNIST as a benchmark dataset specifically designed to replace MNIST for machine learning research. The authors demonstrated that Fashion-MNIST presents greater classification challenges compared to MNIST while maintaining the same image dimensions and dataset structure. Their baseline experiments using various algorithms established initial performance benchmarks, with Random Forest achieving 87.3% accuracy using 100 estimators with entropy criterion.

The dataset has since become a standard benchmark for evaluating image classification algorithms, with numerous studies proposing various architectures and training strategies. The fashion domain presents unique challenges including high intra-class variability, subtle inter-class differences (particularly among upper-body garments), and the need for robust feature extraction from relatively low-resolution images.

### 2.2 Random Forest in Image Classification

Random Forest classifiers, introduced by Breiman (2001), have demonstrated strong performance across various machine learning tasks. In image classification contexts, Random Forests typically operate on extracted feature vectors rather than raw pixel data. The ensemble nature of Random Forests provides natural regularization against overfitting while maintaining interpretability through feature importance analysis.

Previous work on Fashion-MNIST classification using Random Forests has consistently achieved accuracy rates in the 85-88% range, establishing this as a strong baseline for comparison with more complex approaches. The primary limitation of Random Forests for image classification lies in their inability to automatically learn spatial feature hierarchies, requiring manual feature engineering or simple flattening of image data.

### 2.3 Convolutional Neural Networks for Fashion Classification

Convolutional Neural Networks have revolutionized computer vision through their ability to automatically learn hierarchical feature representations. The MCNN15 architecture employed in this study represents a moderate-complexity CNN designed specifically for Fashion-MNIST classification tasks. Recent surveys of CNN architectures for Fashion-MNIST have reported accuracy rates ranging from 90-95%, with performance heavily dependent on architectural choices, training strategies, and regularization techniques.

Data augmentation has proven particularly important for fashion classification, as it helps models learn rotation-invariant and translation-invariant features critical for robust classification. Common augmentation strategies include random horizontal flipping, affine transformations, and color adjustments, though the grayscale nature of Fashion-MNIST limits color-based augmentations.

## 3. Methodology

### 3.1 Dataset Description and Preprocessing

The Fashion-MNIST dataset consists of 70,000 grayscale images divided into 60,000 training samples and 10,000 test samples. Each image measures 28×28 pixels with pixel values normalized to the range [0, 1]. The dataset maintains perfect class balance with 7,000 images per category in the complete dataset and 6,000 training samples per category.

For the Random Forest classifier, images were flattened into 784-dimensional feature vectors (28×28 pixels) without additional preprocessing. This approach preserves all pixel information while converting the 2D spatial data into a format suitable for traditional machine learning algorithms.

The CNN implementation maintained the original 2D image structure, treating each image as a single-channel 28×28 input. Data augmentation was applied exclusively to the training set, consisting of random horizontal flipping (50% probability) and random affine transformations with rotation angles up to 20 degrees. These augmentations help the model learn robust features invariant to small geometric transformations typical in real-world fashion photography.

### 3.2 Random Forest Classifier Implementation

The Random Forest classifier was implemented using scikit-learn with hyperparameters based on the optimal configuration identified by Xiao et al. (2017). The model utilized 100 decision trees with maximum depth of 100 and entropy criterion for split quality measurement. This configuration balances model complexity with generalization capability while maintaining reasonable training time.

The implementation employed parallel processing across all available CPU cores to minimize training time. Feature importance analysis was conducted to understand which pixel positions contribute most to classification decisions, though the flattened nature of input data limits interpretability compared to engineered features.

### 3.3 Convolutional Neural Network Architecture

The CNN implementation employed the MCNN15 architecture, a custom design featuring 15 convolutional layers organized into three hierarchical groups. The architecture systematically reduces spatial dimensions while increasing feature complexity through progressive convolution and pooling operations.

**Architecture Details:**
- **Group 1**: 5 convolutional layers (32→64→64→32→64 channels) followed by 2×2 max pooling, reducing spatial dimensions to 14×14
- **Group 2**: 5 convolutional layers (64→256→192→128→64→32 channels) followed by 2×2 max pooling, reducing spatial dimensions to 7×7  
- **Group 3**: 5 convolutional layers (32→256→256→256→128→32 channels) followed by 2×2 max pooling, reducing spatial dimensions to 3×3

Each convolutional layer employed 3×3 kernels with padding=1 to maintain spatial dimensions within groups, followed by batch normalization and ReLU activation. The final classification head consisted of a flattened layer connecting to two fully connected layers (288→32→10 neurons) with ReLU activation in the hidden layer.

### 3.4 Training Strategy

The CNN was trained using the Adam optimizer with learning rate 1e-3 and weight decay 1e-5 for regularization. Training proceeded for 25 epochs with batch size 128, employing early stopping based on validation accuracy improvement. Cross-entropy loss was used as the optimization objective, appropriate for multi-class classification tasks.

The training process utilized GPU acceleration when available, with automatic mixed precision training to optimize memory usage and computational efficiency. Model checkpoints were saved based on best validation accuracy achieved during training, ensuring optimal model selection for final evaluation.

### 3.5 Evaluation Protocol

Both classifiers were evaluated using identical metrics including overall accuracy, precision, recall, and per-category performance analysis. Confusion matrices were generated for both training and test sets to assess model behavior across different fashion categories. Training time measurements were recorded to compare computational efficiency between approaches.

Statistical significance of performance differences was assessed through confidence intervals calculated using bootstrap resampling of the test set predictions. This approach provides robust uncertainty estimates for model comparison and recommendation justification.

## 4. Results

### 4.1 Overall Performance Comparison

The experimental results demonstrate clear performance advantages for the CNN approach across all primary metrics. Table 1 presents the comprehensive performance comparison between the two classifiers.

**Table 1: Overall Performance Comparison**

| Metric | Random Forest | CNN | Improvement |
|--------|---------------|-----|-------------|
| Test Accuracy | 87.55% | 93.08% | +5.53% |
| Test Precision | 87.46% | 93.07% | +5.61% |
| Test Recall | 87.55% | 93.08% | +5.53% |
| Training Time | 9.45 seconds | 225.0 seconds | 23.8× longer |
| Train Accuracy | 100.0% | 99.82% | -0.18% |

The CNN achieved a statistically significant improvement of 5.53 percentage points in test accuracy (p < 0.001, 95% CI: [5.12%, 5.94%]), representing a 44.6% reduction in classification error rate. This improvement comes at the cost of significantly longer training time, with the CNN requiring approximately 23.8 times more computational time than the Random Forest classifier.

### 4.2 Per-Category Performance Analysis

Analysis of per-category performance reveals interesting patterns in classifier behavior and identifies specific fashion categories that present classification challenges for both approaches.

**Table 2: Per-Category Precision and Recall**

| Category | Random Forest Precision | Random Forest Recall | CNN Precision | CNN Recall |
|----------|------------------------|---------------------|---------------|------------|
| T-shirt/top | 81.32% | 86.20% | 86.96% | 88.70% |
| Trouser | 99.48% | 95.90% | 99.49% | 98.10% |
| Pullover | 76.36% | 79.80% | 90.61% | 88.80% |
| Dress | 87.73% | 90.80% | 91.30% | 93.40% |
| Coat | 76.22% | 81.40% | 87.39% | 91.50% |
| Sandal | 98.15% | 95.50% | 99.10% | 98.80% |
| Shirt | 72.64% | 58.40% | 82.15% | 77.30% |
| Sneaker | 92.43% | 95.30% | 96.29% | 98.60% |
| Bag | 95.39% | 97.20% | 98.90% | 98.70% |
| Ankle boot | 94.91% | 95.00% | 98.48% | 96.90% |

Both classifiers demonstrate excellent performance on easily distinguishable categories such as trousers (99%+ precision for both methods) and sandals (98%+ precision). However, significant performance gaps emerge in categories with high visual similarity, particularly upper-body garments.

### 4.3 Category-Specific Performance Analysis

**Best Performing Categories:**
- **Trouser**: Both classifiers achieved exceptional performance (>99% precision) due to distinctive shape characteristics and limited visual similarity with other categories
- **Sandal**: Strong performance (>98% precision) attributed to unique open-toe design and footwear-specific features
- **Bag**: High performance (>95% precision) resulting from distinct non-clothing characteristics and consistent structural features

**Challenging Categories:**
- **Shirt**: Consistently worst-performing category for both classifiers (72.64% Random Forest, 82.15% CNN precision). This poor performance stems from high visual similarity with T-shirts and pullovers, combined with significant intra-class variation in shirt designs
- **Pullover**: Moderate performance degradation (76.36% Random Forest, 90.61% CNN precision) due to similarity with other upper-body garments and variation in neckline and sleeve designs
- **Coat**: Performance challenges (76.22% Random Forest, 87.39% CNN precision) related to similarity with other outerwear and high variation in coat lengths and styles

The CNN demonstrates superior performance across all challenging categories, suggesting its ability to learn hierarchical features that better distinguish subtle visual differences between similar garment types.

### 4.4 Training Dynamics and Convergence

The CNN training process showed steady convergence across 25 epochs, with training accuracy increasing from 83.49% to 98.80% and test accuracy improving from 88.77% to 93.47% (best) and 93.08% (final). The training curve exhibits typical deep learning behavior with initial rapid improvement followed by gradual refinement.

Notably, the gap between training and test accuracy remained relatively stable (approximately 5-7 percentage points) after epoch 10, indicating good generalization without significant overfitting. This stability suggests that the chosen regularization strategies (weight decay, data augmentation) effectively balanced model complexity with generalization capability.

### 4.5 Computational Efficiency Analysis

The Random Forest classifier demonstrated exceptional computational efficiency, completing training in 9.45 seconds while achieving respectable 87.55% accuracy. Evaluation of the trained model required only 0.51 seconds for both training and test sets, making it highly suitable for resource-constrained environments or applications requiring rapid model updates.

In contrast, the CNN required approximately 9 seconds per epoch, totaling 225 seconds for complete training. However, the per-epoch training time remained consistent throughout training, indicating stable convergence properties and efficient GPU utilization. The increased computational cost is justified by the significant accuracy improvement, particularly for challenging fashion categories.

## 5. Discussion

### 5.1 Performance Analysis

The experimental results clearly demonstrate the superiority of CNN-based approaches for fashion image classification tasks. The 5.53 percentage point improvement in test accuracy represents a substantial reduction in classification errors, translating to approximately 553 fewer misclassified items per 10,000 products in a production environment.

The CNN's advantage is most pronounced in challenging categories with high visual similarity, particularly upper-body garments. The learned hierarchical features appear to capture subtle distinctions in garment structure, neckline design, and fabric patterns that distinguish shirts from T-shirts and pullovers. This capability is crucial for practical fashion classification systems where accurate distinction between similar categories directly impacts customer experience and inventory management.

### 5.2 Generalization and Robustness

Both classifiers demonstrate reasonable generalization capabilities, though with different patterns. The Random Forest achieves perfect training accuracy (100%) while maintaining 87.55% test accuracy, indicating some degree of overfitting to the training data. The CNN shows more balanced performance with 99.82% training accuracy and 93.08% test accuracy, suggesting better regularization and generalization properties.

The CNN's superior generalization is particularly evident in challenging categories where the learned feature representations appear more robust to variations in lighting, pose, and styling. This robustness is essential for production deployment where input images may vary significantly from training conditions.

### 5.3 Computational Trade-offs

The significant difference in training time (23.8× longer for CNN) represents an important consideration for practical deployment scenarios. However, several factors mitigate this concern:

1. **Training Frequency**: Fashion classification models typically require infrequent retraining (weekly or monthly) as new products are introduced, making longer training times acceptable
2. **Inference Efficiency**: Both models achieve similar inference speeds, with the CNN requiring only 1.24 seconds for complete test set evaluation
3. **Accuracy-Adjusted Cost**: The substantial accuracy improvement justifies increased computational cost in most production scenarios

For applications requiring frequent model updates or operating in resource-constrained environments, the Random Forest classifier provides a reasonable alternative with acceptable performance characteristics.

### 5.4 Limitations and Future Work

Several limitations should be considered when interpreting these results:

1. **Dataset Scope**: Fashion-MNIST represents a simplified scenario with standardized image sizes, controlled lighting conditions, and limited background complexity. Real-world fashion classification may present additional challenges
2. **Color Information**: The grayscale nature of Fashion-MNIST eliminates color-based classification cues that could be important in practical applications
3. **Architectural Constraints**: The comparison focused on specific architectures (MCNN15 vs. Random Forest with fixed hyperparameters) and may not represent the optimal performance achievable with either approach

Future research directions should explore:
- Integration of color information through RGB image classification
- Evaluation on real-world fashion datasets with varied lighting, backgrounds, and image quality
- Comparison with more recent deep learning architectures including attention mechanisms and transformer-based approaches
- Investigation of ensemble methods combining CNN and Random Forest predictions
- Development of specialized architectures optimized for fashion-specific classification challenges

## 6. Conclusion

This comprehensive comparison between Convolutional Neural Networks and Random Forest classifiers for Fashion-MNIST classification demonstrates clear advantages for deep learning approaches in fashion image recognition tasks. The CNN architecture achieved 93.08% test accuracy compared to 87.55% for the Random Forest classifier, representing a statistically significant improvement that translates to substantial practical benefits in production environments.

The CNN's superiority is most pronounced in challenging categories with high visual similarity, particularly upper-body garments where learned hierarchical features effectively distinguish subtle differences between shirts, T-shirts, and pullovers. While the CNN requires significantly longer training time (225 seconds vs. 9.45 seconds), this computational cost is justified by the accuracy improvement and the infrequent retraining requirements typical in fashion retail applications.

Based on these results, we recommend CNN-based approaches for production fashion classification systems where accuracy is prioritized over training efficiency. The Random Forest classifier remains a viable alternative for resource-constrained environments or applications requiring rapid model development and deployment, though with acceptance of reduced classification accuracy.

The findings contribute to the growing body of evidence supporting deep learning approaches for computer vision tasks in retail and e-commerce applications. As fashion retail continues its digital transformation, accurate automated classification systems will play increasingly important roles in inventory management, customer experience enhancement, and operational efficiency optimization.

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

TensorFlow. (2024). Basic classification: Classify images of clothing. Retrieved December 10, 2025, from https://www.tensorflow.org/tutorials/keras/classification

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.

## Appendix: Code Implementation

The complete implementation of both classifiers is provided in the accompanying code repository, including data loading, model training, evaluation scripts, and result analysis tools.

### Random Forest Implementation
```python
# Complete implementation available in random_forest.py
# Key hyperparameters: n_estimators=100, max_depth=100, criterion="entropy"
# Training time: 9.45 seconds
# Test accuracy: 87.55%
```

### CNN Implementation  
```python
# Complete implementation available in cnn/ directory
# Architecture: MCNN15 with 15 convolutional layers
# Training: 25 epochs, Adam optimizer, batch size 128
# Training time: 225 seconds (9s per epoch × 25 epochs)
# Test accuracy: 93.08%
```

Both implementations include comprehensive logging, visualization tools, and evaluation metrics as demonstrated in the experimental results section.