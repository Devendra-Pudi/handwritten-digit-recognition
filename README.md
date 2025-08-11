# Handwritten Digit Recognition System

A high-accuracy deep learning system for recognizing handwritten digits (0-9) using Convolutional Neural Networks (CNN) and TensorFlow/Keras.

## Features

- **High Accuracy**: Achieves >99% accuracy on MNIST dataset
- **Advanced CNN Architecture**: Multiple convolutional layers with batch normalization and dropout
- **Data Augmentation**: Rotation, zoom, and shift transformations for better generalization
- **Training Optimization**: Learning rate scheduling, early stopping, and model checkpointing
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and classification reports
- **Custom Image Testing**: Support for testing your own handwritten digit images
- **Interactive Interface**: Easy-to-use testing interface

## Model Architecture

The CNN model includes:
- 3 Convolutional blocks with increasing filter sizes (32, 64, 128)
- Batch normalization for stable training
- MaxPooling for dimensionality reduction
- Dropout layers to prevent overfitting
- Dense layers for final classification
- Softmax activation for probability outputs

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Required packages**:
   - TensorFlow >= 2.10.0
   - NumPy >= 1.21.0
   - Matplotlib >= 3.5.0
   - Scikit-learn >= 1.0.0
   - Seaborn >= 0.11.0
   - OpenCV-Python >= 4.5.0
   - Pillow >= 8.0.0

## Usage

### Training the Model

Run the main training script:

```bash
python main.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Build the CNN model
3. Train for 30 epochs with data augmentation
4. Evaluate performance and show metrics
5. Save the trained model as `digit_recognition_model.h5`
6. Generate visualization plots

### Testing the Model

#### Interactive Testing Interface

```bash
python test_model.py
```

This provides an interactive menu with options to:
1. Test custom image files
2. Test random MNIST samples
3. Batch test entire directories
4. Exit

#### Testing Custom Images

```python
from test_model import DigitTester

tester = DigitTester()
prediction, confidence = tester.test_custom_image('path/to/your/digit.png')
print(f"Predicted digit: {prediction} (confidence: {confidence:.3f})")
```

#### Testing MNIST Samples

```python
tester.test_mnist_samples(num_samples=10)  # Test 10 random samples
```

### Using the Model Programmatically

```python
from main import DigitRecognitionModel

# Load pre-trained model
model = DigitRecognitionModel()
model.load_model('digit_recognition_model.h5')

# Predict a digit (image should be 28x28 numpy array)
prediction, confidence = model.predict_digit(image)
print(f"Predicted: {prediction}, Confidence: {confidence:.3f}")
```

## File Structure

```
digit_recognition/
├── main.py              # Main training script
├── test_model.py        # Testing utilities and interactive interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── digit_recognition_model.h5  # Saved model (generated after training)
├── best_digit_model.h5         # Best model checkpoint (generated during training)
├── confusion_matrix.png        # Confusion matrix visualization
└── training_history.png        # Training history plots
```

## Model Performance

Expected performance metrics:
- **Test Accuracy**: >99%
- **Training Time**: ~5-10 minutes (depends on hardware)
- **Model Size**: ~2-3 MB

The model uses several techniques to achieve high accuracy:
- **Data Augmentation**: Improves generalization
- **Batch Normalization**: Speeds up training and improves stability
- **Dropout**: Prevents overfitting
- **Learning Rate Scheduling**: Optimizes convergence
- **Early Stopping**: Prevents overtraining

## Custom Image Guidelines

For best results with custom images:
- **Format**: PNG, JPG, JPEG, BMP, or TIFF
- **Content**: Single digit, clearly written
- **Background**: Preferably white/light background with dark digit
- **Size**: Any size (will be resized to 28x28)
- **Quality**: Clear, not blurry

The preprocessing automatically:
- Converts to grayscale
- Resizes to 28x28 pixels
- Inverts colors if needed (to match MNIST format)
- Normalizes pixel values

## Advanced Usage

### Custom Training Parameters

```python
from main import DigitRecognitionModel

model = DigitRecognitionModel()
model.load_and_preprocess_data()
model.build_cnn_model()

# Custom training with different parameters
model.train_model(epochs=50, batch_size=64)
```

### Model Architecture Customization

You can modify the CNN architecture in the `build_cnn_model()` method to experiment with:
- Different filter sizes
- Additional layers
- Different activation functions
- Alternative optimizers

### Hyperparameter Tuning

Key hyperparameters you can tune:
- Learning rate (default: 0.001)
- Batch size (default: 128)
- Dropout rates (default: 0.25, 0.5)
- Number of filters in each layer
- Data augmentation parameters

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce batch size if running out of memory
   ```python
   model.train_model(batch_size=64)  # Instead of 128
   ```

3. **GPU Issues**: The code automatically uses GPU if available, CPU otherwise

4. **Custom Image Issues**: 
   - Ensure image path is correct
   - Check image format is supported
   - Try different image preprocessing if results are poor

### Performance Tips

- **GPU Acceleration**: Install TensorFlow-GPU for faster training
- **Memory Optimization**: Close other applications during training
- **Batch Size**: Increase batch size if you have more RAM/VRAM

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow/Keras for deep learning framework
- OpenCV for image processing utilities
